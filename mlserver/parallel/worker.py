import asyncio
import select
import signal

from asyncio import Task, CancelledError
from collections.abc import Awaitable, Callable, Sequence
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from ..registry import MultiModelRegistry
from ..model import MLModel
from ..utils import install_uvloop_event_loop, schedule_with_callback
from ..logging import configure_logger
from ..settings import Settings
from ..metrics import configure_metrics
from ..context import model_context
from ..env import Environment
from ..errors import ModelNotFound

from .messages import (
    ModelRequestMessage,
    ModelUpdateType,
    ModelUpdateMessage,
    ModelResponseMessage,
)
from .utils import terminate_queue, END_OF_QUEUE
from .logging import logger
from .errors import WorkerError

IGNORED_SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]

WorkerModelHook = Callable[[MLModel], Awaitable[MLModel]]


def _noop():
    pass


class Worker(Process):
    def __init__(
        self,
        settings: Settings,
        responses: Queue,
        env: Environment | None = None,
        on_worker_load: Sequence[WorkerModelHook] = [],
        on_worker_unload: Sequence[WorkerModelHook] = [],
    ):
        super().__init__()
        self._settings = settings
        self._responses = responses
        self._zmq_request_endpoint: str | None = None
        self._zmq_reply_endpoint: str | None = None
        if settings.parallel_inference_transport == "zmq":
            from .zmq import create_endpoint

            reply = getattr(responses, "_zmq_reply_endpoint", None)
            if not isinstance(reply, str) or not reply.startswith("ipc://"):
                raise ValueError("ZeroMQ worker requires its pool's reply endpoint")
            self._zmq_reply_endpoint = reply
            self._zmq_request_endpoint = create_endpoint(Path(reply[6:]).parent, "q")
        self._requests: Queue[ModelRequestMessage] = Queue()
        self._model_updates: Queue[ModelUpdateMessage] = Queue()
        self._env = env
        self._on_worker_load = on_worker_load
        self._on_worker_unload = on_worker_unload

        self.__executor: ThreadPoolExecutor | None = None

    @property
    def _executor(self) -> ThreadPoolExecutor:
        """
        Helper property to initialise a threadpool executor on demand.
        This is required to avoid having to pickle the executor object into a
        separate process.
        """
        if self.__executor is None:
            self.__executor = ThreadPoolExecutor()

        return self.__executor

    def run(self):
        ctx = nullcontext()
        if self._env:
            ctx = self._env

        with ctx:
            install_uvloop_event_loop()
            configure_logger(self._settings)
            configure_metrics(self._settings)
            self._ignore_signals()
            asyncio.run(self.coro_run())

    def _ignore_signals(self):
        """
        Uvloop will try to propagate the main process' signals to the
        underlying workers.
        However, this would just kill off the workers without any cleaning.
        To avoid this, and be able to properly shut them down, we forcefully
        ignore the signals coming from the main parent process.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running yet, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for sign in IGNORED_SIGNALS:
            # Ensure that signal handlers are a no-op, to let the main process
            # take care of cleaning up workers
            loop.add_signal_handler(sign, _noop)

    def __inner_init__(self):
        """
        Internal __init__ method that needs to run within the worker process.
        """
        self._model_registry = MultiModelRegistry(
            on_model_load=self._on_worker_load,
            on_model_unload=self._on_worker_unload,
        )
        self._active = True
        self._poller = select.poll()
        self._poller.register(self._requests._reader, select.POLLIN)
        self._poller.register(self._model_updates._reader, select.POLLIN)

    async def coro_run(self):
        self.__inner_init__()
        if self._zmq_request_endpoint is None:
            await self._run_control()
            return

        from .zmq import serve_requests

        control = asyncio.create_task(self._run_control())
        inference = asyncio.create_task(serve_requests(self))
        try:
            done, _ = await asyncio.wait(
                (control, inference), return_when=asyncio.FIRST_COMPLETED
            )
            if control in done:
                await control
            else:
                await inference
                raise RuntimeError("ZeroMQ worker inference loop stopped unexpectedly")
        finally:
            self._active = False
            # Cancelling the asyncio wrapper cannot interrupt select.poll in
            # its executor thread. Wake that reader before joining it, or a
            # failed data loop leaves this process alive at interpreter exit.
            if not control.done():
                await terminate_queue(self._model_updates)
            control.cancel()
            inference.cancel()
            await asyncio.gather(control, inference, return_exceptions=True)
            if self.__executor is not None:
                self.__executor.shutdown()

    async def _run_control(self) -> None:
        # Retain the original queue loop for model updates and shutdown. The
        # ZeroMQ data loop has its own process-local sockets and lifetime.
        loop = asyncio.get_running_loop()
        # Multiprocessing exposes these descriptors for polling, although its
        # public Queue type stubs omit the private implementation attributes.
        request_reader = getattr(self._requests, "_reader")
        update_reader = getattr(self._model_updates, "_reader")
        while self._active:
            readable = await loop.run_in_executor(self._executor, self._select)
            for r in readable:
                if r is request_reader:
                    request = self._requests.get()

                    schedule_with_callback(
                        self._process_request(request), self._handle_response
                    )
                elif r is update_reader:
                    model_update = self._model_updates.get()
                    # If the queue gets terminated, detect the "sentinel value"
                    # and stop reading
                    if model_update is END_OF_QUEUE:
                        self._active = False
                        return

                    schedule_with_callback(
                        self._process_model_update(model_update),
                        self._handle_response,
                    )

    def _select(self):
        ready = {fd for fd, event in self._poller.poll() if event & select.POLLIN}
        readable = []
        for r in [self._requests._reader, self._model_updates._reader]:
            try:
                if r.fileno() in ready:
                    readable.append(r)
            except (ValueError, AttributeError):
                pass
        return readable

    async def _process_request(self, request) -> ModelResponseMessage:
        if self._settings.rest_worker_json:
            from ..rest.worker import RAW_INFER_METHOD, process_raw_request

            if request.method_name == RAW_INFER_METHOD:
                return await process_raw_request(self, request)
        try:
            model = await self._model_registry.get_model(
                request.model_name, request.model_version
            )

            method = getattr(model, request.method_name)
            with model_context(model.settings):
                return_value = await method(
                    *request.method_args, **request.method_kwargs
                )
            return ModelResponseMessage(id=request.id, return_value=return_value)
        except (Exception, CancelledError) as e:
            logger.exception(
                f"An error occurred calling method '{request.method_name}' "
                f"from model '{request.model_name}'."
            )
            worker_error = WorkerError(e)
            return ModelResponseMessage(id=request.id, exception=worker_error)

    def _handle_response(self, process_task: Task):
        response_message = process_task.result()
        self._responses.put(response_message)

    async def _process_model_update(
        self, update: ModelUpdateMessage
    ) -> ModelResponseMessage:
        try:
            model_settings = update.model_settings
            if update.update_type == ModelUpdateType.Load:
                await self._model_registry.load(model_settings, True)
            elif update.update_type == ModelUpdateType.Unload:
                try:
                    await self._model_registry.unload_version(
                        model_settings.name, model_settings.version, update.rollback
                    )
                except ModelNotFound:
                    # Model not present on worker - idempotent operation
                    pass
            else:
                logger.warning(
                    "Unknown model update message with type %s", update.update_type
                )

            return ModelResponseMessage(id=update.id)
        except (Exception, CancelledError) as e:
            logger.exception(
                "An error occurred processing a model update "
                f"of type '{update.update_type.name}'."
            )
            worker_error = WorkerError(e)
            return ModelResponseMessage(id=update.id, exception=worker_error)

    def send_request(self, request_message: ModelRequestMessage):
        """
        Send an inference request message to the worker.
        Note that this method should be both multiprocess- and thread-safe.
        """
        self._requests.put(request_message)

    def send_update(self, model_update: ModelUpdateMessage):
        """
        Send a model update to the worker.
        Note that this method should be both multiprocess- and thread-safe.
        """
        self._model_updates.put(model_update)

    async def stop(self):
        """
        Close the worker's main loop.
        Note that this method should be both multiprocess- and thread-safe.
        """
        await terminate_queue(self._model_updates)
        self._model_updates.close()
        self._requests.close()
        self._executor.shutdown()
