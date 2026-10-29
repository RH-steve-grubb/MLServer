import asyncio

from typing import cast, TYPE_CHECKING
from collections import defaultdict
from collections.abc import Awaitable, Iterator
from itertools import cycle
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from asyncio import Future

from ..utils import schedule_with_callback, generate_uuid
from ..metrics import REGISTRY

from .errors import WorkerStop, NoWorkersAvailable
from .worker import Worker
from .logging import logger
from .utils import END_OF_QUEUE, cancel_task
from .messages import (
    Message,
    ModelUpdateMessage,
    ModelUpdateType,
    ModelRequestMessage,
    ModelResponseMessage,
)
from prometheus_client import Histogram

QUEUE_METRIC_NAME = "parallel_request_queue"

if TYPE_CHECKING:
    from .zmq import ParentTransport


class AsyncResponses:
    def __init__(self) -> None:
        self._futures: dict[str, Future[ModelResponseMessage]] = {}

        # _workers_map keeps track of which in-flight requests are being served
        # by each worker
        self._workers_map: dict[int, set[str]] = defaultdict(set)
        # _futures_map keeps track of which worker is serving each in-flight
        # request
        self._futures_map: dict[str, int] = {}

        self.parallel_request_queue_size = self._get_or_create_metric()

    def _get_or_create_metric(self) -> Histogram:
        if QUEUE_METRIC_NAME in REGISTRY:
            return REGISTRY[QUEUE_METRIC_NAME]  # type: ignore

        return Histogram(
            QUEUE_METRIC_NAME,
            "counter of request queue size for workers",
            registry=REGISTRY,
        )

    async def schedule_and_wait(
        self, message: Message, worker: Worker
    ) -> ModelResponseMessage:
        """
        Schedule a response and wait until it gets resolved.
        """
        message_id = message.id
        self._schedule(message, worker)
        return await self._wait(message_id)

    def _schedule(self, message: Message, worker: Worker) -> Future:
        """
        Schedule a response to be resolved in the future.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        message_id = message.id
        self._futures[message_id] = future

        # Keep track of allocation for in-flight requests
        self._track_message(message, worker)

        # Monitor current in-flight requests
        in_flight_count = len(self._futures)
        self.parallel_request_queue_size.observe(in_flight_count)

        return future

    def _track_message(self, message: Message, worker: Worker) -> None:
        self._futures_map[message.id] = worker.pid  # type: ignore
        self._workers_map[worker.pid].add(message.id)  # type: ignore

    async def _wait(self, message_id: str) -> ModelResponseMessage:
        future = self._futures[message_id]

        try:
            response_message = await future
            return response_message
        finally:
            self._clear_message(message_id)

    def _clear_message(self, message_id: str) -> None:
        del self._futures[message_id]
        worker_pid = self._futures_map.pop(message_id)
        self._workers_map[worker_pid].remove(message_id)

    def resolve(self, response: ModelResponseMessage):
        """
        Resolve a previously scheduled response future.
        """
        message_id = response.id
        future = self._futures.get(message_id)
        if future is None:
            # Future already cancelled and cleared (e.g. worker crashed after
            # sending its response) — nothing left to resolve
            return

        # NOTE: Use call_soon_threadsafe to cover cases where `model.predict()`
        # (or other methods) get called from a separate thread (and a separate
        # AsyncIO loop)
        loop = future.get_loop()
        loop.call_soon_threadsafe(self._complete, future, response)

    @staticmethod
    def _complete(
        future: Future[ModelResponseMessage], response: ModelResponseMessage
    ) -> None:
        # Cancellation can run after resolve() schedules this callback. Check
        # at completion time, not only when the response first arrives.
        if future.done():
            return
        if response.exception:
            future.set_exception(response.exception)
        else:
            future.set_result(response)

    @staticmethod
    def _fail(future: Future[ModelResponseMessage], error: BaseException) -> None:
        if not future.done():
            future.set_exception(error)

    def fail_all(self, error: BaseException) -> None:
        """Wake inference waiters if the optional response transport fails."""
        for future in list(self._futures.values()):
            future.get_loop().call_soon_threadsafe(self._fail, future, error)

    def cancel(self, worker: Worker, exit_code: int):
        """
        Cancel in-flight requests for worker (e.g. because it died
        unexpectedly).
        """
        in_flight = self._workers_map[worker.pid]  # type: ignore
        if in_flight:
            logger.info(
                f"Cancelling {len(in_flight)} in-flight requests for "
                f"worker {worker.pid} which died unexpectedly with "
                f"exit code {exit_code}..."
            )
        for message_id in in_flight:
            err = WorkerStop(exit_code)
            future = self._futures[message_id]
            loop = future.get_loop()
            loop.call_soon_threadsafe(self._fail, future, err)


class Dispatcher:
    def __init__(self, workers: dict[int, Worker], responses: Queue):
        self._responses = responses
        self._workers = workers
        self._ready_workers = dict(workers)
        self._workers_round_robin = self._reset_round_robin()
        self._worker_starting_lock = asyncio.Lock()
        self._active = False
        self._process_responses_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor()
        self._async_responses = AsyncResponses()
        self._transport: ParentTransport | None = None
        reply = getattr(responses, "_zmq_reply_endpoint", None)
        if reply is not None:
            from .zmq import ParentTransport

            self._transport = ParentTransport(reply)

    def _reset_round_robin(self) -> Iterator[int]:
        worker_pids = list(self._ready_workers.keys())
        self._workers_round_robin = cycle(worker_pids)
        return self._workers_round_robin

    async def on_worker_start(
        self,
        worker: Worker,
        init_coro: Awaitable[None] | None = None,
    ):
        """
        Handler for workers who have just started but are still not ready to
        receive traffic. Holds the lock throughout init_coro (Phase 1/2 model
        replay) so no concurrent dispatch_update can reach the worker in an
        inconsistent state during initialization.
        """
        # Lock while worker is coming up to ensure no model updates get lost in
        # translation
        async with self._worker_starting_lock:
            self._workers[worker.pid] = worker  # type: ignore
            if init_coro is not None:
                await init_coro

    def on_worker_ready(self, worker: Worker):
        """
        Handler for workers who are now ready to receive traffic.
        """
        self._ready_workers[worker.pid] = worker  # type: ignore
        self._reset_round_robin()

    def on_worker_stop(self, worker: Worker, exit_code: int):
        """
        Handler for workers that stopped unexpectedly. Removes the worker from
        the active pool, resets the round-robin iterator, and cancels all
        in-flight requests assigned to that worker.
        """
        pid = worker.pid
        if pid in self._workers:
            del self._workers[pid]

        if pid in self._ready_workers:
            del self._ready_workers[pid]

        self._reset_round_robin()
        if self._transport is not None and pid is not None:
            self._transport.forget_worker(pid, exit_code)
        self._async_responses.cancel(worker, exit_code)

    def start(self):
        self._active = True
        if self._transport is not None:
            self._transport.start(
                self._async_responses.resolve, self._async_responses.fail_all
            )
        self._process_responses_task = schedule_with_callback(
            self._process_responses(), self._process_responses_cb
        )

    def _process_responses_cb(self, process_responses):
        try:
            process_responses.result()
        except asyncio.CancelledError:
            # NOTE: The response loop was cancelled from the outside, so don't
            # restart
            return
        except Exception:
            if not self._active:
                return
            logger.exception("Response processing loop crashed. Restarting the loop...")
            # If process loop crashed, restart it
            self.start()

    async def _process_responses(self):
        logger.debug("Starting response processing loop...")
        loop = asyncio.get_running_loop()
        while self._active:
            response = await loop.run_in_executor(self._executor, self._responses.get)

            # If the queue gets terminated, detect the "sentinel value" and
            # stop reading
            if response is END_OF_QUEUE:
                return

            self._async_responses.resolve(response)

    async def dispatch_request(
        self, request_message: ModelRequestMessage
    ) -> ModelResponseMessage:
        worker, wpid = self._get_worker()
        if self._transport is not None:
            # Register before send: ZeroMQ can suspend and a response may
            # arrive before send returns, unlike the old Queue.put boundary.
            future = self._async_responses._schedule(request_message, worker)
            try:
                await self._transport.send(worker, request_message)
            except BaseException:
                if not future.cancel():
                    if not future.cancelled():
                        future.exception()
                self._async_responses._clear_message(request_message.id)
                raise
            return await self._async_responses._wait(request_message.id)
        worker.send_request(request_message)

        return await self._async_responses.schedule_and_wait(request_message, worker)

    def _get_worker(self) -> tuple[Worker, int]:
        """
        Get next available worker.
        By default, this is just a round-robin through all the workers.
        """
        try:
            worker_pid = next(self._workers_round_robin)
        except StopIteration:
            raise NoWorkersAvailable() from None
        return self._ready_workers[worker_pid], worker_pid

    async def dispatch_update(
        self, model_update: ModelUpdateMessage
    ) -> list[ModelResponseMessage]:
        async with self._worker_starting_lock:
            # Dispatch to all workers concurrently and wait for results
            # to ensure state is properly maintained
            results = await asyncio.gather(
                *[
                    self.dispatch_update_to_worker(worker, model_update)
                    for worker in self._workers.values()
                ],
                return_exceptions=True,
            )
            non_stop = None
            successes = []
            for r in results:
                if isinstance(r, ModelResponseMessage):
                    successes.append(r)
                elif not isinstance(r, WorkerStop):
                    non_stop = non_stop or r

            # If any error other than a worker stop error is returned raise immediately
            if non_stop:
                raise non_stop
            # If no successful results or all results are worker stop errors, this
            # indicates all workers crashed. For load operations we must raise here
            # to perform rollback and maintain a consistent state
            if not successes and model_update.update_type == ModelUpdateType.Load:
                raise NoWorkersAvailable()

            # If at least one result is a success, this means any workers that crashed
            # will properly recover on replay
            return cast(list[ModelResponseMessage], successes)

    async def dispatch_update_to_worker(
        self, worker: Worker, model_update: ModelUpdateMessage
    ) -> ModelResponseMessage:
        # NOTE: Need to rewrite the UUID to ensure each worker sends back a
        # unique result
        worker_update = model_update.copy()
        worker_update.id = generate_uuid()
        worker.send_update(worker_update)
        return await self._async_responses.schedule_and_wait(worker_update, worker)

    async def stop(self):
        self._active = False
        self._ready_workers.clear()
        if self._transport is not None:
            await self._transport.close()
        if self._process_responses_task is not None:
            # Cancelling an executor Future does not stop its Queue.get().
            # Wake the reader before joining it, while the queue is still open.
            if not self._process_responses_task.done():
                self._responses.put(END_OF_QUEUE)
            await cancel_task(self._process_responses_task)
        self._executor.shutdown()
