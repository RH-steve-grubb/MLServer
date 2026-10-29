import asyncio
import random

from contextlib import nullcontext
from weakref import WeakSet
from multiprocessing import Queue
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Awaitable, Callable, Iterable, Sequence
from prometheus_client import Counter

from ..model import MLModel
from ..registry import _model_load_hooks_active
from ..utils import defer_cancellation, with_operation_lock, create_task_independent
from ..types import InferenceRequest, InferenceResponse
from ..settings import Settings, ModelSettings
from ..env import Environment

from .model import ParallelModel
from .errors import InferencePoolUnavailable
from .worker import Worker, WorkerModelHook
from .logging import logger
from .utils import configure_inference_pool, terminate_queue
from .messages import (
    ModelResponseMessage,
    ModelUpdateMessage,
    ModelUpdateType,
)
from .dispatcher import Dispatcher


PredictMethod = Callable[[InferenceRequest], Awaitable[InferenceResponse]]
InferencePoolHook = Callable[[Worker], Awaitable[None]]


def _spawn_worker(
    settings: Settings,
    responses: Queue,
    env: Environment | None,
    on_worker_load: Sequence[WorkerModelHook],
    on_worker_unload: Sequence[WorkerModelHook],
) -> Worker:
    with env or nullcontext():
        worker = Worker(settings, responses, env, on_worker_load, on_worker_unload)
        worker.start()

    return worker


class WorkerRegistry:
    """
    Simple registry to keep track of which models have been loaded.
    This can be used to re-load all models when a worker stops unexpectedly.
    """

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelSettings] = {}

    def _key(self, model_settings: ModelSettings) -> tuple[str, str]:
        return (model_settings.name, model_settings.version or "")

    def add(self, model_settings: ModelSettings):
        model_key = self._key(model_settings)
        self._models[model_key] = model_settings.model_copy(deep=True)

    def remove(self, model_settings: ModelSettings):
        model_key = self._key(model_settings)
        if model_key in self._models:
            del self._models[model_key]

    def has_model(self, model_settings: ModelSettings) -> bool:
        return self._key(model_settings) in self._models

    def clear(self) -> None:
        self._models.clear()

    def __len__(self) -> int:
        return len(self._models)

    @property
    def models(self) -> Iterable[ModelSettings]:
        return self._models.values()


class InferencePool:
    """
    The InferencePool class represents a pool of workers where we can run
    inference on.

    Under the hood, it's responsible for managing a pool of multiprocessing
    workers, where the model is loaded.
    This approach lets MLServer work around the GIL to make sure that inference
    can occur in parallel across multiple models or instances of a model.
    """

    # Shared counter across all instances - created on first use (lazy init
    # required because PROMETHEUS_MULTIPROC_DIR must be set before any metric
    # objects are created, and that happens at server start, not import time)
    _PoolCleanupFailuresTotal = None

    @classmethod
    def _increment_cleanup_failure_metric(cls, pool_id: str | None):
        if cls._PoolCleanupFailuresTotal is None:
            cls._PoolCleanupFailuresTotal = Counter(
                "pool_cleanup_failures_total",
                "Total number of inference pool cleanup failures",
                ["pool_id"],
            )
        cls._PoolCleanupFailuresTotal.labels(pool_id=pool_id or "").inc()

    def __init__(
        self,
        settings: Settings,
        env: Environment | None = None,
        pool_gid: str | None = None,
        on_worker_stop: Sequence[InferencePoolHook] = [],
        on_worker_load: Sequence[WorkerModelHook] = [],
        on_worker_unload: Sequence[WorkerModelHook] = [],
    ):
        configure_inference_pool(settings)

        self._on_worker_stop = on_worker_stop
        self._on_worker_load = on_worker_load
        self._on_worker_unload = on_worker_unload
        self._env = env
        self._pool_gid = pool_gid
        self._workers: dict[int, Worker] = {}
        self._worker_registry = WorkerRegistry()
        self._pending_reload: dict[tuple[str, str], MLModel] = {}
        self._closing = False
        self._close_task: asyncio.Task | None = None
        self._closing_workers: dict[int, Worker] = {}
        self._restart_tasks: set[asyncio.Task] = set()
        self._recovery_task: asyncio.Task | None = None
        self._staged: dict[ParallelModel, asyncio.Future] = {}
        self._quarantined_workers: dict[int, Worker] = {}
        self._retired_models: WeakSet[MLModel] = WeakSet()
        self._operation_lock = asyncio.Lock()
        self._settings = settings
        self._responses: Queue[ModelResponseMessage] = Queue()
        self._transport_directory: TemporaryDirectory[str] | None = None
        if settings.parallel_inference_transport == "zmq":
            from .zmq import create_endpoint

            # Only endpoint strings cross spawn. The parent owns the private
            # directory until workers and the response transport have stopped.
            self._transport_directory = TemporaryDirectory(prefix="mlserver-ipc-")
            reply = create_endpoint(Path(self._transport_directory.name), "r")
            setattr(self._responses, "_zmq_reply_endpoint", reply)
        for _ in range(self._settings.parallel_workers):
            worker = _spawn_worker(
                self._settings,
                self._responses,
                self._env,
                self._on_worker_load,
                self._on_worker_unload,
            )
            self._workers[worker.pid] = worker  # type: ignore

        self._dispatcher = Dispatcher(
            self._workers,
            self._responses,
            model_operation_timeout=self._settings.model_operation_timeout,
            worker_start_timeout=self._settings.worker_start_timeout,
        )
        self._dispatcher.start()

    @property
    def env_hash(self) -> str | None:
        if not self._env:
            return None

        return self._env.env_hash

    @property
    def pool_gid(self) -> str | None:
        return self._pool_gid

    @property
    def name(self) -> str:
        if self.env_hash:
            return f"inference pool with hash '{self.env_hash}'"

        if self._pool_gid:
            return f"inference pool with GID '{self._pool_gid}'"

        return "default inference pool"

    async def on_worker_stop(self, pid: int, exit_code: int):
        # If the inference pool is closing, or the current worker
        # is not in this inference pool, worker stop handling
        # should be skipped
        if pid in self._closing_workers:
            self._closing_workers[pid].record_exit(exit_code)
            return
        if pid not in self._workers:
            return

        worker = self._workers[pid]
        worker.record_exit(exit_code)
        logger.warning(
            f"Worker with PID {worker.pid} on {self.name} stopped "
            f"unexpectedly with exit code {exit_code}. "
            "Triggering worker restart..."
        )

        # Unregister worker from the dispatcher and inference pool
        self._dispatcher.on_worker_stop(worker, exit_code)
        if pid in self._workers:
            # NOTE: worker may be removed by dispatcher
            del self._workers[pid]

        try:
            # Best effort execution of on worker stop hooks
            results = await asyncio.gather(
                *[callback(worker) for callback in self._on_worker_stop],
                return_exceptions=True,
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            if failures:
                # Track cleanup failures
                self._increment_cleanup_failure_metric(self.env_hash or self._pool_gid)
                logger.error(
                    f"{len(failures)} worker stop hook(s) failed "
                    f"for PID {pid} on {self.name}",
                )
                raise failures[0]
        finally:
            # The child has already been reaped by the SIGCHLD handler. Clean
            # up the parent-side queue resources before discarding the worker.
            worker.close_queues()

            # Restart independently of the signal handler and its inherited
            # timeout context. Keep each owner until its result is observed.
            if not self._closing:
                task = create_task_independent(self._safe_start_worker())
                self._restart_tasks.add(task)
                task.add_done_callback(self._restart_tasks.discard)

    async def _safe_start_worker(self):
        """
        Fire-and-forget worker restart for use with asyncio.create_task.
        Logs failures rather than raising.
        """
        # Jitter before spawning: rate-limits rapid fork cycles and guarantees
        # a lock-free window so operator unloads can break a crash loop
        await asyncio.sleep(random.uniform(0.5, 2.0))
        try:
            await self._start_worker()
        except Exception:
            logger.error(
                f"Failed to start replacement worker on {self.name}",
                exc_info=True,
            )

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.worker_start_timeout
    )
    async def _start_worker(self) -> Worker | None:
        # Do not start a worker if the pool is closing
        if self._closing:
            return None
        if self._staged:
            await asyncio.gather(*[asyncio.shield(f) for f in self._staged.values()])
        await self._settle_recovery()
        if self._closing:
            return None
        worker = _spawn_worker(
            self._settings,
            self._responses,
            self._env,
            self._on_worker_load,
            self._on_worker_unload,
        )
        logger.info(f"Starting new worker with PID {worker.pid} on {self.name}...")

        # Phase 1/2 replay runs inside the dispatcher lock via on_worker_start
        # to ensure no concurrent dispatch_update can reach the replacement
        # worker in an inconsistent state during initialization
        try:
            await self._dispatcher.on_worker_start(worker, self._replay_worker(worker))
        except (Exception, asyncio.CancelledError):
            # Replay can fail through an operation error, timeout, or
            # cancellation. In every case the replacement worker is not safe
            # to publish as ready.
            self._dispatcher.on_worker_stop(worker, -1)
            assert worker.pid is not None
            self._closing_workers[worker.pid] = worker
            worker.kill()
            await asyncio.to_thread(
                worker.join, self._settings.parallel_workers_timeout
            )
            if worker.exitcode is None:
                logger.error(f"Replay worker {worker.pid} has not confirmed exit")
            else:
                worker.close_queues()
                self._closing_workers.pop(worker.pid, None)
            raise

        if not self._closing:
            self._dispatcher.on_worker_ready(worker)
            logger.info(
                f"New worker with PID {worker.pid} on {self.name} is now ready."
            )
        else:
            worker.kill()
            return None
        return worker

    async def _replay_worker(self, worker: Worker):
        """
        Replay all model state to a replacement worker. Runs inside the
        dispatcher lock (via on_worker_start) so no concurrent dispatch_update
        can interleave with initialization.
        """
        # If the pool started closing while waiting for the lock, skip replay.
        if self._closing:
            return

        # Phase 1: load all committed models from the worker registry
        load_results = await asyncio.gather(
            *[
                self._dispatcher.dispatch_update_to_worker(
                    worker,
                    ModelUpdateMessage(
                        update_type=ModelUpdateType.Load,
                        model_settings=model_settings,  # type: ignore
                    ),
                )
                for model_settings in self._worker_registry.models
            ],
            return_exceptions=True,
        )
        load_failures = [r for r in load_results if isinstance(r, BaseException)]
        if load_failures:
            raise load_failures[0]

        # Phase 2: load all in-progress reload models so the replacement
        # mirrors the staged state of existing workers
        reload_results = await asyncio.gather(
            *[
                self._dispatcher.dispatch_update_to_worker(
                    worker,
                    ModelUpdateMessage(
                        update_type=ModelUpdateType.Load,
                        model_settings=model.settings,  # type: ignore
                    ),
                )
                for model in self._pending_reload.values()
            ],
            return_exceptions=True,
        )
        reload_failures = [r for r in reload_results if isinstance(r, BaseException)]
        if reload_failures:
            raise reload_failures[0]

    def _model_key(self, model_settings: ModelSettings) -> tuple[str, str]:
        return (model_settings.name, model_settings.version or "")

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.model_operation_timeout
    )
    async def load_model(self, model: MLModel) -> MLModel:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        await self._settle_recovery()
        reload = self.has_model(model.settings)
        # Replay only committed fresh loads. Reloads are staged only after
        # every live worker acknowledges preparation.
        try:
            load_message = ModelUpdateMessage(
                update_type=ModelUpdateType.Load,
                model_settings=model.settings,  # type: ignore
                rollback=reload,
            )
            await self._dispatch_owned(load_message)
        except (Exception, asyncio.CancelledError):
            self._retire(model)
            raise
        if self._closing:
            self._retire(model)
            raise InferencePoolUnavailable(self.name)
        self._retired_models.discard(model)
        parallel_model = self._wrap(model)
        if _model_load_hooks_active.get():
            # The pool hook is preparation, not frontend publication. Replay
            # waits for the registry to finish every subsequent hook as well.
            self._staged[parallel_model] = asyncio.get_running_loop().create_future()
            parallel_model._on_load_complete = lambda: self._commit_model(
                parallel_model, reload
            )
            parallel_model._on_load_failed = lambda: self._abandon_model(
                parallel_model, load_message
            )
        else:
            self._commit_model(parallel_model, reload)
        return parallel_model

    def _release_stage(self, model: ParallelModel) -> None:
        future = self._staged.pop(model, None)
        if future is not None and not future.done():
            future.set_result(None)

    def _commit_model(self, model: ParallelModel, reload: bool) -> None:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        if reload:
            self._pending_reload[self._model_key(model.settings)] = model
            if model in self._staged:
                model.ready = True
        else:
            self._worker_registry.add(model.settings)
        self._release_stage(model)

    def _abandon_model(self, model: ParallelModel, message: ModelUpdateMessage) -> None:
        if model not in self._staged:
            return
        self._retire(model)
        dispatch = asyncio.get_running_loop().create_future()
        dispatch.set_result(None)
        self._begin_recovery(dispatch, message)
        self._release_stage(model)

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.model_operation_timeout
    )
    async def unload_model(self, model: MLModel) -> MLModel:
        if self._closing:
            raise InferencePoolUnavailable(self.name)
        await self._settle_recovery()
        if model in self._retired_models:
            return self._wrap(model)
        key = self._model_key(model.settings)
        pending_model = self._pending_reload.pop(key, None)
        rollback = pending_model is not None and not pending_model.ready
        if pending_model is not None and not rollback:
            # This is the publication point for replay. A later failure while
            # releasing the old instance must preserve the new settings.
            self._worker_registry.add(pending_model.settings)
        elif pending_model is None:
            self._worker_registry.remove(model.settings)
        self._retire(model)
        unload_message = ModelUpdateMessage(
            update_type=ModelUpdateType.Unload,
            model_settings=model.settings,  # type: ignore
            rollback=rollback,
        )
        await self._dispatch_owned(unload_message)

        # Wrap the model in the ParallelModel class so that the main process
        # does not load/unload the model (only done on workers)
        if isinstance(model, ParallelModel):
            return model
        return ParallelModel(model, self._dispatcher)

    def _wrap(self, model: MLModel) -> ParallelModel:
        return (
            model
            if isinstance(model, ParallelModel)
            else ParallelModel(model, self._dispatcher)
        )

    def _retire(self, model: MLModel) -> None:
        self._retired_models.add(model)
        if isinstance(model, ParallelModel):
            self._retired_models.add(model._model)

    async def _settle_recovery(self) -> None:
        while self._recovery_task is not None:
            task = self._recovery_task
            await asyncio.shield(task)
            if self._recovery_task is task:
                self._recovery_task = None

    async def _dispatch_owned(self, message: ModelUpdateMessage) -> None:
        # A frontend deadline cannot cancel an update already delivered to an
        # independent process. Keep the response waiter until it settles.
        dispatch = asyncio.create_task(self._dispatcher.dispatch_update(message))
        try:
            await asyncio.shield(dispatch)
        except (Exception, asyncio.CancelledError):
            self._begin_recovery(dispatch, message)
            raise

    def _begin_recovery(
        self, dispatch: asyncio.Future, message: ModelUpdateMessage
    ) -> None:
        self._quarantined_workers.update(self._dispatcher._ready_workers)
        self._dispatcher._ready_workers.clear()
        self._dispatcher._reset_round_robin()
        previous = self._recovery_task

        async def recover():
            if previous is not None:
                await previous
            await self._recover_update(dispatch, message)

        self._recovery_task = create_task_independent(recover())
        self._recovery_task.add_done_callback(self._recovery_done)

    def _recovery_done(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            logger.error(
                f"Recovery of {self.name} failed; pool remains quarantined "
                "and owned until close.",
                exc_info=None if task.cancelled() else task.exception(),
            )

    async def _recover_update(
        self,
        dispatch: asyncio.Future,
        message: ModelUpdateMessage,
    ) -> None:
        failed = None
        try:
            await dispatch
        except (Exception, asyncio.CancelledError) as error:
            failed = error
        if self._closing:
            return
        if message.update_type == ModelUpdateType.Load:
            # Settlement alone is insufficient: partial success may have left
            # staged instances on some workers. Restore the old committed state
            # with acknowledged cleanup before allowing any routing or replay.
            await self._dispatcher.dispatch_update(
                ModelUpdateMessage(
                    update_type=ModelUpdateType.Unload,
                    model_settings=message.model_settings,  # type: ignore
                    rollback=message.rollback,
                )
            )
        elif failed is not None:
            # An unacknowledged unload is unsafe to replay by name: a second
            # unload could remove the replacement which the first committed.
            # Preserve ownership and the failure instead of guessing state.
            raise failed
        if not self._closing and asyncio.current_task() is self._recovery_task:
            for pid, worker in self._quarantined_workers.items():
                if pid in self._workers:
                    self._dispatcher.on_worker_ready(worker)
            self._quarantined_workers.clear()

    def has_model(self, model_settings: ModelSettings) -> bool:
        return self._worker_registry.has_model(model_settings)

    def empty(self) -> bool:
        return len(self._worker_registry) == 0

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            for model in list(self._staged):
                self._release_stage(model)
            self._close_task = create_task_independent(self._close())
            self._close_task.add_done_callback(self._close_done)
        await asyncio.shield(self._close_task)

    def _close_done(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            logger.error(
                f"Close of {self.name} failed; result and resources retained.",
                exc_info=None if task.cancelled() else task.exception(),
            )

    async def _close(self) -> None:
        # Existing accepted operations get their configured window to settle.
        # If they stall, closure escalates to process exit rather than waiting
        # forever on a frontend lifecycle lock.
        try:
            await asyncio.wait_for(
                self._operation_lock.acquire(), self._settings.model_operation_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Lifecycle lock did not drain while closing {self.name}")
        else:
            self._operation_lock.release()

        # Best effort cleanup
        cleanup_errors: list[BaseException] = []
        try:
            await self._close_workers()
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to fully cleanup workers on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            await terminate_queue(self._responses)
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to terminate response queue on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            await self._dispatcher.stop(self._settings.parallel_workers_timeout)
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to stop dispatcher on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            # The dispatcher's executor may still be reading this handle.
            # Stop and join that reader before closing the queue.
            self._responses.close()
        except (Exception, asyncio.CancelledError) as e:
            logger.error(
                f"Failed to close response queue on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        if self._recovery_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._recovery_task),
                    self._settings.model_operation_timeout,
                )
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
        if self._restart_tasks:
            results = await asyncio.gather(*self._restart_tasks, return_exceptions=True)
            cleanup_errors.extend(r for r in results if isinstance(r, BaseException))

        # The pool is terminal after close. No model state can be replayed or
        # rolled back, so release the tracking references after cleanup.
        self._worker_registry.clear()
        self._pending_reload.clear()

        if cleanup_errors:
            # Track cleanup failures
            self._increment_cleanup_failure_metric(self.env_hash or self._pool_gid)
            raise cleanup_errors[0]

        if self._transport_directory is not None:
            # Its endpoints are resources of the workers and dispatcher. Retain
            # their directory too if cleanup did not confirm their release.
            self._transport_directory.cleanup()

    async def _close_workers(self):
        cleanup_errors: list[BaseException] = []
        self._closing_workers.update(self._workers)
        for pid, worker in list(self._closing_workers.items()):
            # Quarantine first, but keep a resource-owning reference until exit
            # has actually been observed. A kill request is not confirmation.
            self._dispatcher.on_worker_stop(worker, -1)
            try:
                await asyncio.wait_for(
                    worker.stop(), self._settings.parallel_workers_timeout
                )
                await asyncio.to_thread(
                    worker.join, self._settings.parallel_workers_timeout
                )
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
            try:
                if worker.exitcode is None:
                    worker.kill()
                    await asyncio.to_thread(
                        worker.join, self._settings.parallel_workers_timeout
                    )
                if worker.exitcode is None:
                    raise RuntimeError(f"Worker {pid} has not confirmed exit")
                worker.close_queues()
                del self._closing_workers[pid]
            except (Exception, asyncio.CancelledError) as error:
                cleanup_errors.append(error)
                continue

            results = await asyncio.gather(
                *[
                    asyncio.wait_for(
                        callback(worker), self._settings.parallel_workers_timeout
                    )
                    for callback in self._on_worker_stop
                ],
                return_exceptions=True,
            )
            cleanup_errors.extend(r for r in results if isinstance(r, BaseException))
        if cleanup_errors:
            raise cleanup_errors[0]
