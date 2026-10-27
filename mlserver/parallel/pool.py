import asyncio
import random

from contextlib import nullcontext
from multiprocessing import Queue
from collections.abc import Awaitable, Callable, Iterable, Sequence
from prometheus_client import Counter

from ..model import MLModel
from ..types import InferenceRequest, InferenceResponse
from ..settings import Settings, ModelSettings
from ..env import Environment

from .model import ParallelModel
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
        self._models[model_key] = model_settings

    def remove(self, model_settings: ModelSettings):
        model_key = self._key(model_settings)
        if model_key in self._models:
            del self._models[model_key]

    def has_model(self, model_settings: ModelSettings) -> bool:
        return self._key(model_settings) in self._models

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
        self._settings = settings
        self._responses: Queue[ModelResponseMessage] = Queue()
        for _ in range(self._settings.parallel_workers):
            worker = _spawn_worker(
                self._settings,
                self._responses,
                self._env,
                self._on_worker_load,
                self._on_worker_unload,
            )
            self._workers[worker.pid] = worker  # type: ignore

        self._dispatcher = Dispatcher(self._workers, self._responses)
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
        if pid not in self._workers:
            return

        worker = self._workers[pid]
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
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                # Track cleanup failures
                self._increment_cleanup_failure_metric(self.env_hash or self._pool_gid)
                logger.error(
                    f"{len(failures)} worker stop hook(s) failed "
                    f"for PID {pid} on {self.name}",
                )
                raise failures[0]
        finally:
            # create_task is intentional: allows the SIGCHLD handler to process
            # all crashed PIDs immediately rather than blocking on each
            # replacement's replay. The task is kept alive by the event loop's
            # execution machinery (Handle → lock waiter → gather futures)
            # throughout its lifetime — no GC risk.
            if not self._closing:
                asyncio.create_task(self._safe_start_worker())  # noqa: RUF006

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

    async def _start_worker(self) -> Worker | None:
        # Do not start a worker if the pool is closing
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
        await self._dispatcher.on_worker_start(worker, self._replay_worker(worker))

        if not self._closing:
            self._dispatcher.on_worker_ready(worker)
            logger.info(
                f"New worker with PID {worker.pid} on {self.name} is now ready."
            )
        return worker

    async def _replay_worker(self, worker: Worker):
        """
        Replay all model state to a replacement worker. Runs inside the
        dispatcher lock (via on_worker_start) so no concurrent dispatch_update
        can interleave with initialization.
        """
        # If the pool started closing while waiting for the lock, kill the
        # spawned worker rather than leaving it orphaned
        if self._closing:
            worker.kill()
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
        load_failures = [r for r in load_results if isinstance(r, Exception)]
        if load_failures:
            worker.kill()
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
        reload_failures = [r for r in reload_results if isinstance(r, Exception)]
        if reload_failures:
            worker.kill()
            raise reload_failures[0]

    def _model_key(self, model_settings: ModelSettings) -> tuple[str, str]:
        return (model_settings.name, model_settings.version or "")

    async def load_model(self, model: MLModel) -> MLModel:
        # Check for reload
        reload = self.has_model(model.settings)

        # Register the model prior to dispatch
        # This allows for rollback for load failure or worker crash
        if reload:
            self._pending_reload[self._model_key(model.settings)] = model
        else:
            self._worker_registry.add(model.settings)

        try:
            load_message = ModelUpdateMessage(
                update_type=ModelUpdateType.Load,
                model_settings=model.settings,  # type: ignore
            )
            await self._dispatcher.dispatch_update(load_message)
        except Exception:
            # If load fails, only unregister for fresh load
            if not reload:
                self._worker_registry.remove(model.settings)
            raise

        # Wrap the model in the ParallelModel class so that the main process
        # does not load/unload the model (only done on workers)
        if isinstance(model, ParallelModel):
            return model

        parallel_model = ParallelModel(model, self._dispatcher)
        if reload:
            # Ensure the model tracked in pending reload list is the same
            # as the returned instance
            self._pending_reload[self._model_key(model.settings)] = parallel_model
        return parallel_model

    async def unload_model(self, model: MLModel) -> MLModel:
        # Unregister any pending reloads - unload overrides them
        pending_model = self._pending_reload.pop(self._model_key(model.settings), None)

        # Check rollback
        rollback = False
        if pending_model:
            if pending_model.ready:
                # If a pending model exists with ready status, this indicates
                # successful reload load, so we register the new model and
                # unload the old model
                self._worker_registry.add(pending_model.settings)
            else:
                # If a pending model exists with not ready status, this indicates
                # failed reload load, so we keep the old model registered and
                # perform rollback on the new model
                rollback = True
        else:
            # If there is no pending model then this indicates non-reload
            # Unregister and unload the model
            self._worker_registry.remove(model.settings)

        unload_message = ModelUpdateMessage(
            update_type=ModelUpdateType.Unload,
            model_settings=model.settings,  # type: ignore
            rollback=rollback,
        )
        await self._dispatcher.dispatch_update(unload_message)

        # Wrap the model in the ParallelModel class so that the main process
        # does not load/unload the model (only done on workers)
        if isinstance(model, ParallelModel):
            return model
        return ParallelModel(model, self._dispatcher)

    def has_model(self, model_settings: ModelSettings) -> bool:
        return self._worker_registry.has_model(model_settings)

    def empty(self) -> bool:
        return len(self._worker_registry) == 0

    async def close(self):
        if self._closing:
            return
        self._closing = True

        # Best effort cleanup
        cleanup_errors = []
        try:
            await self._close_workers()
        except Exception as e:
            logger.error(
                f"Failed to fully cleanup workers on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            await terminate_queue(self._responses)
        except Exception as e:
            logger.error(
                f"Failed to terminate response queue on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            await self._dispatcher.stop()
        except Exception as e:
            logger.error(
                f"Failed to stop dispatcher on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        try:
            # The dispatcher's executor may still be reading this handle.
            # Stop and join that reader before closing the queue.
            self._responses.close()
        except Exception as e:
            logger.error(
                f"Failed to close response queue on {self.name}",
                exc_info=True,
            )
            cleanup_errors.append(e)

        if cleanup_errors:
            # Track cleanup failures
            self._increment_cleanup_failure_metric(self.env_hash or self._pool_gid)
            raise cleanup_errors[0]

    async def _close_workers(self):
        cleanup_errors = []
        for pid, worker in list(self._workers.items()):
            # Remove from registry first so that the SIGCHLD handler's
            # on_worker_stop call returns early (pid not in _workers),
            # preventing duplicate cleanup
            if pid in self._workers:
                del self._workers[pid]
            else:
                continue

            # Best effort cleanup
            try:
                await worker.stop()
                worker.join(self._settings.parallel_workers_timeout)
            except Exception as e:
                logger.error(
                    "Failed to complete cleanup of worker "
                    f"with PID {pid} on {self.name}. ",
                    exc_info=True,
                )
                cleanup_errors.append(e)
            finally:
                # Always ensure the worker is terminated
                if worker.exitcode is None:
                    worker.kill()

            logger.debug(f"Worker with PID {pid} on {self.name} stopped.")

            # Cancel any in-flight requests for this worker
            self._dispatcher.on_worker_stop(
                worker, worker.exitcode if worker.exitcode is not None else -1
            )

            results = await asyncio.gather(
                *[callback(worker) for callback in self._on_worker_stop],
                return_exceptions=True,
            )
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                logger.error(
                    f"{len(failures)} worker stop hook(s) failed "
                    f"for worker with PID {pid} on {self.name}",
                )
                cleanup_errors.extend(failures)

        if cleanup_errors:
            raise cleanup_errors[0]
