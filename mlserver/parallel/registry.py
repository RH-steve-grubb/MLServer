import asyncio
import os
import signal

from collections.abc import Sequence

from ..settings import ModelSettings
from ..utils import to_absolute_path
from ..model import MLModel
from ..utils import defer_cancellation, with_operation_lock
from weakref import WeakValueDictionary
from ..settings import Settings
from ..env import Environment, compute_hash_of_file, compute_hash_of_string
from ..registry import model_initialiser

from .errors import EnvironmentNotFound, InferencePoolUnavailable
from .logging import logger
from .pool import InferencePool, InferencePoolHook
from .worker import WorkerModelHook

ENV_HASH_ATTR = "__env_hash__"


def _set_environment_hash(model: MLModel, env_hash: str | None):
    setattr(model, ENV_HASH_ATTR, env_hash)


def _get_environment_hash(model: MLModel) -> str | None:
    # No default — AttributeError signals that pool identity was not resolved.
    return getattr(model, ENV_HASH_ATTR)


def _get_env_tarball(model: MLModel) -> str | None:
    model_settings = model.settings
    if model_settings.parameters is None:
        return None

    env_tarball = model_settings.parameters.environment_tarball
    if env_tarball is None:
        return None

    return to_absolute_path(model_settings, env_tarball)


def _append_gid_environment_hash(
    env_hash: str, inference_pool_gid: str | None = None
) -> str:
    return f"{env_hash}-{inference_pool_gid}"


class InferencePoolRegistry:
    """
    Keeps track of the different inference pools loaded in the server.
    Each inference pool will generally be used to load a different environment.
    """

    def __init__(
        self,
        settings: Settings,
        on_worker_stop: Sequence[InferencePoolHook] = [],
        on_worker_load: Sequence[WorkerModelHook] = [],
        on_worker_unload: Sequence[WorkerModelHook] = [],
    ):
        self._settings = settings
        self._operation_locks: WeakValueDictionary[str | None, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._closing = False
        self._on_worker_stop = on_worker_stop
        self._on_worker_load = on_worker_load
        self._on_worker_unload = on_worker_unload
        self._default_pool: InferencePool | None = InferencePool(
            self._settings,
            on_worker_stop=on_worker_stop,
            on_worker_load=on_worker_load,
            on_worker_unload=on_worker_unload,
        )
        self._pools: dict[str, InferencePool] = {}

        os.makedirs(self._settings.environments_dir, exist_ok=True)

        # Register sigchld signal handler (saving original to restore it later)
        self._original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
        signal.signal(
            signal.SIGCHLD,
            lambda *args: asyncio.create_task(self._handle_worker_stop(*args)),
        )

    async def _handle_worker_stop(self, signum, frame):
        try:
            # Loop to reap all stopped children — SIGCHLD signals can coalesce,
            # so a single signal may represent multiple stopped workers.
            while True:
                pid, wait_status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    # No more stopped children
                    break
                exit_code = os.waitstatus_to_exitcode(wait_status)
                if exit_code == 0:
                    # Clean exit — no crash recovery needed
                    continue
                try:
                    # Notify all pools since we don't know which pool owns
                    # this PID — each pool checks internally
                    if self._default_pool is not None:
                        await self._default_pool.on_worker_stop(pid, exit_code)
                    await asyncio.gather(
                        *[
                            pool.on_worker_stop(pid, exit_code)
                            for pool in self._pools.values()
                        ]
                    )
                except Exception:
                    logger.error(
                        f"Failed to handle stop for worker with PID {pid}",
                        exc_info=True,
                    )
        except ChildProcessError:
            # Raised when there are no child processes left to wait on
            pass

    async def _get_or_create(self, model: MLModel) -> InferencePool:
        if (
            model.settings.parameters is not None
            and model.settings.parameters.environment_path
        ):
            pool = await self._get_or_create_with_existing_env(
                model.settings.parameters.environment_path,
                model.settings.parameters.inference_pool_gid,
            )
        else:
            pool = await self._get_or_create_with_tarball(model)
        return pool

    async def _get_or_create_with_existing_env(
        self,
        environment_path: str,
        inference_pool_gid: str | None,
    ) -> InferencePool:
        """
        Creates or returns the InferencePool for a model that uses an existing
        python environment.
        """
        expanded_environment_path = os.path.abspath(
            os.path.expanduser(os.path.expandvars(environment_path))
        )
        logger.info(f"Using environment {expanded_environment_path}")
        env_hash = await compute_hash_of_string(expanded_environment_path)

        if inference_pool_gid is not None:
            env_hash = _append_gid_environment_hash(env_hash, inference_pool_gid)

        if env_hash in self._pools:
            return self._pools[env_hash]

        env = Environment(
            env_path=expanded_environment_path,
            env_hash=env_hash,
            delete_env=False,
        )
        pool = InferencePool(
            self._settings,
            env=env,
            on_worker_stop=self._on_worker_stop,
            on_worker_load=self._on_worker_load,
            on_worker_unload=self._on_worker_unload,
        )
        self._pools[env_hash] = pool
        return pool

    async def _get_or_create_with_tarball(self, model: MLModel) -> InferencePool:
        """
        Creates or returns the InferencePool for a model that uses a
        tarball as a Python environment.
        """
        env_tarball = _get_env_tarball(model)
        inference_pool_gid = (
            model.settings.parameters.inference_pool_gid
            if model.settings.parameters
            else None
        )

        if not env_tarball:
            if not inference_pool_gid:
                if self._default_pool is None:
                    raise InferencePoolUnavailable("default inference pool")
                return self._default_pool
            if inference_pool_gid not in self._pools:
                self._pools[inference_pool_gid] = InferencePool(
                    self._settings,
                    pool_gid=inference_pool_gid,
                    on_worker_stop=self._on_worker_stop,
                    on_worker_load=self._on_worker_load,
                    on_worker_unload=self._on_worker_unload,
                )
            return self._pools[inference_pool_gid]

        env_hash = await compute_hash_of_file(env_tarball)
        if inference_pool_gid is not None:
            env_hash = _append_gid_environment_hash(env_hash, inference_pool_gid)

        if env_hash in self._pools:
            return self._pools[env_hash]

        env = await self._extract_tarball(env_hash, env_tarball)
        self._pools[env_hash] = InferencePool(
            self._settings,
            env=env,
            on_worker_stop=self._on_worker_stop,
            on_worker_load=self._on_worker_load,
            on_worker_unload=self._on_worker_unload,
        )

        return self._pools[env_hash]

    async def _extract_tarball(self, env_hash: str, env_tarball: str) -> Environment:
        env_path = self._get_env_path(env_hash)
        if os.path.isdir(env_path):
            # If env has already been extracted, use that
            return Environment(env_path, env_hash)

        os.makedirs(env_path)
        return await Environment.from_tarball(env_tarball, env_path, env_hash)

    def _get_env_path(self, env_hash: str) -> str:
        return os.path.join(self._settings.environments_dir, env_hash)

    async def _find(self, model: MLModel) -> InferencePool:
        env_hash = _get_environment_hash(model)
        inference_pool_gid = (
            model.settings.parameters.inference_pool_gid
            if model.settings.parameters
            else None
        )

        if not env_hash:
            if not inference_pool_gid:
                if self._default_pool is None:
                    raise InferencePoolUnavailable("default inference pool")
                return self._default_pool
            else:
                return self._pools[inference_pool_gid]

        if env_hash not in self._pools:
            raise EnvironmentNotFound(model, env_hash)

        return self._pools[env_hash]

    def _should_load_model(self, model_settings: ModelSettings):
        if model_settings.parallel_workers is not None:
            logger.warning(
                "DEPRECATED!! The `parallel_workers` setting at the model-level "
                "has now been deprecated and moved "
                "to the top-level server "
                "settings. "
                "This field will be removed in MLServer 1.2.0. "
                "To access the new field, you can either update the "
                "`settings.json` file, or update the `MLSERVER_PARALLEL_WORKERS` "
                "environment variable. "
                f"The current value of the server-level's `parallel_workers` field is "
                f"'{self._settings.parallel_workers}'."
            )

            # NOTE: This is a remnant from the previous architecture for parallel
            # workers, where each worker had its own pool.
            # For backwards compatibility, we will respect when a model disables
            # parallel inference.
            if model_settings.parallel_workers <= 0:
                return False

        if not self._settings.parallel_workers:
            return False

        return True

    def model_initialiser(self, model_settings: ModelSettings) -> MLModel:
        """
        Used to initialise a model object in the ModelRegistry.
        """
        if not self._should_load_model(model_settings):
            # If parallel inference should not be used, instantiate the model
            # as normal.
            return model_initialiser(model_settings)

        parameters = model_settings.parameters
        if not parameters or not parameters.environment_tarball:
            # If model is not using a custom environment, instantiate the model
            # as normal.
            return model_initialiser(model_settings)

        # Otherwise, return a dummy model for now and wait for the load_model
        # hook to create the actual thing.
        # This avoids instantiating the model's actual class within the
        # main process.
        return MLModel(model_settings)

    async def _get_pool_lock(
        self, model: MLModel, *, loading: bool = False
    ) -> asyncio.Lock:
        """Resolve pool identity and return its shared operation lock."""
        if not self._should_load_model(model.settings):
            # No pool resources are accessed for this no-op.
            return asyncio.Lock()

        parameters = model.settings.parameters
        gid = parameters.inference_pool_gid if parameters else None
        if loading:
            env_hash = None
            if parameters and parameters.environment_path:
                path = os.path.abspath(
                    os.path.expanduser(os.path.expandvars(parameters.environment_path))
                )
                env_hash = await compute_hash_of_string(path)
            else:
                tarball = _get_env_tarball(model)
                if tarball:
                    env_hash = await compute_hash_of_file(tarball)
            if env_hash is not None and gid is not None:
                env_hash = _append_gid_environment_hash(env_hash, gid)
        else:
            try:
                env_hash = _get_environment_hash(model)
            except AttributeError:
                # Unload will treat an unresolved pool identity as a no-op.
                return asyncio.Lock()

        key = env_hash if env_hash is not None else gid or None
        return self._operation_locks.setdefault(key, asyncio.Lock())

    @with_operation_lock(lambda self, model: self._get_pool_lock(model, loading=True))
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.model_operation_timeout
    )
    async def load_model(self, model: MLModel) -> MLModel:
        if self._closing:
            raise InferencePoolUnavailable()
        if not self._should_load_model(model.settings):
            # Skip load if model has disabled parallel workers
            return model

        pool = await self._get_or_create(model)
        # Set env_hash before dispatch so cleanup can find the correct pool
        # even if pool.load_model fails
        _set_environment_hash(model, pool.env_hash)

        try:
            loaded = await pool.load_model(model)
        except Exception:
            try:
                await self._close_pool_if_empty(pool)
            except Exception:
                pass
            raise

        _set_environment_hash(loaded, pool.env_hash)
        return loaded

    @with_operation_lock(lambda self, model: self._get_pool_lock(model))
    @defer_cancellation(
        lambda self, *args, **kwargs: self._settings.model_operation_timeout
    )
    async def unload_model(self, model: MLModel) -> MLModel:
        if self._closing:
            raise InferencePoolUnavailable()
        if not self._should_load_model(model.settings):
            # Skip unload if model has disabled parallel workers
            return model

        try:
            pool = await self._find(model)
        except (EnvironmentNotFound, KeyError, AttributeError):
            logger.debug(
                f"No pool found for model '{model.settings.name}' during unload"
            )
            return model

        try:
            unloaded = await pool.unload_model(model)
        finally:
            try:
                await self._close_pool_if_empty(pool)
            except Exception:
                pass

        return unloaded

    async def _close_pool_if_empty(self, pool: InferencePool) -> None:
        if pool == self._default_pool or not pool.empty():
            return

        if pool.env_hash:
            logger.info(f"Inference pool with hash '{pool.env_hash}' is now empty")
            await self._close_pool(pool.env_hash)
        elif pool.pool_gid:
            logger.info(f"Inference pool with GID '{pool.pool_gid}' is now empty")
            await self._close_pool(pool.pool_gid)

    async def close(self):
        if self._closing:
            return
        self._closing = True
        for lock in list(self._operation_locks.values()):
            async with lock:
                pass
        # Reset signal handler
        signal.signal(signal.SIGCHLD, self._original_sigchld_handler)
        # Best effort cleanup - closure attempted for all pools and
        # pools always removed from registry
        results = await asyncio.gather(
            self._close_pool(None),
            *[self._close_pool(env_hash) for env_hash in self._pools],
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            raise failures[0]

    async def _close_pool(self, env_hash: str | None = None):
        pool: InferencePool | None = self._default_pool
        if env_hash is not None:
            pool = self._pools.get(env_hash)
        if pool is None:
            return

        # Best effort cleanup
        logger.info(f"Waiting for shutdown of {pool.name}...")
        try:
            await pool.close()
            logger.info(f"Shutdown of {pool.name} complete")
        except Exception:
            logger.error(
                f"Failed to shut down {pool.name}.",
                exc_info=True,
            )
            raise
        finally:
            # Always remove pool from registry,
            # cannot guarantee pool state at this point
            if env_hash is not None:
                pool = self._pools.pop(env_hash, None)
                if pool is not None:
                    pool._env = None  # pylint: disable=protected-access
            else:
                self._default_pool = None
