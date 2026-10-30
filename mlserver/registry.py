import asyncio
import json

from collections.abc import Awaitable, Callable, Sequence
from itertools import chain
from functools import cmp_to_key
from prometheus_client import Counter

from .context import model_context
from .model import MLModel
from .utils import defer_cancellation, with_operation_lock
from weakref import WeakValueDictionary
from .errors import ModelNotFound
from .logging import logger
from .settings import ModelSettings, DEFAULT_MODEL_OPERATION_TIMEOUT
from .metrics.context import SELDON_MODEL_NAME_LABEL, SELDON_MODEL_VERSION_LABEL

from mlserver.errors import ModelLoadError, ModelUnloadError

ModelInitialiser = Callable[[ModelSettings], MLModel]
ModelRegistryHook = Callable[[MLModel], Awaitable[MLModel]]


def _get_version(model_settings: ModelSettings) -> str | None:
    if model_settings.parameters:
        return model_settings.parameters.version

    return None


def _is_newer(a: MLModel, b: MLModel) -> int:
    """
    Returns true if 'a' is newer than 'b'.

    TODO: Support other ordering schemes (e.g. semver).
    """
    if a.version is None:
        return 1

    if b.version is None:
        return -1

    try:
        a_int = int(a.version)
        b_int = int(b.version)

        return a_int - b_int
    except ValueError:
        if a.version > b.version:
            return 1
        elif a.version < b.version:
            return -1
        else:
            return 0


def model_initialiser(model_settings: ModelSettings) -> MLModel:
    try:
        model_class = model_settings.implementation
    except (
        ValueError,
        ImportError,
        AttributeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"Refused to load model '{model_settings.name}': {exc}"
        ) from exc
    return model_class(model_settings)  # type: ignore


class SingleModelRegistry:
    """
    Registry for a single model with multiple versions.
    """

    # Shared counter across all instances - created on first use (lazy init
    # required because PROMETHEUS_MULTIPROC_DIR must be set before any metric
    # objects are created, and that happens at server start, not import time)
    _ModelCleanupFailuresTotal = None

    @classmethod
    def _increment_cleanup_failure_metric(
        cls, model_name: str, model_version: str | None
    ):
        if cls._ModelCleanupFailuresTotal is None:
            cls._ModelCleanupFailuresTotal = Counter(
                "model_cleanup_failures_total",
                "Total number of model cleanup failures",
                [SELDON_MODEL_NAME_LABEL, SELDON_MODEL_VERSION_LABEL],
            )
        cls._ModelCleanupFailuresTotal.labels(
            **{
                SELDON_MODEL_NAME_LABEL: model_name,
                SELDON_MODEL_VERSION_LABEL: model_version or "",
            }
        ).inc()

    def __init__(
        self,
        model_settings: ModelSettings,
        on_model_load: Sequence[ModelRegistryHook] = [],
        on_model_unload: Sequence[ModelRegistryHook] = [],
        model_initialiser: ModelInitialiser = model_initialiser,
        model_operation_timeout: float = DEFAULT_MODEL_OPERATION_TIMEOUT,
    ):
        self._operation_lock = asyncio.Lock()
        self._versions: dict[str, MLModel] = {}
        self._pending_reload: dict[str, MLModel] = {}
        self._default: MLModel | None = None

        self._name = model_settings.name
        self._on_model_load = on_model_load
        self._on_model_unload = on_model_unload
        self._model_initialiser = model_initialiser
        self._model_operation_timeout = model_operation_timeout

    @property
    def default(self) -> MLModel | None:
        if self._default is None:
            self._default = self._find_default()

        return self._default

    def _find_default(self) -> MLModel | None:
        if self._default is None:
            if self._versions:
                version_key = cmp_to_key(_is_newer)
                latest_model = max(self._versions.values(), key=version_key)
                return latest_model

        return self._default

    def _clear_default(self):
        self._default = None

    def _refresh_default(self, new_model: MLModel | None = None) -> MLModel | None:
        if new_model:
            # Check whether new model is "defaulter" than current default
            # NOTE: This should help to avoid iterating through all versioned
            # models each time a new model is loaded to find the latest

            if self._default is None:
                # If default is currently empty, take new one as new default
                self._default = new_model
                return new_model

            if new_model.version is None:
                # If new model doesn't have a version, assume it's "defaulter"
                # than previous default
                self._default = new_model
                return new_model

            if self._default.version is None:
                # If default doesn't have a version (and new one does), assume
                # that current default is "defaulter" than new one
                return self._default

            # Otherwise, compare versions
            if _is_newer(new_model, self._default) >= 0:
                self._default = new_model
                return new_model

            return self._default

        if self._default and self._default.version is None:
            # If there isn't a new model to compare, and current default has no
            # version, then consider that current one is "defaulter" than other
            # versioned models
            return self._default

        # Otherwise, find latest from current set of versions
        self._default = self._find_default()
        return self._default

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def load(
        self, model_settings: ModelSettings, parallel: bool = False
    ) -> MLModel:
        """
        Load or reload a model version.

        If a model with the same version is already loaded, a reload is triggered:
        the new model is loaded first, then the old one is unloaded. If no previous
        version exists, a fresh load is performed.

        :param model_settings: Settings for the model version to load.
        :param parallel: Whether this load is being performed on a parallel worker.
            When True, unloading the previous model is skipped — the main process
            dispatches unload to all workers separately.
        """
        version = _get_version(model_settings)
        previous_loaded_model = self._find_exact_version(version)
        new_model = self._model_initialiser(model_settings)

        with model_context(model_settings):
            # If a previous loaded model is found this is the trigger for reload
            # Otherwise a fresh load is triggered
            if previous_loaded_model:
                new_model = await self._load_model(new_model, True)
                # If this operation is being performed on a parallel worker,
                # skip unload. The main process will dispatch an Unload message
                # to all workers via InferencePool.unload_model, which triggers
                # unload_version -> _unload_model on each worker.
                if not parallel:
                    await self._unload_model(previous_loaded_model, False)
            else:
                new_model = await self._load_model(new_model, False)

        return new_model

    async def _load_model(self, model: MLModel, reload: bool = False):
        model_msg = f"model '{model.name}'"
        version = _get_version(model.settings) or ""
        if version:
            model_msg = f"version {model.version} of {model_msg}"

        try:
            if reload:
                # For reload scenarios, the model should only be registered
                # after it is successfully loaded to allow rollback
                self._pending_reload[version] = model
            else:
                # Register the model before loading it, to ensure that the model
                # appears as a not-ready (i.e. loading) model in the registry
                self._register(model)

            try:
                for callback in self._on_model_load:
                    # NOTE: Callbacks need to be executed sequentially to ensure that
                    # they go in the right order
                    model = await callback(model)
            finally:
                # Register model again to ensure we save version modified by hooks
                if reload:
                    self._pending_reload[version] = model
                else:
                    self._register(model)

            if not await model.load():
                raise ModelLoadError(f"Model load returned False for {model_msg}.")
        except Exception as load_error:
            logger.error(f"Failed to load {model_msg}. Attempting cleanup...")
            try:
                await self._unload_model(model, reload)
            except Exception:
                # Still raise a load error since this is a failure during
                # load failure cleanup
                raise ModelLoadError(
                    f"Model load and cleanup failed for {model_msg}"
                ) from load_error
            raise

        logger.info(f"Loaded {model_msg} successfully.")
        model.ready = True

        return model

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def unload(self) -> None:
        """
        Unload all versions of this model. Always a fresh unload — never called
        as part of a reload operation. Best-effort — all versions are attempted
        regardless of individual failures. Failures are logged and tracked in
        metrics but the registry is always cleared, since a model that fails to
        unload cleanly is still removed from the active registry.
        """
        models = await self.get_models()

        # Perform all unloads concurrently and wait for results to ensure
        # state is properly maintained
        try:
            results = await asyncio.gather(
                *[self._unload_model(model) for model in models],
                return_exceptions=True,
            )
            unload_failures = [r for r in results if isinstance(r, Exception)]
            if unload_failures:
                raise ModelUnloadError(
                    f"Failed to unload {len(unload_failures)} version(s) of "
                    f"model {self._name}."
                )

            logger.info(f"Unloaded all versions of model '{self._name}' successfully.")
        finally:
            # Always remove registry entries
            self._versions.clear()
            self._clear_default()

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def unload_version(
        self, version: str | None = None, rollback: bool = False
    ) -> None:
        """
        Unload a specific version of the model.

        :param version: Version to unload. If None, the default version is used.
        :param rollback: When True, treats this as a failed-reload cleanup —
            unloads the pending new model and keeps the existing registered
            version intact.

        Unlike :meth:`load`, there is no ``parallel`` parameter here because unloading
        is performed the same way on both the main process and worker processes.

        Failures are logged and tracked in metrics but the registry is always
        updated, since a model that fails to unload cleanly is still removed
        from the active registry.
        """
        model = await self.get_model(version)
        await self._unload_model(model, rollback)

    async def _unload_model(self, model: MLModel, rollback: bool = False):
        with model_context(model.settings):
            model_msg = f"model '{model.name}'"
            version = _get_version(model.settings) or ""
            if version:
                model_msg = f"version {model.version} of {model_msg}"

            # Unregister any pending reloads - unload overrides them
            pending_model = self._pending_reload.pop(version, None)

            if rollback:
                if pending_model:
                    # If this is a rollback unload, and a pending reload model exists,
                    # then we want to cleanup the pending model and maintain the state
                    # of any existing old model
                    model = pending_model
                else:
                    # If this is a rollback unload, and a pending reload model does
                    # not exist, then there is nothing to rollback so this is a no-op
                    return
            else:
                if pending_model:
                    # If this is not a rollback unload, and a pending reload model
                    # exists, then that means we need to complete the reload by
                    # registering the new model and cleaning up the old one
                    self._register(pending_model)
                else:
                    # If this is not a rollback unload, and no pending reload model
                    # exists, then that means this is a standard unload and the model
                    # should be unregistered and cleaned up
                    self._unregister(model)

            model.ready = False

            unload_hook_failures = []
            for callback in self._on_model_unload:
                try:
                    # NOTE: Callbacks need to be executed sequentially to ensure that
                    # they go in the right order
                    model = await callback(model)
                except Exception as unload_hook_failure:
                    logger.error(
                        f"Unload hook failed for {model_msg}",
                        exc_info=True,
                    )
                    unload_hook_failures.append(unload_hook_failure)

            # Cleanup is best effort
            # Always attempt model.unload() even if hooks failed
            try:
                if not await model.unload():
                    raise ModelUnloadError(
                        f"Model unload returned False for {model_msg}."
                    )
            except Exception:
                # Track cleanup failures
                self._increment_cleanup_failure_metric(model.name, model.version)
                logger.error(
                    f"Failed to unload {model_msg}. Model removed from registry, "
                    f"but resources may still be held.",
                    exc_info=True,
                )
                raise

            if unload_hook_failures:
                # Track cleanup failures
                self._increment_cleanup_failure_metric(model.name, model.version)
                raise ModelUnloadError(
                    f"{len(unload_hook_failures)} unload hook(s) "
                    f"failed for {model_msg}. Model removed "
                    "from registry, but some resources "
                    "may still be held."
                )

            logger.info(f"Unloaded {model_msg} successfully.")

    def _find_exact_version(self, version: str | None) -> MLModel | None:
        if version:
            return self._versions.get(version)
        return (
            self._default if self._default and self._default.version is None else None
        )

    def _find_model(self, version: str | None = None) -> MLModel | None:
        if version:
            if version not in self._versions:
                return None

            return self._versions[version]

        return self.default

    async def get_model(self, version: str | None = None) -> MLModel:
        model = self._find_model(version)

        if model is None:
            raise ModelNotFound(self._name, version)

        return model

    async def get_models(self) -> list[MLModel]:
        # NOTE: `.values()` returns a "view" instead of a list
        models = list(self._versions.values())

        # Add default if not versioned (as it won't be present on the
        # `_versions` dict
        if self.default and not self.default.version:
            models.append(self.default)

        return models

    def _register(self, model: MLModel):
        if model.version:
            self._versions[model.version] = model

        self._refresh_default(model)

    def _unregister(self, model: MLModel):
        if model.version:
            del self._versions[model.version]
        if model == self.default:
            self._clear_default()

    def empty(self) -> bool:
        if self._versions:
            return False

        return self.default is None


class MultiModelRegistry:
    """
    Multiple model registry, where each model can have multiple versions.
    """

    def __init__(
        self,
        on_model_load: Sequence[ModelRegistryHook] = [],
        on_model_unload: Sequence[ModelRegistryHook] = [],
        model_initialiser: ModelInitialiser = model_initialiser,
        model_operation_timeout: float = DEFAULT_MODEL_OPERATION_TIMEOUT,
    ):
        # Locks outlive registry entries while any owner or waiter holds them.
        self._operation_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._models: dict[str, SingleModelRegistry] = {}
        self._on_model_load = on_model_load
        self._on_model_unload = on_model_unload
        self._model_initialiser = model_initialiser
        self._model_operation_timeout = model_operation_timeout

    def _get_operation_lock(self, name: str) -> asyncio.Lock:
        return self._operation_locks.setdefault(name, asyncio.Lock())

    @with_operation_lock(
        lambda self, model_settings, *args, **kwargs: self._get_operation_lock(
            model_settings.name
        )
    )
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def load(
        self, model_settings: ModelSettings, parallel: bool = False
    ) -> MLModel:
        """
        Load or reload a model. Lazily creates a :class:`SingleModelRegistry`
        for the model name if one does not exist. Cleans it up if the load
        fails and the registry is left empty.

        :param model_settings: Settings for the model version to load.
        :param parallel: Whether this load is being performed on a parallel worker.
        """
        # Lazily create single model registry if it does not exist already
        if model_settings.name not in self._models:
            self._models[model_settings.name] = SingleModelRegistry(
                model_settings,
                on_model_load=self._on_model_load,
                on_model_unload=self._on_model_unload,
                model_initialiser=self._model_initialiser,
                model_operation_timeout=self._model_operation_timeout,
            )
        try:
            return await self._models[model_settings.name].load(
                model_settings, parallel
            )
        except Exception:
            # Clean up the single model registry on load failure if empty
            if (
                model_settings.name in self._models
                and self._models[model_settings.name].empty()
            ):
                del self._models[model_settings.name]
            raise

    @with_operation_lock(lambda self, name: self._get_operation_lock(name))
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def unload(self, name: str) -> None:
        """
        Unload all versions of a model and remove it from the registry.
        Always a fresh unload — never called as part of a reload operation.
        Unloading is performed the same way on both the main process and
        worker processes.
        """
        model_registry = self._get_model_registry(name)
        try:
            await model_registry.unload()
        finally:
            # Always remove from registry - SingleModelRegistry is empty after
            # unload regardless of success (models removed before actual unload)
            del self._models[name]

    @with_operation_lock(
        lambda self, name, *args, **kwargs: self._get_operation_lock(name)
    )
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def unload_version(
        self, name: str, version: str | None = None, rollback: bool = False
    ) -> None:
        """
        Unload a specific version of a model. Removes the
        :class:`SingleModelRegistry` if it becomes empty after the unload.

        :param name: Model name.
        :param version: Version to unload. If None, the default version is used.
        :param rollback: When True, treats this as a failed-reload cleanup.

        Unloading is performed the same way on both the main process and
        worker processes.
        """
        model_registry = self._get_model_registry(name, version)
        try:
            await model_registry.unload_version(version, rollback)
        finally:
            # If single model registry is empty after unload version failure
            # remove it from the multi model registry
            if model_registry.empty():
                del self._models[name]

    async def get_model(self, name: str, version: str | None = None) -> MLModel:
        model_registry = self._get_model_registry(name, version)
        return await model_registry.get_model(version)

    async def get_models(self, name: str | None = None) -> list[MLModel]:
        if name is not None:
            model_registry = self._get_model_registry(name)
            return await model_registry.get_models()

        models_list = await asyncio.gather(
            *[model.get_models() for model in self._models.values()]
        )

        return list(chain.from_iterable(models_list))

    def _get_model_registry(
        self, name: str, version: str | None = None
    ) -> SingleModelRegistry:
        if name not in self._models:
            raise ModelNotFound(name, version)

        return self._models[name]
