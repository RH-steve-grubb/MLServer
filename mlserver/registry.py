import asyncio
import json
from contextvars import ContextVar

from collections.abc import Awaitable, Callable, Sequence
from itertools import chain
from functools import cmp_to_key
from prometheus_client import Counter

from .context import model_context
from .model import MLModel
from .utils import defer_cancellation, with_operation_lock, create_task_independent
from weakref import WeakValueDictionary
from .errors import ModelNotFound
from .logging import logger
from .settings import ModelSettings, DEFAULT_MODEL_OPERATION_TIMEOUT
from .metrics.context import MODEL_NAME_LABEL, MODEL_VERSION_LABEL
from mlserver.errors import ModelLoadError, ModelUnloadError

_model_load_hooks_active: ContextVar[bool] = ContextVar(
    "model_load_hooks_active", default=False
)

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
                [MODEL_NAME_LABEL, MODEL_VERSION_LABEL],
            )
        cls._ModelCleanupFailuresTotal.labels(
            **{
                MODEL_NAME_LABEL: model_name,
                MODEL_VERSION_LABEL: model_version or "",
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
        # These are resource owners, separate from entries eligible for serving.
        # Failed cleanup stays here until an explicit unload retries it.
        self._cleanup_tasks: dict[MLModel, asyncio.Task] = {}
        self._cleanup_preparations: dict[MLModel, asyncio.Task] = {}
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
        await self._drain_cleanup()
        version = _get_version(model_settings)
        previous_loaded_model = self._find_exact_version(version)
        # Hooks may amend preparation settings. Do not let a reused settings
        # object mutate the configuration of the committed generation.
        new_model = self._model_initialiser(
            model_settings.model_copy(deep=True)
            if previous_loaded_model is not None
            else model_settings
        )

        with model_context(model_settings):
            new_model = await self._load_model(
                new_model, previous_loaded_model is not None
            )
            # Workers stage replacement publication until the main process
            # dispatches the old generation's unload separately.
            if not parallel:
                await self._publish_model(new_model, previous_loaded_model)

        return new_model

    async def _publish_model(self, model: MLModel, previous: MLModel | None) -> None:
        # A hook can redirect preparation to another committed version. Both
        # displaced instances stay owned, even if the first cleanup times out.
        target = self._find_exact_version(model.version)
        retired: list[MLModel] = []
        for candidate in (previous, target):
            if candidate is not None and candidate is not model:
                if not any(candidate is owned for owned in retired):
                    retired.append(candidate)

        self._remove_pending(model)
        self._unregister(model)
        for old in retired:
            self._unregister(old)
            old.ready = False
        self._refresh_default()
        self._register(model)
        cleanup = [self._start_cleanup(old) for old in retired]
        results = await asyncio.gather(
            *(asyncio.shield(task) for task in cleanup), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _load_model(self, model: MLModel, reload: bool = False):
        model_msg = f"model '{model.name}'"
        version = _get_version(model.settings) or ""
        if version:
            model_msg = f"version {model.version} of {model_msg}"

        preparation = None
        initial_model = model
        hook_models = []
        try:
            target = self._find_exact_version(model.version)
            reload = reload or (target is not None and target is not model)
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
                    token = _model_load_hooks_active.set(True)
                    try:
                        model = await callback(model)
                        hook_models.append(model)
                    finally:
                        _model_load_hooks_active.reset(token)
            finally:
                # Register model again to ensure we save version modified by hooks
                if reload:
                    self._pending_reload[version] = model
                else:
                    if model is not initial_model or (model.version or "") != version:
                        self._unregister(initial_model)
                        self._refresh_default()
                    target = self._find_exact_version(model.version)
                    if target is not None and target is not model:
                        self._pending_reload[version] = model
                    else:
                        self._register(model)

            # Do not cancel runtime preparation: it may own native work or a
            # thread which asyncio cancellation cannot stop. An abandoned load
            # is cleaned only after this exact preparation has settled.
            preparation = asyncio.create_task(model.load())
            if not await asyncio.shield(preparation):
                raise ModelLoadError(f"Model load returned False for {model_msg}.")
            for hooked in hook_models:
                hooked._load_complete()
        except (Exception, asyncio.CancelledError) as load_error:
            for hooked in hook_models:
                hooked._load_failed()
            logger.error(f"Failed to load {model_msg}. Attempting cleanup...")
            try:
                self._discard_load(model)
                cleanup = self._start_cleanup(model, preparation)
                if isinstance(load_error, asyncio.CancelledError):
                    raise
                await asyncio.shield(cleanup)
            except (Exception, asyncio.CancelledError) as cleanup_error:
                if isinstance(load_error, asyncio.CancelledError):
                    # The timeout remains the primary outcome. Cleanup is owned
                    # by _cleanup_tasks and its failure is logged and retained.
                    raise load_error
                # Still raise a load error since this is a failure during
                # load failure cleanup
                failure = ModelLoadError(
                    f"Model load and cleanup failed for {model_msg}"
                )
                failure.cleanup_error = cleanup_error
                raise failure from load_error
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
        retained_failures = await self._retry_cleanup()
        models = await self.get_models()

        # Perform all unloads concurrently and wait for results to ensure
        # state is properly maintained
        try:
            # Transfer every selected instance to a resource owner before the
            # next await. Cancellation can otherwise stop an unload coroutine
            # before it starts, while the finally block removes its registration.
            cleanup = [self._start_unload(model) for model in models]
            results = await asyncio.gather(
                *(asyncio.shield(task) for task in cleanup if task is not None),
                return_exceptions=True,
            )
            unload_failures = [r for r in results if isinstance(r, BaseException)]
            if retained_failures:
                # Report the retained cause after independent versions have
                # also been attempted. Failed resource owners remain tracked.
                raise retained_failures[0]
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
        await self._unload_version(version, rollback, exact=False)

    @with_operation_lock(lambda self, *args, **kwargs: self._operation_lock)
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def _unload_exact_version(
        self, version: str | None, rollback: bool = False
    ) -> None:
        # Worker updates already identify the frontend's selected version.
        # An absent unversioned attempt must not select a versioned default.
        await self._unload_version(version, rollback, exact=True)

    async def _unload_version(
        self, version: str | None, rollback: bool, exact: bool
    ) -> None:
        current = (
            self._find_exact_version(version) if exact else self._find_model(version)
        )
        cleanup_version = current.version if current else version
        cleanup_models = None
        if exact or current is not None or version is not None:
            cleanup_models = [
                model
                for model in self._cleanup_tasks
                if model.version == cleanup_version
            ]
        await self._drain_cleanup(retry=True, models=cleanup_models)
        if self.empty():
            return
        if rollback:
            model = self._pending_reload.get(version or "")
            if model is None:
                return
        elif exact:
            model = self._find_exact_version(version)
            if model is None:
                return
        else:
            model = await self.get_model(version)
        await self._unload_model(model, rollback)

    def _remove_pending(self, model: MLModel) -> bool:
        # A hook may return another version while the staged entry still has
        # the original version key. Only this instance belongs to this cleanup.
        removed = False
        for version, pending in list(self._pending_reload.items()):
            if pending is model:
                del self._pending_reload[version]
                removed = True
        return removed

    def _discard_load(self, model: MLModel) -> None:
        self._remove_pending(model)
        self._unregister(model)
        model.ready = False

    async def _unload_model(
        self,
        model: MLModel,
        rollback: bool = False,
        preparation: asyncio.Task | None = None,
    ):
        task = self._start_unload(model, rollback, preparation)
        if task is not None:
            await asyncio.shield(task)

    def _start_unload(
        self,
        model: MLModel,
        rollback: bool = False,
        preparation: asyncio.Task | None = None,
    ) -> asyncio.Task | None:
        version = _get_version(model.settings) or ""
        pending = self._pending_reload.get(version)
        if rollback:
            if not self._remove_pending(model):
                return None
        elif pending is not None and pending is not model and pending.ready:
            # Publication precedes old-generation cleanup. A cleanup failure
            # must never roll back this successfully prepared replacement.
            del self._pending_reload[version]
            self._unregister(model)
            self._refresh_default()
            self._register(pending)
        else:
            self._unregister(model)
        model.ready = False
        return self._start_cleanup(model, preparation)

    def _start_cleanup(
        self,
        model: MLModel,
        preparation: asyncio.Task | None = None,
    ) -> asyncio.Task:
        task = self._cleanup_tasks.get(model)
        if task is None:
            if preparation is not None:
                self._cleanup_preparations[model] = preparation
            preparation = self._cleanup_preparations.get(model)
            task = create_task_independent(self._cleanup_model(model, preparation))
            self._cleanup_tasks[model] = task
            task.add_done_callback(lambda done: self._cleanup_done(model, done))
        return task

    def _cleanup_done(self, model: MLModel, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is None:
            self._cleanup_tasks.pop(model, None)
            self._cleanup_preparations.pop(model, None)
        else:
            logger.error(
                f"Cleanup of model '{model.name}' did not complete; "
                "ownership retained until explicit unload retries cleanup.",
                exc_info=None if task.cancelled() else task.exception(),
            )

    async def _drain_cleanup(
        self,
        retry: bool = False,
        models: Sequence[MLModel] | None = None,
    ) -> None:
        for model, task in list(self._cleanup_tasks.items()):
            if models is not None and model not in models:
                # Preserve serialization across the entire model name while
                # another version is still mutating. A settled sibling failure
                # stays recorded but must not trigger an unrelated cleanup retry.
                if task.cancelled():
                    raise ModelUnloadError(
                        "Another version has interrupted cleanup; "
                        "retry its unload first"
                    )
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                continue
            if retry and task.done() and (task.cancelled() or task.exception()):
                self._cleanup_tasks.pop(model)
                task = self._start_cleanup(model)
            await asyncio.shield(task)

    async def _retry_cleanup(self) -> list[BaseException]:
        # Join unfinished mutations before starting any independent cleanup.
        # A cancelled cleanup waiter may still own live preparation separately.
        for model, task in list(self._cleanup_tasks.items()):
            if not task.done():
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            preparation = self._cleanup_preparations.get(model)
            if preparation is not None:
                try:
                    await asyncio.shield(preparation)
                except asyncio.CancelledError:
                    if preparation.cancelled():
                        raise ModelUnloadError(
                            "Preparation completion is uncertain; ownership retained"
                        )
                    raise
                except Exception:
                    pass

        retries = []
        for model, task in list(self._cleanup_tasks.items()):
            if task.cancelled() or task.exception() is not None:
                self._cleanup_tasks.pop(model)
                retries.append(asyncio.shield(self._start_cleanup(model)))
        results = await asyncio.gather(*retries, return_exceptions=True)
        return [result for result in results if isinstance(result, BaseException)]

    async def _cleanup_model(
        self,
        model: MLModel,
        preparation: asyncio.Task | None,
    ) -> None:
        if preparation is not None:
            try:
                await asyncio.shield(preparation)
            except asyncio.CancelledError:
                if preparation.cancelled():
                    raise ModelUnloadError(
                        "Preparation was cancelled before settlement; "
                        "native resources may remain live and ownership is retained"
                    )
                raise
            except Exception:
                # Preparation errors are reported by the load caller. They do
                # not remove the obligation to release partially built state.
                pass
        with model_context(model.settings):
            failures = []
            for callback in self._on_model_unload:
                try:
                    model = await callback(model)
                except Exception as error:
                    logger.error("Unload hook failed", exc_info=True)
                    failures.append(error)
            try:
                if not await model.unload():
                    raise ModelUnloadError(
                        f"Model unload returned False for model '{model.name}'."
                    )
            except Exception:
                self._increment_cleanup_failure_metric(model.name, model.version)
                raise
            if failures:
                self._increment_cleanup_failure_metric(model.name, model.version)
                raise ModelUnloadError(
                    f"{len(failures)} unload hook(s) failed for model '{model.name}'."
                ) from failures[0]
            logger.info(f"Unloaded model '{model.name}' successfully.")

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
        # Settings can have changed in a hook since this instance was keyed.
        # Removing by its current version can strand an earlier registration.
        for version, registered in list(self._versions.items()):
            if registered is model:
                del self._versions[version]
        if model is self._default:
            self._clear_default()

    def empty(self) -> bool:
        if self._cleanup_tasks:
            return False
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
        except (Exception, asyncio.CancelledError):
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
            # Keep resource owners after serving entries have been removed.
            if model_registry.empty():
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

    @with_operation_lock(
        lambda self, model_settings, *args, **kwargs: self._get_operation_lock(
            model_settings.name
        )
    )
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def _unload_exact_version(
        self, model_settings: ModelSettings, rollback: bool = False
    ) -> None:
        name = model_settings.name
        model_registry = self._get_model_registry(name, model_settings.version)
        try:
            await model_registry._unload_exact_version(model_settings.version, rollback)
        finally:
            if model_registry.empty():
                del self._models[name]

    async def get_model(self, name: str, version: str | None = None) -> MLModel:
        model_registry = self._get_model_registry(name, version)
        return await model_registry.get_model(version)

    async def get_models(self, name: str | None = None) -> list[MLModel]:
        if name is not None:
            model_registry = self._get_model_registry(name)
            models = await model_registry.get_models()
            if not models:
                raise ModelNotFound(name)
            return models

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
