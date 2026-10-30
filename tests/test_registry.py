import pytest
import asyncio
import json

from asyncio import CancelledError

from mlserver.model import MLModel
from mlserver.errors import (
    MLServerError,
    ModelNotFound,
    ModelLoadError,
    ModelUnloadError,
)
from mlserver.registry import MultiModelRegistry, SingleModelRegistry, model_initialiser
from mlserver.settings import ModelSettings, ModelParameters

from .fixtures import SlowModel, SumModel


@pytest.fixture
async def model_registry(
    model_registry: MultiModelRegistry, mocker
) -> MultiModelRegistry:
    async def _async_val(model: MLModel) -> MLModel:
        return model

    load_stub = mocker.stub("_on_model_load")
    load_stub.side_effect = _async_val
    model_registry._on_model_load = [load_stub]

    unload_stub = mocker.stub("_on_model_unload")
    unload_stub.side_effect = _async_val
    model_registry._on_model_unload = [unload_stub]

    for single_registry in model_registry._models.values():
        single_registry._on_model_load = [load_stub]
        single_registry._on_model_unload = [unload_stub]

    return model_registry


@pytest.mark.parametrize(
    "name, version",
    [
        ("sum-model", "v0"),
        ("sum-model-2", "v0"),
        ("sum-model", "v2"),
        ("sum-model-2", None),
    ],
)
async def test_get_model_not_found(model_registry, name, version):
    with pytest.raises(ModelNotFound) as err:
        await model_registry.get_model(name, version)

    if version is not None:
        assert str(err.value) == f"Model {name} with version {version} not found"
    else:
        assert str(err.value) == f"Model {name} not found"


@pytest.mark.parametrize(
    "name, version",
    [("sum-model", "v1.2.3"), ("sum-model", None), ("sum-model", "")],
)
async def test_get_model(model_registry, sum_model, name, version):
    found_model = await model_registry.get_model(name, version)
    assert found_model.ready
    assert found_model == sum_model


async def test_model_hooks(
    model_registry: MultiModelRegistry, sum_model_settings: ModelSettings
):
    sum_model_settings.name = "sum-model-2"

    sum_model = await model_registry.load(sum_model_settings)
    assert sum_model.ready

    for callback in model_registry._on_model_load:
        callback.assert_called_once_with(sum_model)  # type: ignore[attr-defined]

    await model_registry.unload(sum_model.name)
    for callback in model_registry._on_model_unload:
        callback.assert_called_once_with(sum_model)  # type: ignore[attr-defined]


async def test_load_propagates_model_operation_timeout_to_single_registry(
    sum_model_settings: ModelSettings,
):
    model_operation_timeout = 37
    model_registry = MultiModelRegistry(model_operation_timeout=model_operation_timeout)

    try:
        await model_registry.load(sum_model_settings)

        single_registry = model_registry._models[sum_model_settings.name]
        assert single_registry._model_operation_timeout == model_operation_timeout
    finally:
        await model_registry.unload(sum_model_settings.name)


async def test_reload_model(
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
    mocker,
):
    # Test rolling reload
    sum_model_settings.implementation = SlowModel
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_load(model: SlowModel) -> bool:
        started.set()
        await release.wait()
        return True

    mocker.patch.object(SlowModel, "load", new=controlled_load)
    existing_model = await model_registry.get_model(sum_model_settings.name)
    reload_task = asyncio.create_task(model_registry.load(sum_model_settings))
    await started.wait()

    models = list(await model_registry.get_models())
    assert all(model.ready for model in models)
    assert len(models) == 1

    release.set()
    new_model = await reload_task
    reloaded_model = await model_registry.get_model(sum_model_settings.name)
    assert new_model != existing_model
    assert new_model == reloaded_model
    assert reloaded_model.ready
    assert not existing_model.ready

    for callback in model_registry._on_model_load:
        callback.assert_called_once_with(new_model)  # type: ignore[attr-defined]

    for callback in model_registry._on_model_unload:
        callback.assert_called_once_with(existing_model)  # type: ignore[attr-defined]

    await model_registry.unload(sum_model_settings.name)


async def test_load_multi_version(
    model_registry: MultiModelRegistry, sum_model_settings: ModelSettings
):
    existing_model = await model_registry.get_model(sum_model_settings.name)
    assert sum_model_settings.parameters is not None
    existing_version = sum_model_settings.parameters.version

    # Load new model
    new_model_settings = sum_model_settings.copy(deep=True)
    assert new_model_settings.parameters is not None
    new_model_settings.parameters.version = "v2.0.0"
    new_model = await model_registry.load(new_model_settings)
    assert new_model.ready

    # Ensure latest model is now the default one
    default_model = await model_registry.get_model(sum_model_settings.name)
    assert new_model != existing_model
    assert new_model == default_model

    for callback in model_registry._on_model_load:
        callback.assert_called_once_with(new_model)  # type: ignore[attr-defined]

    # Ensure old model is still reachable
    old_model = await model_registry.get_model(
        sum_model_settings.name, existing_version
    )
    assert old_model == existing_model

    for callback in model_registry._on_model_unload:
        callback.assert_not_called_with(existing_model)  # type: ignore[attr-defined]


async def test_unload(model_registry: MultiModelRegistry, sum_model: MLModel, mocker):
    spy = mocker.spy(sum_model, "unload")
    await model_registry.unload(sum_model.name)

    assert not sum_model.ready
    spy.assert_called_once()


@pytest.mark.parametrize(
    "versions_to_unload",
    [
        [None],
        [None, "v0"],
        ["v0", None],
        ["v0"],
        ["v0", "v1", "v2"],
        [None, "v0", "v1", "v2"],
        ["v0", "v1", "v2", None],
    ],
)
async def test_unload_version(
    versions_to_unload: list[str | None],
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    # Load multiple versions
    to_load = ["v0", "v1", "v2"]
    sum_model_settings.name = "model-foo"

    sum_model_settings = sum_model_settings.copy(deep=True)
    assert sum_model_settings.parameters is not None
    sum_model_settings.parameters.version = None
    default_model = await model_registry.load(sum_model_settings)
    for version in to_load:
        sum_model_settings = sum_model_settings.copy(deep=True)
        assert sum_model_settings.parameters is not None
        sum_model_settings.parameters.version = version
        await model_registry.load(sum_model_settings)

    # Unload versions
    for version in versions_to_unload:  # type: ignore[assignment]
        await model_registry.unload_version(sum_model_settings.name, version)

    if len(versions_to_unload) == len(to_load) + 1:
        # If we have unloaded all models (including the default one), assert
        # that the model has been completely unloaded
        with pytest.raises(ModelNotFound):
            await model_registry.get_models(sum_model_settings.name)
    else:
        models = await model_registry.get_models(sum_model_settings.name)
        for model in models:
            assert model.version not in versions_to_unload

        if None in versions_to_unload:
            new_default_model = await model_registry.get_model(sum_model_settings.name)
            assert new_default_model != default_model


@pytest.mark.parametrize(
    "versions, expected",
    [
        (["4", "3", "2", "1", "7", "5", "6"], "7"),
        (["v1", "v3", "v2"], "v3"),
        (["v10", "v3", "v2"], "v3"),
        (["8", "v3", "7"], "v3"),
        (["v1.0.0", "v1.2.3", "v12.3.4"], "v12.3.4"),
    ],
)
async def test_find_default(
    versions: list[str],
    expected: str,
    sum_model_settings: ModelSettings,
):
    model_settings = sum_model_settings.copy(deep=True)
    model_settings.name = "model-foo"
    foo_registry = SingleModelRegistry(model_settings)

    # Load mock models
    for version in versions:
        model_settings = model_settings.copy(deep=True)
        assert model_settings.parameters is not None
        model_settings.parameters.version = version
        await foo_registry.load(model_settings)

    foo_registry._clear_default()
    default_model = foo_registry._find_default()
    assert default_model is not None
    assert default_model.version == expected


async def test_model_not_ready(model_registry: MultiModelRegistry):
    slow_model_settings = ModelSettings(name="slow-model", implementation=SlowModel)

    load_task = asyncio.create_task(model_registry.load(slow_model_settings))
    # Use asyncio.sleep() to give control back to loop so that the load above
    # gets executed
    await asyncio.sleep(0.1)

    models = list(await model_registry.get_models())
    assert not all([m.ready for m in models])
    assert len(models) == 2

    # Cancel slow load task
    load_task.cancel()
    try:
        await load_task
    except CancelledError:
        pass


async def test_model_load_error(
    model_registry: MultiModelRegistry, load_error_model_settings: ModelSettings
):
    """
    Test that models failing to load are removed from registry.
    """
    with pytest.raises(MLServerError):
        await model_registry.load(load_error_model_settings)

    # Model should NOT be in registry (removed on load failure)
    with pytest.raises(ModelNotFound):
        await model_registry.get_model(load_error_model_settings.name)

    # Verify only the sum-model remains in registry
    models = list(await model_registry.get_models())
    assert len(models) == 1
    assert models[0].name == "sum-model"


async def test_load_error_with_cleanup_failure_preserves_original_error(
    load_error_model_settings: ModelSettings,
):
    """
    Test when load fails AND cleanup fails, original load error is preserved.
    """

    async def failing_unload_hook(model: MLModel) -> MLModel:
        raise RuntimeError("Cleanup hook failed")

    registry = SingleModelRegistry(
        load_error_model_settings,
        on_model_unload=[failing_unload_hook],
    )

    # Load fails, then cleanup also fails
    # Should raise ModelLoadError with original error preserved
    with pytest.raises(ModelLoadError, match="Model load and cleanup failed"):
        await registry.load(load_error_model_settings)

    # Model removed from registry even though cleanup failed
    # (unregister happens before cleanup hooks)
    with pytest.raises(ModelNotFound):
        await registry.get_model()


async def test_cancelled_load_settles_and_marks_model_ready(
    model_registry: MultiModelRegistry, mocker
):
    slow_model_settings = ModelSettings(name="slow-model", implementation=SlowModel)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_load(model: SlowModel) -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(SlowModel, "load", new=controlled_load)

    load_task = asyncio.create_task(model_registry.load(slow_model_settings))
    await started.wait()

    models = list(await model_registry.get_models())
    assert not all([m.ready for m in models])
    assert len(models) == 2
    slow_model = await model_registry.get_model(slow_model_settings.name)
    assert not slow_model.ready

    # Cancellation is deferred after the load has started.
    load_task.cancel()
    await asyncio.sleep(0)
    assert not load_task.done()
    finish.set()

    await load_task

    loaded_model = await model_registry.get_model(slow_model_settings.name)
    assert loaded_model.ready
    await model_registry.unload(slow_model_settings.name)


async def test_cancelled_reload_settles_and_replaces_model(
    model_registry: MultiModelRegistry, sum_model_settings: ModelSettings, mocker
):
    sum_model_settings.implementation = SlowModel
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_load(model: SlowModel) -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(SlowModel, "load", new=controlled_load)
    old_model = await model_registry.get_model(sum_model_settings.name)
    reload_task = asyncio.create_task(model_registry.load(sum_model_settings))
    await started.wait()

    # Cancellation is deferred after the replacement load has started.
    reload_task.cancel()
    await asyncio.sleep(0)
    assert not reload_task.done()
    finish.set()

    await reload_task

    new_model = await model_registry.get_model(sum_model_settings.name)
    assert new_model is not old_model
    assert new_model.ready
    assert not old_model.ready
    await model_registry.unload(sum_model_settings.name)


async def test_unload_cancellation_settles_and_removes_model(
    model_registry: MultiModelRegistry, sum_model: MLModel, mocker
):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_unload() -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(sum_model, "unload", side_effect=controlled_unload)
    unload_task = asyncio.create_task(model_registry.unload(sum_model.name))
    await started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()
    finish.set()

    await unload_task

    with pytest.raises(ModelNotFound):
        await model_registry.get_model(sum_model.name)


async def test_unload_version_cancellation_settles_and_removes_model(
    model_registry: MultiModelRegistry, sum_model: MLModel, mocker
):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_unload() -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(sum_model, "unload", side_effect=controlled_unload)
    unload_task = asyncio.create_task(
        model_registry.unload_version(sum_model.name, sum_model.version)
    )
    await started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()
    finish.set()

    await unload_task

    with pytest.raises(ModelNotFound):
        await model_registry.get_model(sum_model.name)


async def test_single_model_registry_load_cancellation_settles(
    sum_model_settings: ModelSettings, mocker
):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_load(model: MLModel) -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(SumModel, "load", new=controlled_load)
    registry = SingleModelRegistry(sum_model_settings)
    load_task = asyncio.create_task(registry.load(sum_model_settings))
    await started.wait()
    loading_model = await registry.get_model()
    assert not loading_model.ready

    load_task.cancel()
    await asyncio.sleep(0)
    assert not load_task.done()
    finish.set()

    await load_task

    model = await registry.get_model()
    assert model.ready
    await registry.unload()


async def test_single_model_registry_reload_cancellation_settles(
    sum_model_settings: ModelSettings, mocker
):
    registry = SingleModelRegistry(sum_model_settings)
    old_model = await registry.load(sum_model_settings)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_load(model: MLModel) -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(SumModel, "load", new=controlled_load)
    reload_task = asyncio.create_task(registry.load(sum_model_settings))
    await started.wait()

    assert (await registry.get_model()) is old_model
    assert old_model.ready

    reload_task.cancel()
    await asyncio.sleep(0)
    assert not reload_task.done()
    finish.set()

    await reload_task

    new_model = await registry.get_model()
    assert new_model is not old_model
    assert new_model.ready
    assert not old_model.ready
    await registry.unload()


async def test_single_model_registry_unload_cancellation_settles(
    sum_model_settings: ModelSettings, mocker
):
    registry = SingleModelRegistry(sum_model_settings)
    model = await registry.load(sum_model_settings)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_unload() -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(model, "unload", side_effect=controlled_unload)
    unload_task = asyncio.create_task(registry.unload())
    await started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()
    finish.set()

    await unload_task

    assert registry.empty()


async def test_single_model_registry_unload_version_cancellation_settles(
    sum_model_settings: ModelSettings, mocker
):
    registry = SingleModelRegistry(sum_model_settings)
    model = await registry.load(sum_model_settings)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def controlled_unload() -> bool:
        started.set()
        await finish.wait()
        return True

    mocker.patch.object(model, "unload", side_effect=controlled_unload)
    unload_task = asyncio.create_task(registry.unload_version(model.version))
    await started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()
    finish.set()

    await unload_task

    assert registry.empty()


async def test_model_registry_allows_independent_names(
    sum_model_settings: ModelSettings,
):
    active = 0
    maximum_active = 0
    entered = 0
    release = asyncio.Event()
    both_entered = asyncio.Event()

    async def load_hook(model: MLModel) -> MLModel:
        nonlocal active, maximum_active, entered
        entered += 1
        active += 1
        maximum_active = max(maximum_active, active)
        if entered == 2:
            both_entered.set()
        await release.wait()
        active -= 1
        return model

    registry = MultiModelRegistry(on_model_load=[load_hook])
    first_settings = sum_model_settings.model_copy(deep=True)
    first_settings.name = "first-name"
    second_settings = sum_model_settings.model_copy(deep=True)
    second_settings.name = "second-name"

    first_load = asyncio.create_task(registry.load(first_settings))
    second_load = asyncio.create_task(registry.load(second_settings))
    await both_entered.wait()
    assert maximum_active == 2
    release.set()
    try:
        await asyncio.gather(first_load, second_load)
    finally:
        await asyncio.gather(
            registry.unload("first-name"),
            registry.unload("second-name"),
        )


async def test_model_registry_serializes_load_for_same_name(
    sum_model_settings: ModelSettings,
):
    active = 0
    maximum_active = 0
    entered = 0
    first_entered = asyncio.Event()
    release = asyncio.Event()

    async def load_hook(model: MLModel) -> MLModel:
        nonlocal active, maximum_active, entered
        entered += 1
        active += 1
        maximum_active = max(maximum_active, active)
        if entered == 1:
            first_entered.set()
        await release.wait()
        active -= 1
        return model

    same_name_registry = MultiModelRegistry(on_model_load=[load_hook])
    first_settings = sum_model_settings.model_copy(deep=True)
    first_settings.name = "same-name"
    assert first_settings.parameters is not None
    first_settings.parameters.version = "v1"
    second_settings = first_settings.model_copy(deep=True)
    assert second_settings.parameters is not None
    second_settings.parameters.version = "v2"

    first_load = asyncio.create_task(same_name_registry.load(first_settings))
    await first_entered.wait()
    second_load = asyncio.create_task(same_name_registry.load(second_settings))
    await asyncio.sleep(0)
    assert entered == 1
    assert maximum_active == 1
    release.set()
    try:
        await asyncio.gather(first_load, second_load)
    finally:
        await same_name_registry.unload("same-name")


async def test_model_registry_serializes_load_and_unload_for_same_name(
    sum_model_settings: ModelSettings,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    async def load_hook(model: MLModel) -> MLModel:
        load_started.set()
        await release_load.wait()
        return model

    registry = MultiModelRegistry(on_model_load=[load_hook])
    settings = sum_model_settings.model_copy(deep=True)
    settings.name = "load-unload-model"

    load_task = asyncio.create_task(registry.load(settings))
    await load_started.wait()
    unload_task = asyncio.create_task(registry.unload(settings.name))
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_load.set()
    loaded_model, _ = await asyncio.gather(load_task, unload_task)

    assert loaded_model.ready is False
    with pytest.raises(ModelNotFound):
        await registry.get_model(settings.name)


async def test_model_registry_serializes_load_and_unload_version_for_same_name(
    sum_model_settings: ModelSettings,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    async def load_hook(model: MLModel) -> MLModel:
        load_started.set()
        await release_load.wait()
        return model

    registry = MultiModelRegistry(on_model_load=[load_hook])
    settings = sum_model_settings.model_copy(deep=True)
    settings.name = "load-unload-version-model"

    load_task = asyncio.create_task(registry.load(settings))
    await load_started.wait()
    unload_task = asyncio.create_task(
        registry.unload_version(settings.name, settings.version)
    )
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_load.set()
    loaded_model, _ = await asyncio.gather(load_task, unload_task)

    assert loaded_model.ready is False
    with pytest.raises(ModelNotFound):
        await registry.get_model(settings.name)


async def test_single_model_registry_serializes_loads(
    sum_model_settings: ModelSettings,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    entered = 0
    active = 0
    maximum_active = 0

    async def load_hook(model: MLModel) -> MLModel:
        nonlocal active, entered, maximum_active
        entered += 1
        active += 1
        maximum_active = max(maximum_active, active)
        if entered == 1:
            load_started.set()
        await release_load.wait()
        active -= 1
        return model

    registry = SingleModelRegistry(
        sum_model_settings,
        on_model_load=[load_hook],
    )

    first_load = asyncio.create_task(registry.load(sum_model_settings))
    await load_started.wait()
    second_load = asyncio.create_task(registry.load(sum_model_settings))
    await asyncio.sleep(0)

    assert entered == 1
    assert maximum_active == 1
    assert not second_load.done()

    release_load.set()
    await asyncio.gather(first_load, second_load)

    assert maximum_active == 1
    await registry.unload()


async def test_single_model_registry_serializes_load_and_unload(
    sum_model_settings: ModelSettings,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    async def load_hook(model: MLModel) -> MLModel:
        load_started.set()
        await release_load.wait()
        return model

    registry = SingleModelRegistry(
        sum_model_settings,
        on_model_load=[load_hook],
    )

    load_task = asyncio.create_task(registry.load(sum_model_settings))
    await load_started.wait()
    unload_task = asyncio.create_task(registry.unload())
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_load.set()
    loaded_model, _ = await asyncio.gather(load_task, unload_task)

    assert loaded_model.ready is False
    assert registry.empty()


async def test_single_model_registry_serializes_load_and_unload_version(
    sum_model_settings: ModelSettings,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    async def load_hook(model: MLModel) -> MLModel:
        load_started.set()
        await release_load.wait()
        return model

    registry = SingleModelRegistry(
        sum_model_settings,
        on_model_load=[load_hook],
    )

    load_task = asyncio.create_task(registry.load(sum_model_settings))
    await load_started.wait()
    unload_task = asyncio.create_task(
        registry.unload_version(sum_model_settings.version)
    )
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_load.set()
    loaded_model, _ = await asyncio.gather(load_task, unload_task)

    assert loaded_model.ready is False
    assert registry.empty()


def test_model_initialiser_wraps_runtime_allowlist_value_error():
    class _InvalidModelSettings:
        name = "bad-model"

        @property
        def implementation(self):
            raise ValueError(
                "Model implementation 'malicious.CustomModel' is not trusted"
            )

    with pytest.raises(RuntimeError, match="Refused to load model 'bad-model'"):
        model_initialiser(_InvalidModelSettings())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "exc_type", [ImportError, AttributeError, OSError, ValueError, json.JSONDecodeError]
)
def test_model_initialiser_wraps_runtime_import_resolution_errors(exc_type):
    class _InvalidModelSettings:
        name = "bad-model"

        @property
        def implementation(self):
            # JSONDecodeError requires msg, doc, pos arguments
            if exc_type == json.JSONDecodeError:
                raise exc_type("failed to resolve runtime import", "", 0)
            raise exc_type("failed to resolve runtime import")

    with pytest.raises(RuntimeError, match="Refused to load model 'bad-model'"):
        model_initialiser(_InvalidModelSettings())  # type: ignore[arg-type]


async def test_unload_hook_failure_removes_model_from_registry(
    sum_model_settings: ModelSettings,
):
    """Test when unload hook fails, model is removed
    from registry and marked not ready"""
    hook_error = RuntimeError("Hook failed")

    async def failing_unload_hook(model: MLModel) -> MLModel:
        raise hook_error

    registry = SingleModelRegistry(
        sum_model_settings,
        on_model_unload=[failing_unload_hook],
    )

    # Load model first
    model = await registry.load(sum_model_settings)
    assert model.ready

    # Try to unload - should fail with ModelUnloadError wrapping hook error
    with pytest.raises(ModelUnloadError, match="Failed to unload 1 version"):
        await registry.unload()

    # Model removed from registry (unregister happens
    # before hooks), but marked not ready
    with pytest.raises(ModelNotFound):
        await registry.get_model()
    assert not model.ready


async def test_unload_model_failure_removes_model_from_registry(
    unload_error_model_settings: ModelSettings,
):
    """Test when unload raises exception, model is removed
    from registry and marked not ready"""
    registry = SingleModelRegistry(unload_error_model_settings)

    # Load model first
    model = await registry.load(unload_error_model_settings)
    assert model.ready

    # Try to unload - should fail with ModelUnloadError
    with pytest.raises(ModelUnloadError, match="Failed to unload 1 version"):
        await registry.unload()

    # Model removed from registry (unregister happens
    # before unload), but marked not ready
    with pytest.raises(ModelNotFound):
        await registry.get_model()
    assert not model.ready


async def test_unload_returns_false_removes_model_from_registry(
    unload_returns_false_model_settings: ModelSettings,
):
    """Test when unload returns False, model is removed
    from registry and marked not ready"""
    registry = SingleModelRegistry(unload_returns_false_model_settings)

    # Load model first
    model = await registry.load(unload_returns_false_model_settings)
    assert model.ready

    # Try to unload - should fail with ModelUnloadError
    with pytest.raises(ModelUnloadError, match="Failed to unload 1 version"):
        await registry.unload()

    # Model removed from registry (unregister happens
    # before unload), but marked not ready
    with pytest.raises(ModelNotFound):
        await registry.get_model()
    assert not model.ready


async def test_unload_success_removes_from_registry(
    sum_model_settings: ModelSettings,
):
    """Test that successful unload removes model from registry"""
    registry = SingleModelRegistry(sum_model_settings)

    # Load model first
    model = await registry.load(sum_model_settings)
    assert model.ready

    # Unload should succeed
    await registry.unload()

    # Model should be removed from registry
    with pytest.raises(ModelNotFound):
        await registry.get_model()


async def test_load_returns_false_raises_model_load_error():
    """Test that when model.load() returns False, ModelLoadError is raised"""
    from .fixtures import ErrorModel

    load_returns_false_settings = ModelSettings(
        name="error-model",
        implementation=ErrorModel,
        parameters=ModelParameters(load_returns_false=True),
    )

    registry = SingleModelRegistry(load_returns_false_settings)

    # Load should fail with ModelLoadError
    with pytest.raises(ModelLoadError, match="Model load returned False"):
        await registry.load(load_returns_false_settings)

    # Model should NOT be in registry after error
    with pytest.raises(ModelNotFound):
        await registry.get_model()


async def test_unload_model_unregisters_before_hooks_and_unload(
    sum_model_settings: ModelSettings,
):
    """Test that _unload_model removes model from registry before running hooks"""
    # Track whether model was in registry when hook ran
    model_in_registry_during_hook = None

    async def check_registry_hook(model: MLModel) -> MLModel:
        nonlocal model_in_registry_during_hook
        # Check if model is still in registry during hook execution
        try:
            await registry.get_model()
            model_in_registry_during_hook = True
        except ModelNotFound:
            model_in_registry_during_hook = False
        return model

    registry = SingleModelRegistry(
        sum_model_settings,
        on_model_unload=[check_registry_hook],
    )

    # Load model first
    model = await registry.load(sum_model_settings)
    assert model.ready

    # Unload model
    await registry.unload()

    # Model should NOT have been in registry when hook ran
    assert model_in_registry_during_hook is False

    # Model should not be in registry after unload
    with pytest.raises(ModelNotFound):
        await registry.get_model()


async def test_reload_new_model_load_failure_preserves_old_model(
    sum_model_settings: ModelSettings,
):
    """Test that when new model fails to load during
    reload, old model stays in registry"""
    from .fixtures import ErrorModel

    registry = SingleModelRegistry(sum_model_settings)

    # Load initial model (v1.2.3)
    old_model = await registry.load(sum_model_settings)
    assert old_model.ready
    assert old_model.version == "v1.2.3"

    # Try to reload with a model that fails to load (same version)
    failing_settings = ModelSettings(
        name=sum_model_settings.name,
        implementation=ErrorModel,
        parameters=ModelParameters(version="v1.2.3", load_error=True),
    )

    # Reload should fail
    with pytest.raises(MLServerError, match="something really bad happened"):
        await registry.load(failing_settings)

    # Old model should still be in registry and ready
    registry_model = await registry.get_model()
    assert registry_model is old_model
    assert old_model.ready
    assert old_model.version == "v1.2.3"

    # Old model should still be the default
    assert registry.default is old_model


async def test_reload_old_model_unload_failure_keeps_new_model_and_increments_metric(
    sum_model_settings: ModelSettings, mocker
):
    """Test that when old model fails to unload during
    reload, new model stays in registry and metric
    increments"""
    from .fixtures import ErrorModel

    # Load initial model that will fail on unload
    old_settings = ModelSettings(
        name="error-model",
        implementation=ErrorModel,
        parameters=ModelParameters(version="v1.2.3", unload_error=True),
    )

    registry = SingleModelRegistry(old_settings)
    old_model = await registry.load(old_settings)
    assert old_model.ready

    metric_spy = mocker.spy(SingleModelRegistry, "_increment_cleanup_failure_metric")

    # Try to reload with new model (same version, but won't fail)
    new_settings = ModelSettings(
        name="error-model",
        implementation=sum_model_settings.implementation,
        parameters=ModelParameters(version="v1.2.3"),
    )

    # Reload should fail because old model unload fails
    with pytest.raises(MLServerError, match="something really bad happened"):
        await registry.load(new_settings)

    # New model should be in registry and ready
    registry_model = await registry.get_model()
    assert registry_model is not old_model
    assert registry_model.ready
    assert registry_model.version == "v1.2.3"

    # Metric should have been incremented
    metric_spy.assert_called_once()


async def test_version_matching_same_version_triggers_reload(
    sum_model_settings: ModelSettings, mocker
):
    """Test that loading same version triggers reload,
    different version triggers load"""
    # Create a registry with mocked hooks
    unload_hook = mocker.AsyncMock(side_effect=lambda m: m)
    load_hook = mocker.AsyncMock(side_effect=lambda m: m)

    registry = SingleModelRegistry(
        sum_model_settings,
        on_model_load=[load_hook],
        on_model_unload=[unload_hook],
    )

    # Load initial model (v1.2.3)
    model_v1 = await registry.load(sum_model_settings)
    assert model_v1.version == "v1.2.3"
    load_hook.assert_called_once()
    load_hook.reset_mock()

    # Test Case A: Load same version should trigger reload
    same_version_settings = sum_model_settings.copy(deep=True)
    model_v1_reloaded = await registry.load(same_version_settings)

    # Should have called both load hook (on new model) and unload hook (on old model).
    # Load hooks run first (on new model), then unload hooks run (on old model).
    load_hook.assert_called_once()
    unload_hook.assert_called_once()
    assert model_v1_reloaded.version == "v1.2.3"

    load_hook.reset_mock()
    unload_hook.reset_mock()

    # Test Case B: Load different version should trigger load (not reload)
    different_version_settings = sum_model_settings.copy(deep=True)
    assert different_version_settings.parameters is not None
    different_version_settings.parameters.version = "v2.0.0"
    model_v2 = await registry.load(different_version_settings)

    # Should have called load hook (not reload hook)
    load_hook.assert_called_once()
    unload_hook.assert_not_called()
    assert model_v2.version == "v2.0.0"

    # Both models should be in registry
    models = await registry.get_models()
    assert len(models) == 2


async def test_unload_all_versions_attempts_all_with_failures(
    sum_model_settings: ModelSettings,
):
    """Test bulk unload uses return_exceptions and attempts
    all versions even when some fail"""
    from .fixtures import ErrorModel

    # Create registry with mix of normal and error models
    registry = MultiModelRegistry()

    # Load 2 normal versions
    sum_model_settings.name = "bulk-model"
    for version in ["v1", "v3"]:
        settings = sum_model_settings.copy(deep=True)
        assert settings.parameters is not None
        settings.parameters.version = version
        await registry.load(settings)

    # Load 1 error model (v2) that will fail on unload
    error_settings = ModelSettings(
        name="bulk-model",
        implementation=ErrorModel,
        parameters=ModelParameters(version="v2", unload_error=True),
    )
    await registry.load(error_settings)

    # Verify all 3 loaded
    models = await registry.get_models("bulk-model")
    assert len(models) == 3

    # Unload all versions - should attempt all despite v2 failure
    with pytest.raises(ModelUnloadError, match="Failed to unload 1 version"):
        await registry.unload("bulk-model")

    # Model completely removed from registry, even though unload failed
    with pytest.raises(ModelNotFound):
        await registry.get_models("bulk-model")
