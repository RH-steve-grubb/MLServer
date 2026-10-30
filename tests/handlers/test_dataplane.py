import pytest
import uuid

from mlserver.errors import ModelNotReady
from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import MetadataTensor, TensorData

from ..fixtures import SumModel


@pytest.mark.parametrize("ready", [True, False])
async def test_ready(data_plane, model_registry, ready):
    model_settings = ModelSettings(
        name="sum-model-2",
        parameters=ModelParameters(version="v1.2.3"),
        implementation=SumModel,
    )
    new_model = await model_registry.load(model_settings)

    new_model.ready = ready

    all_ready = await data_plane.ready()

    assert all_ready == ready


@pytest.mark.parametrize("strict_readiness", [True, False])
async def test_ready_returns_false_when_startup_not_complete(
    data_plane_during_startup, strict_readiness
):
    """
    Health check returns False when startup hasn't completed.
    This is true in both strict and lenient mode - strict_readiness is ignored
    during startup. Once startup_complete() is called, the readiness settings apply.
    """
    data_plane_during_startup._settings.strict_readiness = strict_readiness

    assert not await data_plane_during_startup.ready()

    data_plane_during_startup.startup_complete()

    assert await data_plane_during_startup.ready()


async def test_ready_with_failed_dynamic_model(
    data_plane, model_registry, load_error_model_settings
):
    """
    Test that server ready check returns True when a model fails to load.
    Failed models are removed from registry, so they don't affect health.

    Fixture already loads sum-model which is ready=True.
    """
    from mlserver.errors import MLServerError, ModelNotFound

    # Verify we start in ready state (sum-model from fixture is ready)
    assert await data_plane.ready()

    # Load one failing model (removed from registry on failure)
    with pytest.raises(MLServerError):
        await model_registry.load(load_error_model_settings)

    # Health check should return True because error-model was removed
    # Only success-model remains, and it's ready
    all_ready = await data_plane.ready()
    assert all_ready

    # Verify the failed model is NOT in registry
    models = list(await model_registry.get_models())
    assert len(models) == 1

    with pytest.raises(ModelNotFound):
        await model_registry.get_model(load_error_model_settings.name)


@pytest.mark.parametrize("strict_readiness", [True, False])
async def test_ready_with_empty_registry_default(
    settings, data_plane, model_registry, strict_readiness
):
    """
    Test that empty registry returns True when empty_registry_readiness=True
    (default behavior), in both strict and lenient mode.
    """
    data_plane._settings.strict_readiness = strict_readiness

    # Verify default setting
    assert settings.empty_registry_readiness is True

    # Unload all models to make registry empty
    await model_registry.unload("sum-model")

    # Verify registry is empty
    models = list(await model_registry.get_models())
    assert len(models) == 0

    # Health check should return True (default behavior)
    all_ready = await data_plane.ready()
    assert all_ready


@pytest.mark.parametrize("strict_readiness", [True, False])
async def test_ready_with_empty_registry_disabled(
    settings, data_plane, model_registry, strict_readiness
):
    """
    Test that empty registry returns False when empty_registry_readiness=False,
    in both strict and lenient mode.
    """
    data_plane._settings.strict_readiness = strict_readiness

    # Configure to report not ready when registry is empty
    data_plane._settings.empty_registry_readiness = False

    # Unload all models to make registry empty
    await model_registry.unload("sum-model")

    # Verify registry is empty
    models = list(await model_registry.get_models())
    assert len(models) == 0

    # Health check should return False
    all_ready = await data_plane.ready()
    assert not all_ready


@pytest.mark.parametrize("strict_readiness", [True, False])
async def test_ready_with_non_empty_registry_flag_ignored(
    data_plane, model_registry, sum_model, strict_readiness
):
    """
    Test that empty_registry_readiness flag doesn't affect behavior when
    registry has models, in both strict and lenient mode. The flag should ONLY
    matter when registry is empty.
    """
    data_plane._settings.strict_readiness = strict_readiness

    # Set flag to False (would make empty registry report not ready)
    data_plane._settings.empty_registry_readiness = False

    # Verify registry has models
    models = list(await model_registry.get_models())
    assert len(models) == 1

    # Health check should return True
    all_ready = await data_plane.ready()
    assert all_ready


async def test_ready_strict_mode_with_mixed_states(
    data_plane, model_registry, simple_model_settings
):
    """
    Test that strict_readiness=True (default) requires ALL models to be ready.
    """

    # Verify strict_readiness is True by default
    assert data_plane._settings.strict_readiness is True

    # Load a model and mark it as not ready
    loaded_model = await model_registry.load(simple_model_settings)
    loaded_model.ready = False

    # Health check should return False (strict mode: all must be ready)
    all_ready = await data_plane.ready()
    assert not all_ready

    # Verify: sum-model (ready=True) + simple-model (ready=False)
    models = list(await model_registry.get_models())
    assert len(models) == 2
    assert models[0].ready != models[1].ready  # One ready, one not


async def test_ready_lenient_mode_after_startup(
    data_plane, model_registry, simple_model_settings
):
    """
    Test that strict_readiness=False allows health check to pass even if
    some models are not ready (after startup completes).
    """

    # Enable lenient mode
    data_plane._settings.strict_readiness = False

    # Load a model and mark it as not ready
    loaded_model = await model_registry.load(simple_model_settings)
    loaded_model.ready = False

    # Health check should return True (lenient mode: model state is ignored)
    all_ready = await data_plane.ready()
    assert all_ready

    # Verify: sum-model (ready=True) + simple-model (ready=False)
    models = list(await model_registry.get_models())
    assert len(models) == 2
    ready_count = sum(1 for m in models if m.ready)
    assert ready_count == 1  # Only one model ready


async def test_ready_lenient_mode_with_no_ready_models(
    data_plane, model_registry, sum_model
):
    """
    Test that strict_readiness=False returns True even when NO models are
    ready, so that model lifecycle requests are never blocked by readiness.
    """
    # Enable lenient mode
    data_plane._settings.strict_readiness = False

    # Mark all models as not ready
    sum_model.ready = False

    # Health check should return True (lenient mode: model state is ignored)
    ready = await data_plane.ready()
    assert ready


@pytest.mark.parametrize("ready", [True, False])
async def test_model_ready(data_plane, sum_model, ready):
    sum_model.ready = ready
    model_ready = await data_plane.model_ready(sum_model.name, sum_model.version)

    assert model_ready == ready


@pytest.mark.parametrize(
    "server_name,server_version,extensions",
    [(None, None, None), ("my-server", "v2", ["foo", "bar"])],
)
async def test_metadata(settings, data_plane, server_name, server_version, extensions):
    if server_name is not None:
        settings.server_name = server_name

    if server_version is not None:
        settings.server_version = server_version

    if extensions is not None:
        settings.extensions = extensions

    metadata = await data_plane.metadata()

    assert metadata.name == settings.server_name
    assert metadata.version == settings.server_version
    # Built-in extensions are always present
    expected_extensions = ["model_repository", "runtime_security"]
    if extensions is not None:
        expected_extensions += extensions
    assert metadata.extensions == expected_extensions


@pytest.mark.parametrize(
    "platform,versions,inputs",
    [
        (None, None, None),
        ("sklearn", ["sklearn/0.22.3"], None),
        (
            "xgboost",
            ["xgboost/1.1.0"],
            [MetadataTensor(name="input-0", datatype="FP32", shape=[6, 7])],
        ),
    ],
)
async def test_model_metadata(
    sum_model_settings, data_plane, platform, versions, inputs
):
    if platform is not None:
        sum_model_settings.platform = platform

    if versions is not None:
        sum_model_settings.versions = versions

    if inputs is not None:
        sum_model_settings.inputs = inputs

    metadata = await data_plane.model_metadata(
        name=sum_model_settings.name, version=sum_model_settings.parameters.version
    )

    assert metadata.name == sum_model_settings.name
    assert metadata.platform == sum_model_settings.platform
    assert metadata.versions == sum_model_settings.versions
    assert metadata.inputs == sum_model_settings.inputs


async def test_infer(data_plane, sum_model, inference_request):
    prediction = await data_plane.infer(
        payload=inference_request, name=sum_model.name, version=sum_model.version
    )

    assert len(prediction.outputs) == 1
    assert prediction.outputs[0].data == TensorData(root=[6])


async def test_infer_stream(data_plane, text_stream_model, generate_request):
    async def streamed_request(request):
        yield request

    stream = data_plane.infer_stream(
        payloads=streamed_request(generate_request),
        name=text_stream_model.name,
        version=text_stream_model.version,
    )

    completion = [tok async for tok in stream]
    assert len(completion) == 6

    concat_completion = b"".join([tok.outputs[0].data.root[0] for tok in completion])
    assert concat_completion == b"What is the capital of France?"


async def test_infer_error_not_ready(data_plane, sum_model, inference_request):
    sum_model.ready = False
    with pytest.raises(ModelNotReady):
        await data_plane.infer(payload=inference_request, name=sum_model.name)

    sum_model.ready = True
    prediction = await data_plane.infer(payload=inference_request, name=sum_model.name)
    assert len(prediction.outputs) == 1


async def test_infer_generates_uuid(data_plane, sum_model, inference_request):
    inference_request.id = None
    prediction = await data_plane.infer(
        payload=inference_request, name=sum_model.name, version=sum_model.version
    )

    assert prediction.id is not None
    assert prediction.id == str(uuid.UUID(prediction.id))


async def test_runtimes_handler_production_mode(data_plane):
    """Test DataPlane.runtimes() in PRODUCTION mode."""
    response = await data_plane.runtimes()

    assert response.mode == "PRODUCTION"
    assert response.allowed_model_implementations is not None
    assert isinstance(response.allowed_model_implementations, list)
    assert len(response.allowed_model_implementations) > 0
    # Should include builtin runtimes
    assert "mlserver_sklearn.SKLearnModel" in response.allowed_model_implementations


async def test_runtimes_handler_development_mode(development_mode, data_plane):
    """Test DataPlane.runtimes() in DEVELOPMENT mode."""
    response = await data_plane.runtimes()

    assert response.mode == "DEVELOPMENT"
    assert response.allowed_model_implementations is None


async def test_runtimes_handler_empty_allowlist(empty_allowlist_mode):
    """Test DataPlane.runtimes() with empty allowlist.

    Note: We skip the data_plane fixture since empty_allowlist_mode prevents
    models from loading. We call the method directly instead.
    """
    from mlserver.handlers import DataPlane

    # Call runtimes() static method - it doesn't need DataPlane instance
    # since it only reads from settings files
    response = await DataPlane.runtimes()

    assert response.mode == "PRODUCTION"
    assert response.allowed_model_implementations == []


async def test_runtimes_handler_returns_sorted_list(data_plane):
    """Test that DataPlane.runtimes() returns a sorted list in PRODUCTION mode."""
    response = await data_plane.runtimes()

    assert response.allowed_model_implementations is not None
    # Verify list is sorted
    sorted_list = sorted(response.allowed_model_implementations)
    assert response.allowed_model_implementations == sorted_list


async def test_runtimes_handler_includes_all_implementations(data_plane):
    """Test that DataPlane.runtimes() includes all allowlisted implementations."""
    import mlserver.settings as mlserver_settings
    from conftest import TEST_ONLY_EXTRA_IMPLEMENTATIONS

    response = await data_plane.runtimes()

    assert response.allowed_model_implementations is not None
    implementations_set = set(response.allowed_model_implementations)

    # Should include all builtin runtimes
    for builtin in mlserver_settings.ALLOWED_MODEL_IMPLEMENTATIONS:
        assert builtin in implementations_set

    # Should include test-only implementations
    for test_impl in TEST_ONLY_EXTRA_IMPLEMENTATIONS:
        assert test_impl in implementations_set


async def test_runtimes_handler_returns_correct_type(data_plane):
    """Test that DataPlane.runtimes() returns RuntimeSecurityResponse type."""
    from mlserver.types import RuntimeSecurityResponse

    response = await data_plane.runtimes()

    assert isinstance(response, RuntimeSecurityResponse)
