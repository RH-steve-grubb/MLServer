import pytest
from pytest_lazyfixture import lazy_fixture

from httpx import AsyncClient
from httpx_sse import aconnect_sse

from mlserver import __version__
from mlserver.batching.adaptive import AdaptiveBatcher
from mlserver.settings import ModelSettings
from mlserver.model import MLModel
from mlserver.registry import MultiModelRegistry
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    MetadataServerResponse,
    MetadataModelResponse,
    Parameters,
    RequestInput,
    TensorData,
    RepositoryIndexRequest,
)
from mlserver.cloudevents import (
    CLOUDEVENTS_HEADER_SPECVERSION_DEFAULT,
    CLOUDEVENTS_HEADER_SPECVERSION,
)

from ..fixtures import EchoModel


async def test_live(rest_client: AsyncClient):
    endpoint = "/v2/health/live"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200


async def test_ready(rest_client: AsyncClient):
    endpoint = "/v2/health/ready"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200


async def test_model_ready(rest_client: AsyncClient, sum_model: MLModel):
    endpoint = f"/v2/models/{sum_model.name}/versions/{sum_model.version}/ready"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200


async def test_metadata(rest_client: AsyncClient):
    endpoint = "/v2"
    response = await rest_client.get(endpoint)

    metadata = MetadataServerResponse.model_validate(response.json())

    assert metadata.name == "mlserver"
    assert metadata.version == __version__
    assert metadata.extensions == ["model_repository", "runtime_security"]


async def test_openapi(rest_client: AsyncClient):
    endpoint = "/v2/docs"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "html" in response.headers["content-type"]


async def test_docs(rest_client: AsyncClient):
    endpoint = "/v2/docs/dataplane.json"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "openapi" in response.json()


async def test_model_metadata(
    rest_client: AsyncClient, sum_model_settings: ModelSettings
):
    endpoint = f"v2/models/{sum_model_settings.name}"
    response = await rest_client.get(endpoint)

    metadata = MetadataModelResponse.model_validate(response.json())

    assert metadata.name == sum_model_settings.name
    assert metadata.platform == sum_model_settings.platform
    assert metadata.versions == sum_model_settings.versions
    assert metadata.inputs == sum_model_settings.inputs


@pytest.mark.parametrize(
    "model_name,model_version", [("sum-model", "v1.2.3"), ("sum-model", None)]
)
async def test_model_openapi(
    rest_client: AsyncClient, model_name: str, model_version: str | None
):
    endpoint = f"/v2/models/{model_name}/docs/dataplane.json"
    if model_version is not None:
        endpoint = (
            f"/v2/models/{model_name}/versions/{model_version}/docs/dataplane.json"
        )
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "openapi" in response.json()


@pytest.mark.parametrize(
    "model_name,model_version", [("sum-model", "v1.2.3"), ("sum-model", None)]
)
async def test_model_docs(
    rest_client: AsyncClient, model_name: str, model_version: str | None
):
    endpoint = f"/v2/models/{model_name}/docs"
    if model_version is not None:
        endpoint = f"/v2/models/{model_name}/versions/{model_version}/docs"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    assert "html" in response.headers["content-type"]


@pytest.mark.parametrize(
    "model_name,model_version", [("sum-model", "v1.2.3"), ("sum-model", None)]
)
async def test_infer(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    model_name: str,
    model_version: str | None,
):
    endpoint = f"/v2/models/{model_name}/infer"
    if model_version is not None:
        endpoint = f"/v2/models/{model_name}/versions/{model_version}/infer"

    response = await rest_client.post(endpoint, json=inference_request.model_dump())
    assert response.status_code == 200

    prediction = InferenceResponse.model_validate(response.json())
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0].data == TensorData(root=[6])


async def test_infer_single_adaptive_batch_preserves_output_parameters(
    rest_client: AsyncClient, model_registry: MultiModelRegistry
):
    model_settings = ModelSettings(
        name="echo-model",
        implementation=EchoModel,
        parallel_workers=0,
        max_batch_size=32,
        max_batch_time=0.01,
    )
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="docs",
                datatype="INT32",
                shape=[2],
                data=[10, 11],
                parameters=Parameters(id="123", tags=["123", "456"], count=123),
            )
        ]
    )

    await model_registry.load(model_settings)
    try:
        model = await model_registry.get_model(model_settings.name)
        assert AdaptiveBatcher.get_batcher(model) is not None

        response = await rest_client.post(
            "/v2/models/echo-model/infer", json=request.model_dump()
        )

        assert response.status_code == 200
        prediction = InferenceResponse.model_validate(response.json())
        output = prediction.outputs[0]
        assert output.data == TensorData(root=[10, 11])
        assert output.parameters is not None
        assert output.parameters.model_dump(exclude_none=True) == {
            "id": "123",
            "tags": ["123", "456"],
            "count": 123,
        }
    finally:
        await model_registry.unload(model_settings.name)


@pytest.mark.parametrize("sum_model", [lazy_fixture("text_model")])
@pytest.mark.parametrize(
    "model_name,model_version", [("text-model", "v1.2.3"), ("text-model", None)]
)
async def test_generate(
    rest_client: AsyncClient,
    generate_request: InferenceRequest,
    model_name: str,
    model_version: str | None,
):
    endpoint = f"/v2/models/{model_name}/generate"
    if model_version is not None:
        endpoint = f"/v2/models/{model_name}/versions/{model_version}/generate"

    response = await rest_client.post(endpoint, json=generate_request.model_dump())
    assert response.status_code == 200

    prediction = InferenceResponse.model_validate(response.json())
    assert len(prediction.outputs) == 1
    assert prediction.outputs[0].data == TensorData(
        root=["What is the capital of France?"]
    )


@pytest.mark.parametrize("settings", [lazy_fixture("settings_stream")])
@pytest.mark.parametrize("sum_model", [lazy_fixture("text_stream_model")])
@pytest.mark.parametrize("endpoint", ["generate_stream", "infer_stream"])
async def test_generate_stream(
    rest_client: AsyncClient,
    generate_request: InferenceRequest,
    text_stream_model: MLModel,
    endpoint: str,
):
    endpoint = f"/v2/models/{text_stream_model.name}/{endpoint}"
    conn = aconnect_sse(
        rest_client, "POST", endpoint, json=generate_request.model_dump()
    )
    ref_text = ["What", " is", " the", " capital", " of", " France?"]

    async with conn as stream:
        i = 0
        async for response in stream.aiter_sse():
            prediction = InferenceResponse.model_validate(response.json())
            assert len(prediction.outputs) == 1
            assert prediction.outputs[0].data == TensorData(root=[ref_text[i]])
            i += 1


async def test_infer_headers(
    rest_client: AsyncClient,
    inference_request: InferenceRequest,
    sum_model_settings: ModelSettings,
):
    endpoint = f"/v2/models/{sum_model_settings.name}/infer"
    response = await rest_client.post(
        endpoint, json=inference_request.model_dump(), headers={"x-foo": "bar"}
    )

    assert response.status_code == 200
    assert "x-foo" in response.headers
    assert response.headers["x-foo"] == "bar"

    assert CLOUDEVENTS_HEADER_SPECVERSION in response.headers
    assert (
        response.headers[CLOUDEVENTS_HEADER_SPECVERSION]
        == CLOUDEVENTS_HEADER_SPECVERSION_DEFAULT
    )


@pytest.mark.parametrize("endpoint", ["infer", "generate"])
async def test_infer_error(
    rest_client: AsyncClient, inference_request: InferenceRequest, endpoint: str
):
    endpoint = f"/v2/models/my-model/versions/v0/{endpoint}"
    response = await rest_client.post(endpoint, json=inference_request.model_dump())

    assert response.status_code == 404
    assert response.json()["error"] == "Model my-model with version v0 not found"


async def test_model_repository_index_excludes_registry_models_by_default(
    rest_client: AsyncClient,
    repository_index_request: RepositoryIndexRequest,
    model_registry,
    sum_model_settings: ModelSettings,
):
    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)

    try:
        response = await rest_client.post(
            "/v2/repository/index",
            json=repository_index_request.model_dump(
                exclude={"include_registry_models"}
            ),
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert response.status_code == 200
    assert [model["name"] for model in response.json()] == [sum_model_settings.name]


async def test_model_repository_index_includes_registry_models(
    rest_client: AsyncClient,
    repository_index_request: RepositoryIndexRequest,
    model_registry,
    sum_model_settings: ModelSettings,
):
    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)
    repository_index_request.include_registry_models = True

    try:
        response = await rest_client.post(
            "/v2/repository/index",
            json=repository_index_request.model_dump(),
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert response.status_code == 200
    assert {model["name"] for model in response.json()} == {
        sum_model_settings.name,
        registry_only_settings.name,
    }


@pytest.mark.parametrize("ready", [False, True])
async def test_model_repository_index_ready_filters_selected_models(
    rest_client: AsyncClient,
    repository_index_request: RepositoryIndexRequest,
    model_registry,
    sum_model_settings: ModelSettings,
    ready: bool,
):
    await model_registry.unload(sum_model_settings.name)

    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)
    repository_index_request.include_registry_models = True
    repository_index_request.ready = ready

    try:
        response = await rest_client.post(
            "/v2/repository/index",
            json=repository_index_request.model_dump(),
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert response.status_code == 200
    expected_names = {registry_only_settings.name}
    if not ready:
        expected_names.add(sum_model_settings.name)

    assert {model["name"] for model in response.json()} == expected_names


async def test_model_repository_unload(
    rest_client: AsyncClient, sum_model_settings: ModelSettings
):
    endpoint = f"/v2/repository/models/{sum_model_settings.name}/unload"
    response = await rest_client.post(endpoint)

    assert response.status_code == 200

    model_metadata = await rest_client.get(f"/v2/models/{sum_model_settings.name}")
    assert model_metadata.status_code == 404


async def test_model_repository_load(
    rest_client: AsyncClient,
    sum_model_settings: ModelSettings,
):
    await rest_client.post(f"/v2/repository/models/{sum_model_settings.name}/unload")

    endpoint = f"/v2/repository/models/{sum_model_settings.name}/load"
    response = await rest_client.post(endpoint)

    assert response.status_code == 200

    model_metadata = await rest_client.get(f"/v2/models/{sum_model_settings.name}")
    assert model_metadata.status_code == 200


async def test_model_repository_load_error(
    rest_client: AsyncClient, sum_model_settings: ModelSettings
):
    endpoint = "/v2/repository/models/my-model/load"
    response = await rest_client.post(endpoint)

    assert response.status_code == 404
    assert response.json()["error"] == "Model my-model not found"


async def test_infer_invalid_datatype_error(
    rest_client, inference_request_invalid_datatype, datatype_error_message
):
    endpoint = "/v2/models/sum-model/infer"
    response = await rest_client.post(endpoint, json=inference_request_invalid_datatype)

    assert response.status_code == 422

    assert response.json()["detail"][0]["msg"] == datatype_error_message


async def test_runtimes_endpoint_production_mode(rest_client):
    """Test GET /v2/runtimes in PRODUCTION mode."""
    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "PRODUCTION"
    assert "allowed_model_implementations" in data
    assert isinstance(data["allowed_model_implementations"], list)
    assert len(data["allowed_model_implementations"]) > 0
    # Should include builtin runtimes
    assert "mlserver_sklearn.SKLearnModel" in data["allowed_model_implementations"]


async def test_runtimes_endpoint_development_mode(development_mode, rest_client):
    """Test GET /v2/runtimes in DEVELOPMENT mode."""
    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "DEVELOPMENT"
    # In DEVELOPMENT mode, allowed_model_implementations field should not be in response
    assert "allowed_model_implementations" not in data


async def test_runtimes_endpoint_schema_validation(rest_client):
    """Test that /v2/runtimes response matches RuntimeSecurityResponse schema."""
    from mlserver.types import RuntimeSecurityResponse

    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    # In PRODUCTION mode, field should be present in JSON response
    assert "allowed_model_implementations" in data

    # Should be able to parse as RuntimeSecurityResponse
    runtime_response = RuntimeSecurityResponse.model_validate(data)
    assert runtime_response.mode == "PRODUCTION"
    assert runtime_response.allowed_model_implementations is not None


async def test_runtimes_endpoint_includes_all_implementations(rest_client):
    """Test that /v2/runtimes includes all allowlisted implementations."""
    import mlserver.settings as mlserver_settings
    from conftest import TEST_ONLY_EXTRA_IMPLEMENTATIONS

    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()
    implementations_set = set(data["allowed_model_implementations"])

    # Should include all builtin runtimes
    for builtin in mlserver_settings.ALLOWED_MODEL_IMPLEMENTATIONS:
        assert builtin in implementations_set

    # Should include test-only implementations
    for test_impl in TEST_ONLY_EXTRA_IMPLEMENTATIONS:
        assert test_impl in implementations_set


async def test_runtimes_endpoint_sorted_list(rest_client):
    """Test that /v2/runtimes returns a sorted list in PRODUCTION mode."""
    endpoint = "/v2/runtimes"
    response = await rest_client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    implementations = data["allowed_model_implementations"]
    # Verify list is sorted
    sorted_list = sorted(implementations)
    assert implementations == sorted_list
