"""Normal app construction tests for opt-in worker JSON; no network listeners."""

import json
from types import MethodType
from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient
import pytest

from mlserver import ModelSettings, Settings
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.model import MLModel
from mlserver.rest.app import APIRoute, create_app
from mlserver.rest.endpoints import Endpoints
from mlserver.rest.worker import RawEndpoints
from mlserver.types import InferenceRequest


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.parallel_workers = 0
    settings.gzip_enabled = False
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


def resolve(schema: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    if "$ref" not in schema:
        return schema
    value = document
    for key in schema["$ref"].removeprefix("#/").split("/"):
        value = value[key]
    return value


def check_references(value: Any, document: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if "$ref" in value:
            resolve(value, document)
        for item in value.values():
            check_references(item, document)
    elif isinstance(value, list):
        for item in value:
            check_references(item, document)


@pytest.mark.parametrize("enabled", [False, True])
async def test_startup_flag_preserves_nonparallel_inference_and_headers(
    enabled: bool,
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    inference_request: InferenceRequest,
):
    settings.rest_worker_json = enabled
    app = create_app(settings, data_plane, model_repository_handlers)
    inference_request.id = "startup-request"
    async with AsyncClient(app=app, base_url="http://test") as client:
        for path in (
            "/v2/models/sum-model/infer",
            "/v2/models/sum-model/versions/v1.2.3/infer",
        ):
            response = await client.post(
                path,
                json=inference_request.model_dump(),
                headers={"x-startup": "echo"},
            )
            assert response.status_code == 200
            assert response.json()["id"] == "startup-request"
            assert response.json()["outputs"][0]["data"] == [6.0]
            assert response.headers["x-startup"] == "echo"


@pytest.mark.parametrize("enabled", [False, True])
def test_only_infer_routes_change_and_preserve_openapi_contract(
    enabled: bool,
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
):
    settings.rest_worker_json = enabled
    app = create_app(settings, data_plane, model_repository_handlers)
    schema = app.openapi()
    routes = {route.path: route for route in app.routes if isinstance(route, APIRoute)}
    for path in (
        "/v2/models/{model_name}/infer",
        "/v2/models/{model_name}/versions/{model_version}/infer",
    ):
        endpoint = routes[path].endpoint
        assert isinstance(endpoint, MethodType)
        assert isinstance(endpoint.__self__, RawEndpoints if enabled else Endpoints)
        operation = schema["paths"][path]["post"]
        assert operation["requestBody"]["required"] is True
        request = resolve(
            operation["requestBody"]["content"]["application/json"]["schema"], schema
        )
        response = resolve(
            operation["responses"]["200"]["content"]["application/json"]["schema"],
            schema,
        )
        assert request["type"] == "object"
        assert "inputs" in request["required"]
        assert {"id", "parameters", "inputs", "outputs"} <= set(request["properties"])
        assert response["type"] == "object"
        assert {"model_name", "outputs"} <= set(response["required"])
        assert {"id", "parameters", "outputs"} <= set(response["properties"])
    check_references(schema, schema)
    for suffix, function in (
        ("generate", Endpoints.infer),
        ("infer_stream", Endpoints.infer_stream),
        ("generate_stream", Endpoints.infer_stream),
    ):
        for prefix in (
            "/v2/models/{model_name}",
            "/v2/models/{model_name}/versions/{model_version}",
        ):
            endpoint = routes[f"{prefix}/{suffix}"].endpoint
            assert isinstance(endpoint, MethodType)
            assert isinstance(endpoint.__self__, Endpoints)
            assert endpoint.__func__ is function


@pytest.mark.parametrize("enabled", [False, True])
async def test_generate_and_stream_requests_keep_the_standard_paths(
    enabled: bool,
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    inference_request: InferenceRequest,
):
    settings.rest_worker_json = enabled
    app = create_app(settings, data_plane, model_repository_handlers)
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/v2/models/sum-model/generate", json=inference_request.model_dump()
        )
        assert response.status_code == 200
        assert response.json()["outputs"][0]["data"] == [6.0]
        for path in ("infer_stream", "generate_stream"):
            response = await client.post(
                f"/v2/models/sum-model/{path}", json=inference_request.model_dump()
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            record = response.text.removeprefix("data: ").strip()
            assert json.loads(record)["outputs"][0]["data"] == [6.0]


@pytest.mark.parametrize("state", ["unknown", "not-ready"])
@pytest.mark.parametrize("body", [b"", b"{", b"null", b"{}", None])
async def test_startup_flag_preserves_validation_and_model_error_precedence(
    state: str,
    body: bytes | None,
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    name = "missing-model" if state == "unknown" else sum_model.name
    sum_model.ready = state != "not-ready"
    if body is None:
        body = inference_request.model_dump_json().encode()
    responses = []
    for enabled in (False, True):
        settings.rest_worker_json = enabled
        app: FastAPI = create_app(settings, data_plane, model_repository_handlers)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                f"/v2/models/{name}/infer",
                content=body,
                headers={"content-type": "application/json"},
            )
        responses.append((response.status_code, response.json()))
    assert responses[0] == responses[1]
