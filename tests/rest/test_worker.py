import asyncio
import json
from multiprocessing.reduction import ForkingPickler
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from httpx import AsyncClient
import pytest
from pytest_mock import MockerFixture
from starlette.requests import Request

from mlserver import ModelSettings, Settings
from mlserver.context import model_name_var, model_version_var
from mlserver.errors import ModelNotFound, ModelNotReady
from mlserver.handlers import DataPlane
from mlserver.model import MLModel
from mlserver.parallel.errors import WorkerError
from mlserver.parallel.messages import ModelRequestMessage, ModelResponseMessage
from mlserver.parallel.model import ParallelModel
from mlserver.parallel.worker import Worker
from mlserver.registry import MultiModelRegistry
from mlserver.rest.app import APIRoute
from mlserver.rest import worker as raw
from mlserver.types import InferenceRequest, InferenceResponse


@pytest.fixture
def settings(settings: Settings) -> Settings:
    # These tests exercise the dispatch boundary without spawning subprocesses.
    settings.parallel_workers = 0
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


def request(body: bytes, headers: dict[str, str] | None = None) -> Request:
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/infer",
        "headers": [
            (key.encode(), value.encode()) for key, value in (headers or {}).items()
        ],
    }
    return Request(scope, receive)


def message(body: bytes, model: MLModel) -> ModelRequestMessage:
    return ModelRequestMessage(
        id="dispatch-id",
        model_name=model.name,
        model_version=model.version,
        method_name=raw.RAW_INFER_METHOD,
        method_args=[body, {"content-type": "application/json", "x-test": "echo-me"}],
    )


def parallel(model: MLModel) -> ParallelModel:
    dispatcher = Mock()
    dispatcher.dispatch_request = AsyncMock(
        return_value=ModelResponseMessage(
            return_value=raw.RawReply(
                b'{"id":"from-worker"}', {"x-reply": "yes"}, worker_pid=os.getpid() + 1
            )
        )
    )
    wrapped = ParallelModel(model, dispatcher)
    wrapped.ready = True
    return wrapped


def dispatch_mock(model: ParallelModel) -> AsyncMock:
    method = model._dispatcher.dispatch_request
    assert isinstance(method, AsyncMock)
    return method


@pytest.mark.parametrize(
    "body", [b"", b"{", b"{}", b"null", b"[]", b'{"inputs":"not-a-list"}']
)
async def test_validation_matches_original_fastapi_response(body: bytes):
    async def original(payload: InferenceRequest) -> InferenceRequest:
        return payload

    app = FastAPI(routes=[APIRoute("/infer", original, methods=["POST"])])
    async with AsyncClient(app=app, base_url="http://test") as client:
        expected = await client.post(
            "/infer", content=body, headers={"content-type": "application/json"}
        )
    reply = raw.parse_body(body)
    assert isinstance(reply, raw.RawReply)
    assert reply.status == expected.status_code == 422
    assert json.loads(reply.body) == expected.json()


@pytest.mark.parametrize("content_type", ["text/plain", "application/octet-stream"])
async def test_non_json_content_type_retains_original_validation(
    content_type: str, inference_request: InferenceRequest
):
    async def original(payload: InferenceRequest) -> InferenceRequest:
        return payload

    app = FastAPI(routes=[APIRoute("/infer", original, methods=["POST"])])
    body = inference_request.model_dump_json().encode()
    async with AsyncClient(app=app, base_url="http://test") as client:
        expected = await client.post(
            "/infer", content=body, headers={"content-type": content_type}
        )
    reply = raw.parse_body(body, raw._json_content({"content-type": content_type}))
    assert isinstance(reply, raw.RawReply)
    assert reply.status == expected.status_code == 422
    assert json.loads(reply.body) == expected.json()


@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "application/json",
        "application/json; charset=utf-8",
        "Application/Vnd.Test+Json",
    ],
)
def test_json_content_types_keep_normal_input_schema(
    content_type: str | None, inference_request: InferenceRequest
):
    headers = {} if content_type is None else {"content-type": content_type}
    body = inference_request.model_dump_json().encode()
    payload = raw.parse_body(body, raw._json_content(headers))
    assert payload == inference_request


async def test_worker_preserves_dataplane_context_headers_ids_outputs_and_metrics(
    settings: Settings,
    model_registry: MultiModelRegistry,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    inference_request.id = "inference-id"
    worker = cast(
        Worker, SimpleNamespace(_settings=settings, _model_registry=model_registry)
    )
    original_predict = sum_model.predict
    contexts = []

    async def predict(payload: InferenceRequest) -> InferenceResponse:
        contexts.append((model_name_var.get(), model_version_var.get()))
        return await original_predict(payload)

    mocker.patch.object(sum_model, "predict", side_effect=predict)
    factory = mocker.spy(raw, "DataPlane")
    envelope = message(inference_request.model_dump_json().encode(), sum_model)
    for _ in range(2):
        response = await raw.process_raw_request(worker, envelope)
        assert response.id == envelope.id
        assert response.exception is None
        reply = response.return_value
        assert isinstance(reply, raw.RawReply)
        assert reply.worker_pid == os.getpid()
        assert reply.headers["x-test"] == "echo-me"
        assert {key.lower(): value for key, value in reply.headers.items()}[
            "ce-id"
        ] == ("inference-id")
        value = json.loads(reply.body)
        assert value["id"] == "inference-id"
        assert value["model_name"] == sum_model.name
        assert value["outputs"][0]["data"] == [6.0]
        assert "headers" not in value.get("parameters", {})
    assert contexts == [(sum_model.name, sum_model.version)] * 2
    factory.assert_called_once_with(settings, model_registry)
    plane = getattr(worker, "_rest_raw_data_plane")
    assert (
        plane._ModelInferRequestSuccess.labels(
            model=sum_model.name, version=sum_model.version
        )._value.get()
        == 2
    )


async def test_worker_invalid_body_never_constructs_dataplane(
    settings: Settings,
    model_registry: MultiModelRegistry,
    sum_model: MLModel,
    mocker: MockerFixture,
):
    worker = cast(
        Worker, SimpleNamespace(_settings=settings, _model_registry=model_registry)
    )
    factory = mocker.patch.object(raw, "DataPlane")
    response = await raw.process_raw_request(worker, message(b"{", sum_model))
    assert response.exception is None
    assert isinstance(response.return_value, raw.RawReply)
    assert response.return_value.status == 422
    factory.assert_not_called()


@pytest.mark.parametrize(
    "failure", [RuntimeError("prediction failed"), asyncio.CancelledError()]
)
async def test_worker_wraps_prediction_failures_and_cancellation(
    failure: BaseException,
    settings: Settings,
    model_registry: MultiModelRegistry,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    worker = cast(
        Worker, SimpleNamespace(_settings=settings, _model_registry=model_registry)
    )
    mocker.patch.object(sum_model, "predict", side_effect=failure)
    response = await raw.process_raw_request(
        worker, message(inference_request.model_dump_json().encode(), sum_model)
    )
    assert response.id == "dispatch-id"
    assert response.return_value is None
    assert isinstance(response.exception, WorkerError)
    assert failure.__class__.__name__ in str(response.exception)


@pytest.mark.parametrize("bad_args", [["not-bytes", {}], [b"{}", "not-headers"], []])
async def test_worker_rejects_malformed_internal_envelopes(bad_args: list):
    envelope = ModelRequestMessage(
        model_name="unused", method_name=raw.RAW_INFER_METHOD, method_args=bad_args
    )
    result = await raw.process_raw_request(cast(Worker, SimpleNamespace()), envelope)
    assert result.id == envelope.id
    assert isinstance(result.exception, WorkerError)


async def test_parallel_frontend_forwards_raw_bytes_and_resolved_version(
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    model = parallel(sum_model)
    get_model = mocker.patch.object(
        data_plane._model_registry, "get_model", return_value=model
    )
    local = mocker.patch.object(data_plane, "infer")
    parse = mocker.patch.object(
        raw, "parse_body", side_effect=AssertionError("frontend parse")
    )
    body = inference_request.model_dump_json().encode()
    response = await raw.RawEndpoints(data_plane).infer(
        request(body, {"content-type": "application/json", "x-test": "value"}),
        model.name,
    )
    get_model.assert_awaited_once_with(model.name, None)
    (envelope,) = dispatch_mock(model).call_args.args
    assert envelope.model_name == model.name
    assert envelope.model_version == model.version
    assert envelope.method_name == raw.RAW_INFER_METHOD
    assert envelope.method_args == [
        body,
        {"content-type": "application/json", "x-test": "value"},
    ]
    assert response.body == b'{"id":"from-worker"}'
    assert response.headers["x-reply"] == "yes"
    assert response.headers["content-type"] == "application/json"
    local.assert_not_called()
    parse.assert_not_called()


async def test_nonparallel_model_retains_local_dataplane(
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    inference_request.id = "local-id"
    response = await raw.RawEndpoints(data_plane).infer(
        request(inference_request.model_dump_json().encode(), {"x-test": "local"}),
        sum_model.name,
        sum_model.version,
    )
    value = json.loads(response.body)
    assert value["id"] == "local-id"
    assert value["outputs"][0]["data"] == [6.0]
    assert response.headers["x-test"] == "local"


async def test_non_json_parallel_request_uses_local_validation(
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    model = parallel(sum_model)
    mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
    infer = mocker.patch.object(data_plane, "infer")
    response = await raw.RawEndpoints(data_plane).infer(
        request(
            inference_request.model_dump_json().encode(),
            {"content-type": "text/plain"},
        ),
        model.name,
    )
    assert response.status_code == 422
    dispatch_mock(model).assert_not_called()
    infer.assert_not_called()


@pytest.mark.parametrize("reply", [None, raw.RawReply(b"{}", {}), {"body": b"{}"}])
async def test_frontend_rejects_missing_local_or_wrong_worker_replies(
    reply: Any,
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    model = parallel(sum_model)
    dispatch_mock(model).return_value = ModelResponseMessage(return_value=reply)
    mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
    with pytest.raises(TypeError, match="did not return a worker reply"):
        await raw.RawEndpoints(data_plane).infer(
            request(inference_request.model_dump_json().encode()), model.name
        )


@pytest.mark.parametrize("batch_setting", ["max_batch_size", "max_batch_time"])
async def test_batching_model_retains_frontend_dataplane_entrypoint(
    batch_setting: str,
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    inference_response: InferenceResponse,
    mocker: MockerFixture,
):
    model = parallel(sum_model)
    setattr(model.settings, batch_setting, 1)
    mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
    infer = mocker.patch.object(data_plane, "infer", return_value=inference_response)
    response = await raw.RawEndpoints(data_plane).infer(
        request(inference_request.model_dump_json().encode()), model.name, model.version
    )
    assert response.status_code == 200
    dispatch_mock(model).assert_not_called()
    infer.assert_awaited_once()
    assert infer.call_args.args[1:] == (model.name, model.version)
    assert infer.call_args.args[0].inputs == inference_request.inputs


async def test_frontend_checks_readiness_before_dispatch(
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    model = parallel(sum_model)
    model.ready = False
    mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
    with pytest.raises(ModelNotReady):
        await raw.RawEndpoints(data_plane).infer(
            request(inference_request.model_dump_json().encode()), model.name
        )
    dispatch_mock(model).assert_not_called()


@pytest.mark.parametrize("state", ["unknown", "not-ready"])
@pytest.mark.parametrize("body", [b"", b"{", b"null", b"{}"])
async def test_body_validation_precedes_unknown_or_unready_model_errors(
    state: str,
    body: bytes,
    data_plane: DataPlane,
    sum_model: MLModel,
    mocker: MockerFixture,
):
    model = parallel(sum_model)
    model.ready = False
    lookup = mocker.patch.object(
        data_plane._model_registry, "get_model", return_value=model
    )
    if state == "unknown":
        lookup.side_effect = ModelNotFound(model.name, model.version)
    response = await raw.RawEndpoints(data_plane).infer(
        request(body), model.name, model.version
    )
    expected = raw.parse_body(body)
    assert isinstance(expected, raw.RawReply)
    assert response.status_code == 422
    assert json.loads(response.body) == json.loads(expected.body)
    dispatch_mock(model).assert_not_called()


async def test_valid_body_preserves_original_lookup_exception(
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    original = ModelNotFound(sum_model.name, sum_model.version)
    mocker.patch.object(data_plane._model_registry, "get_model", side_effect=original)
    with pytest.raises(ModelNotFound) as raised:
        await raw.RawEndpoints(data_plane).infer(
            request(inference_request.model_dump_json().encode()),
            sum_model.name,
            sum_model.version,
        )
    assert raised.value is original


async def test_multiple_names_and_versions_use_the_selected_model_pool(
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    first = parallel(sum_model)
    settings = sum_model.settings.model_copy(deep=True)
    settings.name = "other-model"
    assert settings.parameters is not None
    settings.parameters.version = "other-version"
    second = parallel(type(sum_model)(settings))

    async def selected(name: str, version: str | None) -> MLModel:
        return first if name == first.name else second

    get_model = mocker.patch.object(
        data_plane._model_registry, "get_model", side_effect=selected
    )
    endpoints = raw.RawEndpoints(data_plane)
    for model in (first, second, first):
        await endpoints.infer(
            request(inference_request.model_dump_json().encode()),
            model.name,
            model.version,
        )
        envelope = dispatch_mock(model).call_args.args[0]
        assert (envelope.model_name, envelope.model_version) == (
            model.name,
            model.version,
        )
    assert get_model.await_count == 3
    assert dispatch_mock(first).await_count == 2
    assert dispatch_mock(second).await_count == 1


def test_response_encoder_keeps_multiple_outputs_and_serialization_flags(
    inference_response: InferenceResponse,
):
    prediction = inference_response.model_copy(deep=True)
    prediction.id = "response-id"
    prediction.outputs.append(
        prediction.outputs[0].model_copy(update={"name": "second"})
    )
    value = json.loads(raw.encode_response(prediction))
    assert value == prediction.model_dump(
        mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    assert len(value["outputs"]) == 2
    assert value["id"] == "response-id"


def test_parse_body_rejects_invalid_tensor_datatypes(
    inference_request: InferenceRequest,
):
    value = inference_request.model_dump()
    value["inputs"][0]["datatype"] = "not-a-datatype"
    result = raw.parse_body(json.dumps(value).encode())
    assert isinstance(result, raw.RawReply)
    assert result.status == 422
    assert json.loads(result.body)["detail"][0]["loc"] == [
        "body",
        "inputs",
        0,
        "datatype",
    ]


def test_raw_reply_survives_the_normal_worker_pickle_transport():
    original = ModelResponseMessage(
        id="reply-id",
        return_value=raw.RawReply(b'{"detail":[]}', {"x-test": "value"}, status=422),
    )
    restored = ForkingPickler.loads(ForkingPickler.dumps(original))
    assert restored.id == original.id
    assert restored.return_value == original.return_value
    assert isinstance(restored.return_value.body, bytes)
