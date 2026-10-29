"""Real pending sends must distinguish worker failure from caller cancellation."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import tempfile
from unittest.mock import Mock

import grpc
from httpx import AsyncClient
import pytest
from pytest_mock import MockerFixture

pytest.importorskip("zmq")

from mlserver.grpc import dataplane_pb2_grpc as rpc  # noqa: E402
from mlserver.grpc.converters import ModelInferRequestConverter  # noqa: E402
from mlserver.grpc.servicers import InferenceServicer  # noqa: E402
from mlserver.handlers import DataPlane, ModelRepositoryHandlers  # noqa: E402
from mlserver.model import MLModel  # noqa: E402
from mlserver.parallel.dispatcher import Dispatcher  # noqa: E402
from mlserver.parallel.errors import WorkerStop  # noqa: E402
from mlserver.parallel.messages import ModelRequestMessage  # noqa: E402
from mlserver.parallel.model import ParallelModel  # noqa: E402
from mlserver.parallel.worker import Worker  # noqa: E402
from mlserver.parallel.zmq import create_endpoint  # noqa: E402
from mlserver.rest.app import create_app  # noqa: E402
from mlserver.types import InferenceRequest  # noqa: E402


@dataclass
class PendingSend:
    dispatcher: Dispatcher
    worker: Worker
    plane: DataPlane
    handlers: ModelRepositoryHandlers


@pytest.fixture
async def pending_send(
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
    mocker: MockerFixture,
) -> AsyncGenerator[PendingSend, None]:
    with tempfile.TemporaryDirectory(prefix="mls-stop-") as location:
        directory = Path(location)
        responses = Mock(_zmq_reply_endpoint=create_endpoint(directory, "r"))
        worker = Mock(
            spec=Worker, pid=42, _zmq_request_endpoint=create_endpoint(directory, "q")
        )
        dispatcher = Dispatcher({42: worker}, responses)
        dispatcher.start()
        model = ParallelModel(sum_model, dispatcher)
        model.ready = True
        model.settings.max_batch_size = 0
        model.settings.max_batch_time = 0
        data_plane._settings.rest_worker_json = True
        mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
        try:
            yield PendingSend(dispatcher, worker, data_plane, model_repository_handlers)
        finally:
            await dispatcher.stop()


async def wait_for_pending_send(case: PendingSend) -> None:
    async def connected() -> None:
        transport = case.dispatcher._transport
        assert transport is not None
        while not transport._requests:
            await asyncio.sleep(0)

    # No worker binds its endpoint, so IMMEDIATE keeps the real send pending.
    await asyncio.wait_for(connected(), timeout=2)


async def test_dispatcher_reports_worker_stop_during_pending_send(
    pending_send: PendingSend,
):
    case = pending_send
    task = asyncio.create_task(
        case.dispatcher.dispatch_request(
            ModelRequestMessage(model_name="sum-model", method_name="predict")
        )
    )
    await wait_for_pending_send(case)
    case.dispatcher.on_worker_stop(case.worker, exit_code=23)
    with pytest.raises(WorkerStop, match="23"):
        await asyncio.wait_for(task, timeout=2)
    assert not case.dispatcher._async_responses._futures
    assert case.dispatcher._transport is not None
    assert not case.dispatcher._transport._pending_sends


@pytest.mark.parametrize("order", ["caller-only", "caller-first", "worker-first"])
async def test_genuine_caller_cancellation_wins_even_when_worker_stops(
    pending_send: PendingSend, order: str
):
    case = pending_send
    task = asyncio.create_task(
        case.dispatcher.dispatch_request(
            ModelRequestMessage(model_name="sum-model", method_name="predict")
        )
    )
    await wait_for_pending_send(case)
    if order == "worker-first":
        case.dispatcher.on_worker_stop(case.worker, exit_code=23)
    task.cancel()
    if order == "caller-first":
        case.dispatcher.on_worker_stop(case.worker, exit_code=23)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not case.dispatcher._async_responses._futures
    assert case.dispatcher._transport is not None
    assert not case.dispatcher._transport._pending_sends


async def test_rest_endpoint_returns_json_error_when_pending_send_worker_stops(
    pending_send: PendingSend, inference_request: InferenceRequest
):
    case = pending_send
    app = create_app(case.plane._settings, case.plane, case.handlers)
    async with AsyncClient(app=app, base_url="http://test") as client:
        task = asyncio.create_task(
            client.post(
                "/v2/models/sum-model/infer", json=inference_request.model_dump()
            )
        )
        await wait_for_pending_send(case)
        case.dispatcher.on_worker_stop(case.worker, exit_code=23)
        response = await asyncio.wait_for(task, timeout=2)
        assert response.status_code == 500
        assert response.headers["content-type"] == "application/json"
        assert "exit code 23" in response.json()["error"]


async def test_grpc_endpoint_returns_worker_error_when_pending_send_worker_stops(
    pending_send: PendingSend, inference_request: InferenceRequest
):
    case = pending_send
    server = grpc.aio.server()
    rpc.add_GRPCInferenceServiceServicer_to_server(
        InferenceServicer(case.plane, case.handlers, raw_input_buffers=True), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    values = inference_request.model_dump()
    values["inputs"][0]["datatype"] = "FP32"
    payload = ModelInferRequestConverter.from_types(
        InferenceRequest.model_validate(values), model_name="sum-model", use_raw=True
    )
    assert payload.inputs[0].datatype == "FP32"
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            task = asyncio.ensure_future(
                rpc.GRPCInferenceServiceStub(channel).ModelInfer(payload, timeout=2)
            )
            await wait_for_pending_send(case)
            case.dispatcher.on_worker_stop(case.worker, exit_code=23)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await task
            assert error.value.code() == grpc.StatusCode.INTERNAL
            assert "exit code 23" in error.value.details()
    finally:
        await server.stop(grace=None)
