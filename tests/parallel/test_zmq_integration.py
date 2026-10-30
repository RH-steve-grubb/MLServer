"""Normal spawned-worker integration for the optional ZeroMQ data plane."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest
from pytest_mock import MockerFixture

pytest.importorskip("zmq")

from mlserver.codecs import NumpyCodec  # noqa: E402
from mlserver.grpc import dataplane_pb2 as pb  # noqa: E402
from mlserver.grpc.buffers import convert_request  # noqa: E402
from mlserver.model import MLModel  # noqa: E402
from mlserver.parallel.errors import WorkerError  # noqa: E402
from mlserver.parallel.messages import (  # noqa: E402
    ModelRequestMessage,
    ModelUpdateType,
)
from mlserver.parallel.model import ParallelModel  # noqa: E402
from mlserver.parallel.pool import InferencePool  # noqa: E402
from mlserver.parallel.worker import Worker  # noqa: E402
from mlserver.rest.worker import RAW_INFER_METHOD, RawReply  # noqa: E402
from mlserver.settings import ModelSettings, Settings  # noqa: E402
from mlserver.types import InferenceRequest  # noqa: E402

from ..fixtures import ErrorModel  # noqa: E402


@dataclass
class RunningPool:
    pool: InferencePool
    model: ParallelModel
    directory: Path
    initial_workers: list[Worker]


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.parallel_inference_transport = "zmq"
    settings.rest_worker_json = True
    settings.grpc_raw_input_buffers = True
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


@pytest.fixture
async def running_pool(
    settings: Settings, sum_model: MLModel
) -> AsyncGenerator[RunningPool, None]:
    pool = InferencePool(settings)
    assert pool._transport_directory is not None
    directory = Path(pool._transport_directory.name)
    initial_workers = list(pool._workers.values())
    try:
        model = await asyncio.wait_for(pool.load_model(sum_model), timeout=10)
        assert isinstance(model, ParallelModel)
        yield RunningPool(pool, model, directory, initial_workers)
    finally:
        workers = {*initial_workers, *pool._workers.values()}
        await asyncio.wait_for(pool.close(), timeout=10)
        assert not directory.exists()
        assert not pool._workers
        assert all(not worker.is_alive() for worker in workers)


def fp32_request(value: int) -> InferenceRequest:
    tensor = np.array([[value, 0.25, 0.75]], dtype="<f4")
    wire = pb.ModelInferRequest(
        model_name="sum-model",
        id=f"numeric-{value}",
        inputs=[
            pb.ModelInferRequest.InferInputTensor(
                name="input-0", datatype="FP32", shape=[1, 3]
            )
        ],
        outputs=[pb.ModelInferRequest.InferRequestedOutputTensor(name="total")],
        raw_input_contents=[tensor.tobytes()],
    )
    converted = convert_request(wire)
    data = converted.inputs[0].data.root
    assert isinstance(data, np.ndarray)
    assert not data.flags.writeable
    np.testing.assert_array_equal(data, tensor.reshape(-1))
    return converted


async def raw_predict(case: RunningPool, payload: InferenceRequest) -> RawReply:
    envelope = ModelRequestMessage(
        model_name=case.model.name,
        model_version=case.model.version,
        method_name=RAW_INFER_METHOD,
        method_args=[
            payload.model_dump_json().encode(),
            {"content-type": "application/json", "x-integration": "echo"},
        ],
    )
    response = await asyncio.wait_for(
        case.pool._dispatcher.dispatch_request(envelope), timeout=10
    )
    assert response.id == envelope.id
    assert response.exception is None
    assert isinstance(response.return_value, RawReply)
    reply = response.return_value
    assert isinstance(reply.body, bytes)
    assert reply.status == 200
    assert reply.worker_pid != os.getpid()
    assert reply.worker_pid in case.pool._workers
    assert reply.headers["x-integration"] == "echo"
    prediction = json.loads(reply.body)
    assert prediction["id"] == payload.id
    assert prediction["model_name"] == case.model.name
    assert prediction["outputs"][0]["data"] == [6]
    return reply


async def test_spawned_numeric_buffers_and_many_inflight_requests(
    running_pool: RunningPool,
) -> None:
    case = running_pool
    assert stat.S_IMODE(case.directory.stat().st_mode) == 0o700
    assert case.pool._dispatcher._transport is not None
    assert all(
        worker.is_alive() and worker.pid != os.getpid()
        for worker in case.initial_workers
    )
    payloads = [fp32_request(value) for value in range(12)]
    responses = await asyncio.wait_for(
        asyncio.gather(*(case.model.predict(payload) for payload in payloads)),
        timeout=10,
    )
    assert [response.id for response in responses] == [
        payload.id for payload in payloads
    ]
    for value, response in enumerate(responses):
        assert response.model_name == case.model.name
        assert response.model_version == case.model.version
        np.testing.assert_array_equal(
            NumpyCodec.decode_output(response.outputs[0]), [[value + 1]]
        )
    # Worker-side NumPy decoding must not mutate the parent's compact buffers.
    assert all(
        isinstance(payload.inputs[0].data.root, np.ndarray) for payload in payloads
    )
    assert not case.pool._dispatcher._async_responses._futures


async def test_raw_rest_bytes_are_processed_in_both_spawned_workers(
    running_pool: RunningPool, inference_request: InferenceRequest
) -> None:
    payloads = [inference_request.model_copy(deep=True) for _ in range(8)]
    for index, payload in enumerate(payloads):
        payload.id = f"raw-{index}"
    replies = await asyncio.wait_for(
        asyncio.gather(*(raw_predict(running_pool, payload) for payload in payloads)),
        timeout=10,
    )
    assert {reply.worker_pid for reply in replies} == set(running_pool.pool._workers)


async def test_model_unload_uses_queue_control_and_removes_worker_model(
    running_pool: RunningPool, mocker: MockerFixture
) -> None:
    case = running_pool
    controls = [mocker.spy(worker, "send_update") for worker in case.initial_workers]
    send = mocker.spy(case.pool._dispatcher._transport, "send")
    await asyncio.wait_for(case.pool.unload_model(case.model), timeout=10)
    assert not case.pool.has_model(case.model.settings)
    assert case.pool.empty()
    for control in controls:
        control.assert_called_once()
        assert control.call_args.args[0].update_type == ModelUpdateType.Unload
    send.assert_not_called()
    # The worker acknowledges Queue control, then rejects data-plane traffic
    # for the unloaded model through the independently active ZeroMQ channel.
    with pytest.raises(WorkerError, match="not found"):
        await asyncio.wait_for(case.model.predict(fp32_request(1)), timeout=10)
    assert not case.pool._dispatcher._async_responses._futures


async def test_new_worker_replays_models_while_existing_workers_serve(
    running_pool: RunningPool, inference_request: InferenceRequest
) -> None:
    case = running_pool

    async def start_with_traffic() -> Worker:
        starting = asyncio.create_task(case.pool._start_worker())
        try:
            for value in range(6):
                response = await case.model.predict(fp32_request(value))
                np.testing.assert_array_equal(
                    NumpyCodec.decode_output(response.outputs[0]), [[value + 1]]
                )
            worker = await starting
            assert worker is not None
            return worker
        finally:
            if not starting.done():
                starting.cancel()
                await asyncio.gather(starting, return_exceptions=True)

    new_worker = await asyncio.wait_for(start_with_traffic(), timeout=10)
    assert len(case.pool._workers) == 3
    assert new_worker.pid not in {worker.pid for worker in case.initial_workers}
    replies = []
    for number in range(6):
        payload = inference_request.model_copy(update={"id": f"replayed-{number}"})
        replies.append(await raw_predict(case, payload))
    assert {reply.worker_pid for reply in replies} == set(case.pool._workers)


async def test_worker_error_keeps_other_model_and_control_channel_available(
    running_pool: RunningPool, error_model_settings: ModelSettings
) -> None:
    case = running_pool
    failing_source = ErrorModel(error_model_settings)
    failing = await asyncio.wait_for(case.pool.load_model(failing_source), timeout=10)
    with pytest.raises(WorkerError, match=ErrorModel.error_message):
        await asyncio.wait_for(failing.predict(fp32_request(0)), timeout=10)
    response = await asyncio.wait_for(case.model.predict(fp32_request(2)), timeout=10)
    np.testing.assert_array_equal(NumpyCodec.decode_output(response.outputs[0]), [[3]])
    await asyncio.wait_for(case.pool.unload_model(failing_source), timeout=10)
    assert not case.pool.has_model(failing_source.settings)
    assert case.pool.has_model(case.model.settings)
    assert not case.pool._dispatcher._async_responses._futures
