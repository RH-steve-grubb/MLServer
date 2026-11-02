"""Inference accounting belongs to the frontend, even with worker JSON."""

import os
from multiprocessing.reduction import ForkingPickler
from unittest.mock import AsyncMock, Mock

import pytest
from pytest_mock import MockerFixture

from mlserver.errors import ModelNotFound, ModelNotReady
from mlserver.handlers import DataPlane
from mlserver.model import MLModel
from mlserver.parallel.messages import ModelResponseMessage
from mlserver.parallel.errors import WorkerError
from mlserver.parallel.model import ParallelModel
from mlserver.rest.worker import RawEndpoints, RawReply
from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest

from .test_worker import request


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.parallel_workers = 0
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


@pytest.fixture
def selected_model(
    data_plane: DataPlane, sum_model: MLModel, mocker: MockerFixture
) -> ParallelModel:
    dispatcher = Mock()
    dispatcher.dispatch_request = AsyncMock(
        return_value=ModelResponseMessage(
            return_value=RawReply(b"{}", {}, worker_pid=os.getpid() + 1)
        )
    )
    model = ParallelModel(sum_model, dispatcher)
    model.ready = True
    mocker.patch.object(data_plane._model_registry, "get_model", return_value=model)
    return model


def snapshot(plane: DataPlane, name: str, version: str | None) -> tuple[float, ...]:
    labels = {"model": name, "version": version}
    duration = plane._ModelInferRequestDuration.labels(**labels)
    return (
        plane._ModelInferRequestSuccess.labels(**labels)._value.get(),
        plane._ModelInferRequestFailure.labels(**labels)._value.get(),
        duration._count.get(),
        duration._sum.get(),
    )


@pytest.mark.parametrize("version", [None, "v1.2.3"])
async def test_frontend_records_success_with_requested_version_label(
    version: str | None,
    selected_model: ParallelModel,
    data_plane: DataPlane,
    inference_request: InferenceRequest,
):
    before = snapshot(data_plane, selected_model.name, version)
    resolved_before = snapshot(data_plane, selected_model.name, selected_model.version)
    response = await RawEndpoints(data_plane).infer(
        request(inference_request.model_dump_json().encode()),
        selected_model.name,
        version,
    )
    after = snapshot(data_plane, selected_model.name, version)
    assert response.status_code == 200
    assert after[:3] == (before[0] + 1, before[1], before[2] + 1)
    if version is None:
        assert snapshot(data_plane, selected_model.name, selected_model.version) == (
            resolved_before
        )


@pytest.mark.parametrize("state", ["unknown", "not-ready"])
async def test_frontend_counts_valid_requests_rejected_before_dispatch(
    state: str,
    selected_model: ParallelModel,
    data_plane: DataPlane,
    inference_request: InferenceRequest,
):
    error = (
        ModelNotFound(selected_model.name)
        if state == "unknown"
        else ModelNotReady(selected_model.name)
    )
    lookup = data_plane._model_registry.get_model
    assert isinstance(lookup, AsyncMock)
    lookup.side_effect = error
    before = snapshot(data_plane, selected_model.name, None)
    with pytest.raises(type(error)):
        await RawEndpoints(data_plane).infer(
            request(inference_request.model_dump_json().encode()), selected_model.name
        )
    after = snapshot(data_plane, selected_model.name, None)
    assert after[:3] == (before[0], before[1] + 1, before[2] + 1)
    send = selected_model._dispatcher.dispatch_request
    assert isinstance(send, AsyncMock)
    send.assert_not_called()


@pytest.mark.parametrize("state", ["valid-model", "unknown"])
async def test_body_validation_does_not_count_as_an_inference_failure(
    state: str, selected_model: ParallelModel, data_plane: DataPlane
):
    if state == "unknown":
        lookup = data_plane._model_registry.get_model
        assert isinstance(lookup, AsyncMock)
        lookup.side_effect = ModelNotFound(selected_model.name)
    else:
        send = selected_model._dispatcher.dispatch_request
        assert isinstance(send, AsyncMock)
        send.return_value = ModelResponseMessage(
            return_value=RawReply(b'{"detail":[]}', {}, status=422, worker_pid=-1)
        )
    before = snapshot(data_plane, selected_model.name, None)
    response = await RawEndpoints(data_plane).infer(request(b"{}"), selected_model.name)
    assert response.status_code == 422
    assert snapshot(data_plane, selected_model.name, None) == before


@pytest.mark.parametrize("phase", ["success", "predict-error", "encode-error"])
async def test_duration_includes_dispatch_and_excludes_worker_json_conversion(
    phase: str,
    data_plane: DataPlane,
    selected_model: ParallelModel,
    mocker: MockerFixture,
):
    clock = [0.0]
    mocker.patch(
        "mlserver.handlers.dataplane.perf_counter", side_effect=lambda: clock[0]
    )
    mocker.patch("mlserver.rest.worker.perf_counter", return_value=0.0)
    before = snapshot(data_plane, selected_model.name, None)

    async def dispatch(message) -> ModelResponseMessage:
        clock[0] = 0.25
        if phase != "success":
            error = WorkerError(RuntimeError(phase))
            setattr(error, "_rest_conversion_duration", 0.05)
            setattr(error, "_rest_inference_succeeded", phase == "encode-error")
            raise ForkingPickler.loads(ForkingPickler.dumps(error))
        return ModelResponseMessage(
            return_value=RawReply(
                b"{}", {}, worker_pid=os.getpid() + 1, conversion_duration=0.05
            )
        )

    mocker.patch.object(
        selected_model._dispatcher, "dispatch_request", side_effect=dispatch
    )
    if phase != "success":
        with pytest.raises(WorkerError):
            await RawEndpoints(data_plane).infer(
                request(b'{"inputs":[]}'), selected_model.name
            )
    else:
        await RawEndpoints(data_plane).infer(
            request(b'{"inputs":[]}'), selected_model.name
        )
    after = snapshot(data_plane, selected_model.name, None)
    assert after[:3] == (
        before[0] + int(phase != "predict-error"),
        before[1] + int(phase == "predict-error"),
        before[2] + 1,
    )
    assert after[3] - before[3] == pytest.approx(0.20)
