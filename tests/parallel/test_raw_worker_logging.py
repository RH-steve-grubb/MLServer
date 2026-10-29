"""Raw REST errors need the same worker traceback as ordinary inference."""

from functools import partial
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from prometheus_client import CollectorRegistry
import pytest

from mlserver.parallel.errors import WorkerError
from mlserver.parallel.messages import ModelRequestMessage
from mlserver.parallel.worker import Worker
from mlserver.registry import MultiModelRegistry
from mlserver.rest.worker import RAW_INFER_METHOD, RawReply
from mlserver.settings import Settings, ModelSettings

from ..fixtures import ErrorModel


@pytest.fixture
async def error_worker(
    settings: Settings,
    model_registry: MultiModelRegistry,
    error_model_settings: ModelSettings,
    prometheus_registry: CollectorRegistry,
) -> Worker:
    settings.rest_worker_json = True
    await model_registry.load(error_model_settings)
    worker = cast(
        Worker, SimpleNamespace(_settings=settings, _model_registry=model_registry)
    )
    setattr(worker, "_process_request", partial(Worker._process_request, worker))
    return worker


def message(body: bytes) -> ModelRequestMessage:
    return ModelRequestMessage(
        model_name="error-model",
        method_name=RAW_INFER_METHOD,
        method_args=[body, {"content-type": "application/json"}],
    )


@pytest.mark.parametrize("transport", ["queue", "zmq"])
async def test_raw_model_error_logs_one_worker_traceback(
    transport: str, error_worker: Worker, caplog: pytest.LogCaptureFixture
):
    envelope = message(b'{"inputs":[]}')
    caplog.clear()
    if transport == "queue":
        response = await error_worker._process_request(envelope)
    else:
        pytest.importorskip("zmq")
        from mlserver.parallel.zmq import _reply

        socket = SimpleNamespace(send=AsyncMock())
        await _reply(error_worker, envelope, socket)
        response = ForkingPickler.loads(socket.send.call_args.args[0])
    assert isinstance(response.exception, WorkerError)
    assert response.exception.status_code == 400
    errors = [
        record
        for record in caplog.records
        if record.name == "mlserver.parallel" and record.exc_info
    ]
    assert len(errors) == 1
    assert "error-model" in errors[0].getMessage()
    assert errors[0].exc_info is not None
    assert str(errors[0].exc_info[1]) == ErrorModel.error_message
    assert errors[0].exc_info[2] is not None


async def test_schema_validation_reply_does_not_log_worker_traceback(
    error_worker: Worker, caplog: pytest.LogCaptureFixture
):
    caplog.clear()
    response = await error_worker._process_request(message(b"{}"))
    assert response.exception is None
    assert isinstance(response.return_value, RawReply)
    assert response.return_value.status == 422
    assert not any(record.exc_info for record in caplog.records)
