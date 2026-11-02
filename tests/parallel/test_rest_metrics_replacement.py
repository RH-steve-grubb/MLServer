"""Completed inference accounting must outlive a terminated model worker."""

import asyncio
from pathlib import Path

from httpx import AsyncClient
import pytest
from pytest_mock import MockerFixture

from mlserver.parallel.worker import Worker
from mlserver.server import MLServer
from mlserver.settings import Settings, ModelSettings
from mlserver.types import InferenceRequest

from ..utils import RESTClient


@pytest.fixture
def settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.parallel_workers = 1
    settings.rest_worker_json = True
    settings.metrics_dir = str(tmp_path / "metrics")
    settings.metrics_endpoint = "/metrics"
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


def metrics(server: MLServer) -> tuple[float, ...]:
    assert server._metrics_server is not None
    registry = server._metrics_server._endpoint._registry
    labels = {"model": "sum-model", "version": "v1.2.3"}
    return tuple(
        registry.get_sample_value(name, labels) or 0.0
        for name in (
            "model_infer_request_success_total",
            "model_infer_request_failure_total",
            "model_infer_request_duration_count",
            "model_infer_request_duration_sum",
        )
    )


async def test_completed_metrics_survive_worker_termination_and_replacement(
    mlserver: MLServer,
    rest_client: RESTClient,
    settings: Settings,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
):
    await rest_client.wait_until_ready()
    assert mlserver._inference_pool_registry is not None
    pool = mlserver._inference_pool_registry._default_pool
    assert pool is not None
    before = metrics(mlserver)
    async with AsyncClient(
        base_url=f"http://{settings.host}:{settings.http_port}"
    ) as client:
        path = "/v2/models/sum-model/versions/v1.2.3/infer"
        response = await client.post(path, json=inference_request.model_dump())
        assert response.status_code == 200
        assert response.json()["outputs"][0]["data"] == [6]
        completed = metrics(mlserver)
        assert completed[:3] == (before[0] + 1, before[1], before[2] + 1)
        worker = next(iter(pool._workers.values()))
        assert worker.pid is not None
        ready = asyncio.Event()
        original_ready = pool._dispatcher.on_worker_ready

        def on_ready(replacement: Worker) -> None:
            original_ready(replacement)
            ready.set()

        mocker.patch.object(pool._dispatcher, "on_worker_ready", side_effect=on_ready)
        # Workers deliberately ignore SIGTERM; SIGKILL exercises actual death
        # and the registry's automatic replacement and metric-file cleanup.
        worker.kill()
        await asyncio.wait_for(ready.wait(), timeout=15)
        assert metrics(mlserver) == completed
        response = await client.post(path, json=inference_request.model_dump())
        assert response.status_code == 200
        after = metrics(mlserver)
        assert after[:3] == (completed[0] + 1, completed[1], completed[2] + 1)
        assert after[3] >= completed[3]
