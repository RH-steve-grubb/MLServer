"""Completed inference accounting must outlive a terminated model worker."""

import asyncio
from pathlib import Path

from httpx import AsyncClient
import pytest

from mlserver.parallel.pool import InferencePool
from mlserver.server import MLServer
from mlserver.settings import Settings, ModelSettings
from mlserver.types import InferenceRequest

from ..utils import RESTClient


@pytest.fixture(params=["queue", "zmq"])
def settings(
    settings: Settings, tmp_path: Path, request: pytest.FixtureRequest
) -> Settings:
    if request.param == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_workers = 1
    settings.parallel_inference_transport = request.param
    settings.rest_worker_json = True
    settings.metrics_dir = str(tmp_path / "metrics")
    settings.metrics_endpoint = "/metrics"
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


async def replacement_ready(pool: InferencePool, stopped: int) -> None:
    while True:
        ready = pool._dispatcher._ready_workers
        if len(ready) == 1 and stopped not in ready:
            return
        await asyncio.sleep(0.01)


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
):
    await rest_client.wait_until_ready()
    assert mlserver._inference_pool_registry is not None
    pool = mlserver._inference_pool_registry._default_pool
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
        pid = worker.pid
        # Workers deliberately ignore SIGTERM; SIGKILL exercises actual death
        # and the registry's automatic replacement and metric-file cleanup.
        worker.kill()
        await asyncio.wait_for(replacement_ready(pool, pid), timeout=15)
        assert metrics(mlserver) == completed
        response = await client.post(path, json=inference_request.model_dump())
        assert response.status_code == 200
        after = metrics(mlserver)
        assert after[:3] == (completed[0] + 1, completed[1], completed[2] + 1)
        assert after[3] >= completed[3]
