"""Worker JSON must preserve this release's shared frontend response cache."""

import pytest
from httpx import AsyncClient
from pytest_mock import MockerFixture

from mlserver import ModelSettings, Settings
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.model import MLModel
from mlserver.parallel.model import ParallelModel
from mlserver.rest.app import create_app
from mlserver.types import InferenceRequest


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.parallel_workers = 2
    settings.cache_enabled = True
    return settings


@pytest.fixture
def sum_model_settings(sum_model_settings: ModelSettings) -> ModelSettings:
    sum_model_settings.cache_enabled = True
    sum_model_settings.max_batch_size = 0
    sum_model_settings.max_batch_time = 0
    return sum_model_settings


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("versioned", [False, True])
async def test_repeated_requests_share_the_frontend_cache(
    enabled: bool,
    versioned: bool,
    settings: Settings,
    data_plane: DataPlane,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    mocker: MockerFixture,
    model_repository_handlers: ModelRepositoryHandlers,
):
    assert isinstance(sum_model, ParallelModel)
    settings.rest_worker_json = enabled
    app = create_app(settings, data_plane, model_repository_handlers)
    predict = mocker.spy(sum_model, "predict")
    inference_request.id = "cached-request"
    path = f"/v2/models/{sum_model.name}"
    if versioned:
        path += f"/versions/{sum_model.version}"
    async with AsyncClient(app=app, base_url="http://test") as client:
        first = await client.post(f"{path}/infer", json=inference_request.model_dump())
        second = await client.post(f"{path}/infer", json=inference_request.model_dump())
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["outputs"][0]["data"] == [6]
    # A second dispatch would select the other worker. A shared cache hit must
    # avoid prediction altogether, including when worker JSON is enabled.
    assert predict.await_count == 1
    cache = data_plane._get_response_cache()
    assert cache is not None
    assert await cache.size() == 1
