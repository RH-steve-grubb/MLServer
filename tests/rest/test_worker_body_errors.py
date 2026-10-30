"""Body-read errors must retain FastAPI's response on every model path."""

from collections.abc import Callable, Generator
from gzip import compress
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from httpx import AsyncClient
import pytest

from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.model import MLModel
from mlserver.parallel.model import ParallelModel
from mlserver.rest.app import create_app

AppFactory = Callable[[bool], FastAPI]


@pytest.fixture(params=[False, True], ids=["nonparallel", "parallel"])
def body_error_app(
    request: pytest.FixtureRequest,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: MLModel,
) -> Generator[AppFactory, None, None]:
    model = ParallelModel(sum_model, Mock()) if request.param else sum_model
    lookup = AsyncMock(return_value=model)

    def factory(enabled: bool) -> FastAPI:
        data_plane._settings.rest_worker_json = enabled
        return create_app(data_plane._settings, data_plane, model_repository_handlers)

    with patch.object(data_plane._model_registry, "get_model", lookup):
        yield factory
    lookup.assert_not_called()


@pytest.mark.parametrize("body", [b"not gzip", compress(b"{}", mtime=0)[:-4]])
@pytest.mark.parametrize(
    "path", ["/v2/models/sum-model/infer", "/v2/models/sum-model/versions/v1.2.3/infer"]
)
async def test_malformed_gzip_preserves_status_and_json_error(
    body_error_app: AppFactory, body: bytes, path: str
):
    for enabled in (False, True):
        app = body_error_app(enabled)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                path,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
            )
        assert response.status_code == 400
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"detail": "There was an error parsing the body"}
