import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mlserver.parallel.model import ParallelModel
from mlserver.registry import MultiModelRegistry
from mlserver.rest.server import RESTServer
from mlserver.rest.utils import to_scope

from ..fixtures import SumModel


def test_add_custom_handlers(rest_server: RESTServer, sum_model: SumModel):
    scope = to_scope(sum_model.my_payload.__custom_handler__)
    found_route = None
    for route in rest_server._app.routes:
        match, _ = route.matches(scope)
        if match == match.FULL:
            found_route = route
            break

    assert found_route is not None


async def test_delete_custom_handlers(rest_server: RESTServer, sum_model: SumModel):
    await rest_server.delete_custom_handlers(sum_model)

    scope = to_scope(sum_model.my_payload.__custom_handler__)
    for route in rest_server._app.routes:
        match, _ = route.matches(scope)
        assert match == match.NONE


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_abandoned_load_removes_custom_routes_before_native_cleanup(
    rest_server, rest_client, sum_model, monkeypatch, replacement, wrapped
):
    await rest_server.delete_custom_handlers(sum_model)
    registry = MultiModelRegistry(
        on_model_load=[rest_server.add_custom_handlers],
        on_model_unload=[rest_server.delete_custom_handlers],
    )
    settings = sum_model.settings.model_copy(deep=True)
    old = await registry.load(settings) if replacement else None
    partial = SumModel(settings.model_copy(deep=True))
    if wrapped:
        partial = ParallelModel(partial, Mock())
        monkeypatch.setattr(partial, "_send", AsyncMock(return_value=-999))
    abandoned = Mock(return_value=None)
    partial._on_load_failed = abandoned
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def load():
        entered.set()
        await release.wait()
        return True

    async def unload():
        cleaned.set()
        return True

    monkeypatch.setattr(partial, "load", load)
    monkeypatch.setattr(partial, "unload", unload)
    registry._model_initialiser = lambda _: partial
    if old is not None:
        registry._models[old.name]._model_initialiser = registry._model_initialiser
    registry._model_operation_timeout = 0.05
    attempt = asyncio.create_task(registry.load(settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        abandoned.assert_called_once()
        assert not cleaned.is_set()
        response = await rest_client.post("/my-custom-endpoint", json=[1, 2, 3])
        assert response.status_code == (200 if replacement else 404)
        if replacement:
            assert response.json() == 6
            assert await registry.get_model(old.name, old.version) is old
        assert not any(
            getattr(route, "endpoint", None) == partial.my_payload
            for route in rest_server._app.routes
        )
        release.set()
        await asyncio.wait_for(cleaned.wait(), 5)
        if replacement:
            response = await rest_client.post("/my-custom-endpoint", json=[1, 2, 3])
            assert response.status_code == 200
            assert response.json() == 6
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        registry._model_operation_timeout = 5
        await registry.unload(settings.name)


async def test_failed_custom_route_registration_removes_partial_routes(
    rest_server, rest_client, sum_model, monkeypatch
):
    await rest_server.delete_custom_handlers(sum_model)
    partial = SumModel(sum_model.settings.model_copy(deep=True))
    original = rest_server._app.add_api_route
    calls = 0

    def add_route(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("invalid later route")
        return original(*args, **kwargs)

    monkeypatch.setattr(rest_server._app, "add_api_route", add_route)
    try:
        with pytest.raises(ValueError, match="invalid later route"):
            await rest_server.add_custom_handlers(partial)
        response = await rest_client.post("/custom-endpoint-with-long-response", json=3)
        assert response.status_code == 404
    finally:
        await rest_server.delete_custom_handlers(partial)
