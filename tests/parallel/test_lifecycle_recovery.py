import asyncio
from copy import deepcopy
from pathlib import Path
import os

import pytest

from mlserver.parallel.errors import NoWorkersAvailable
from mlserver.parallel.pool import InferencePool
from mlserver.registry import MultiModelRegistry


async def test_timeout_quarantines_until_rollback(
    inference_pool, sum_model, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()
    pool = inference_pool
    pool._settings.model_operation_timeout = 0.05

    async def dispatch(message):
        entered.set()
        await release.wait()
        return []

    monkeypatch.setattr(pool._dispatcher, "dispatch_update", dispatch)
    operation = asyncio.create_task(pool.load_model(sum_model))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert not pool.has_model(sum_model.settings)
        with pytest.raises(NoWorkersAvailable):
            pool._dispatcher._get_worker()
        release.set()
        pool._settings.model_operation_timeout = 5
        retry = await asyncio.wait_for(pool.load_model(sum_model), 10)
        assert retry is not None
        assert pool.has_model(sum_model.settings)
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


async def controlled_worker_load(model):
    """Existing trusted SumModel, with file barriers visible across spawn."""
    directory = model.settings.parameters.extra.get("lifecycle_barrier")
    if directory:
        path = Path(directory)
        (path / f"entered-{os.getpid()}").touch()
        while not (path / f"release-{os.getpid()}").exists():
            await asyncio.sleep(0.01)
        if (path / f"fail-{os.getpid()}").exists():
            raise ValueError("late worker failure")
    return model


async def wait_until(predicate):
    while not predicate():
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("outcome", ["success", "failure", "death"])
async def test_multiprocess_partial_load_recovery(
    settings, sum_model_settings, inference_request, tmp_path, outcome
):
    # Worker deadlines are generous. Only the frontend deadline expires, after
    # both worker hooks have entered and one worker has acknowledged its load.
    settings.model_operation_timeout = 10
    pool = InferencePool(settings, on_worker_load=[controlled_worker_load])
    original_workers = list(pool._workers.values())
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=10,
    )
    replacement = deepcopy(sum_model_settings)
    replacement.platform = "abandoned-platform"
    replacement.parameters.extra["lifecycle_barrier"] = str(tmp_path)
    operation = None
    retry = None
    try:
        old = await registry.load(sum_model_settings)
        unrelated_settings = deepcopy(sum_model_settings)
        unrelated_settings.name = "unrelated-model"
        unrelated = await registry.load(unrelated_settings)
        registry._model_operation_timeout = 0.5
        operation = asyncio.create_task(registry.load(replacement))
        await asyncio.wait_for(
            wait_until(lambda: len(list(tmp_path.glob("entered-*"))) == 2), 5
        )
        first, second = original_workers
        (tmp_path / f"release-{first.pid}").touch()
        await asyncio.wait_for(
            wait_until(lambda: len(pool._dispatcher._async_responses._futures) == 1), 5
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert await registry.get_model(old.name, old.version) is old
        assert old.ready
        with pytest.raises(NoWorkersAvailable):
            pool._dispatcher._get_worker()
        with pytest.raises(NoWorkersAvailable):
            await unrelated.predict(inference_request)
        assert list(pool._worker_registry.models) == [old.settings, unrelated.settings]
        assert not pool._pending_reload
        registry._model_operation_timeout = 10
        retry = asyncio.create_task(registry.load(sum_model_settings))
        await asyncio.sleep(0)
        assert not retry.done()
        if outcome == "failure":
            (tmp_path / f"fail-{second.pid}").touch()
        if outcome == "death":
            second.kill()
            await asyncio.to_thread(second.join, 5)
            await pool.on_worker_stop(second.pid, -9)
        else:
            (tmp_path / f"release-{second.pid}").touch()
        current = await asyncio.wait_for(retry, 15)
        assert current.ready
        assert current is not old
        response = await asyncio.wait_for(current.predict(inference_request), 5)
        assert response.model_name == old.name
        assert len(response.outputs) == 1
        unaffected = await unrelated.predict(inference_request)
        assert unaffected.model_name == unrelated.name
        assert await registry.get_model(unrelated.name, unrelated.version) is unrelated
        if outcome == "death":
            await asyncio.wait_for(
                wait_until(lambda: len(pool._dispatcher._ready_workers) == 2), 10
            )
        for _ in range(2):
            metadata = await current._send("metadata")
            assert metadata.platform == sum_model_settings.platform
    finally:
        for worker in original_workers:
            (tmp_path / f"release-{worker.pid}").touch()
        pending = [task for task in (operation, retry) if task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await pool.close()
        assert all(not worker.is_alive() for worker in original_workers)


async def test_later_hook_timeout_cannot_enter_replay(
    inference_pool, sum_model_settings
):
    pool = inference_pool
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=10,
    )
    old = await registry.load(sum_model_settings)
    entered, release = asyncio.Event(), asyncio.Event()

    async def later_hook(model):
        entered.set()
        await release.wait()
        return model

    registry._models[old.name]._on_model_load = [pool.load_model, later_hook]
    registry._model_operation_timeout = 0.1
    operation = asyncio.create_task(registry.load(sum_model_settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert list(pool._worker_registry.models) == [old.settings]
        assert not pool._pending_reload
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert await registry.get_model(old.name, old.version) is old
        assert not pool._staged
        assert not pool._pending_reload
        await asyncio.wait_for(pool._settle_recovery(), 5)
        assert len(pool._dispatcher._ready_workers) == 2
        await registry.unload(old.name)
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


async def test_environment_extraction_retains_ownership(
    inference_pool_registry, monkeypatch
):
    from mlserver.env import Environment

    registry = inference_pool_registry
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def extract(tarball, env_path, env_hash):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return Environment(env_path, env_hash)

    monkeypatch.setattr(Environment, "from_tarball", extract)
    first = asyncio.create_task(registry._extract_tarball("owned", "unused"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(registry._extract_tarball("owned", "unused"))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        env = await asyncio.wait_for(second, 5)
        assert env.env_hash == "owned"
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


async def test_incomplete_extraction_is_not_reused_after_owner_exit(
    inference_pool_registry,
):
    registry = inference_pool_registry
    directory = Path(registry._get_env_path("incomplete"))
    directory.mkdir()
    (directory / ".mlserver-extraction-incomplete").touch()
    with pytest.raises(RuntimeError, match="incomplete extraction"):
        await registry._extract_tarball("incomplete", "unused")
