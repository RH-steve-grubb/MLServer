"""A dependency wait retains replacement capacity without retrying bad replay."""

import asyncio
from pathlib import Path

import pytest

from mlserver.parallel.messages import ModelUpdateType
from mlserver.parallel.pool import InferencePool
from mlserver.registry import MultiModelRegistry
from tests.fixtures import SumModel

from .test_lifecycle_recovery import (
    assert_worker_models,
    controlled_worker_load,
    wait_until,
)


@pytest.mark.parametrize(
    "workers,dependency,outcome,expire_wait",
    [
        (workers, dependency, "restore", expire)
        for workers in (1, 2)
        for dependency in ("staging", "recovery")
        for expire in (False, True)
    ]
    + [
        (workers, dependency, "close", False)
        for workers in (1, 2)
        for dependency in ("staging", "recovery")
    ],
)
async def test_restart_dependency_retains_capacity_or_joins_close(
    settings,
    sum_model_settings,
    tmp_path,
    monkeypatch,
    workers,
    dependency,
    outcome,
    expire_wait,
):
    settings.parallel_workers = workers
    settings.model_operation_timeout = 15
    settings.worker_start_timeout = 3
    pool = InferencePool(
        settings,
        on_worker_load=[controlled_worker_load] if dependency == "recovery" else [],
    )
    await pool.start()
    original_workers = list(pool._workers.values())
    spawned = []
    entered, release, started, recovery_entered, close_entered = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    original_start, original_close = pool._start_worker, pool._close
    from mlserver.parallel import pool as pool_module

    original_spawn = pool_module._spawn_worker
    monkeypatch.setattr(pool_module.random, "uniform", lambda *_: 0)

    def spawn(*args):
        worker = original_spawn(*args)
        spawned.append(worker)
        return worker

    async def observe_start():
        started.set()
        return await original_start()

    async def observe_close():
        close_entered.set()
        await original_close()

    monkeypatch.setattr(pool_module, "_spawn_worker", spawn)
    monkeypatch.setattr(pool, "_start_worker", observe_start)
    monkeypatch.setattr(pool, "_close", observe_close)

    async def later_hook(model):
        entered.set()
        await release.wait()
        return model

    registry = MultiModelRegistry(
        on_model_load=[pool.load_model]
        + ([later_hook] if dependency == "staging" else []),
        on_model_unload=[pool.unload_model],
        model_operation_timeout=15,
    )
    attempt = close = None
    try:
        incoming = sum_model_settings.model_copy(deep=True)
        incoming.platform = "committed-restart-configuration"
        if dependency == "staging":
            attempt = asyncio.create_task(registry.load(incoming))
            await asyncio.wait_for(entered.wait(), 5)
        else:
            current = await registry.load(incoming)
            original_dispatch = pool._dispatcher.dispatch_update

            async def dispatch(message):
                if message.update_type == ModelUpdateType.Unload and message.rollback:
                    recovery_entered.set()
                    await release.wait()
                return await original_dispatch(message)

            monkeypatch.setattr(pool._dispatcher, "dispatch_update", dispatch)
            abandoned = incoming.model_copy(deep=True)
            abandoned.platform = "abandoned-restart-configuration"
            abandoned.parameters.extra["lifecycle_barrier"] = str(tmp_path)
            pool._settings.model_operation_timeout = 0.5
            attempt = asyncio.create_task(pool.load_model(SumModel(abandoned)))
            await asyncio.wait_for(
                wait_until(lambda: len(list(tmp_path.glob("entered-*"))) == workers), 5
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(attempt), 5)
            pool._settings.model_operation_timeout = 15
        victim = original_workers[0]
        victim.kill()
        await asyncio.to_thread(victim.join, 5)
        await pool.on_worker_stop(victim.pid, victim.exitcode)
        if dependency == "recovery":
            for worker in original_workers[1:]:
                (tmp_path / f"release-{worker.pid}").touch()
            await asyncio.wait_for(recovery_entered.wait(), 5)
        await asyncio.wait_for(started.wait(), 5)
        if outcome == "restore":
            # Entry is synchronized first; only the actual initialization
            # budget is exceeded. No successful spawn/replay is fabricated.
            if expire_wait:
                await asyncio.sleep(settings.worker_start_timeout + 0.2)
            owned_during_wait = bool(pool._restart_tasks)
            assert not spawned
            release.set()
            if dependency == "staging":
                current = await asyncio.wait_for(attempt, 5)
            await asyncio.wait_for(
                wait_until(lambda: len(pool._dispatcher._ready_workers) == workers), 10
            )
            assert owned_during_wait
            assert current.ready
            assert await registry.get_model(current.name, current.version) is current
            await assert_worker_models(pool, current)
        else:
            close = asyncio.create_task(pool.close())
            await asyncio.wait_for(close_entered.wait(), 5)
            if dependency == "recovery":
                release.set()
            await asyncio.wait_for(close, 15)
            release.set()
            if dependency == "staging":
                with pytest.raises(Exception):
                    await asyncio.wait_for(attempt, 5)
            await asyncio.sleep(0)
            assert not spawned
            assert not pool._restart_tasks
            assert not pool._workers
            assert not pool._dispatcher._ready_workers
    finally:
        release.set()
        for worker in original_workers:
            (tmp_path / f"release-{worker.pid}").touch()
        await asyncio.gather(
            *[task for task in (attempt, close) if task is not None],
            return_exceptions=True,
        )
        try:
            await pool.close()
        finally:
            for worker in [*original_workers, *spawned]:
                if worker.is_alive():
                    worker.kill()
                await asyncio.to_thread(worker.join, 5)
                assert not worker.is_alive()
                await worker.retire_queues(5)


async def reject_replay(model):
    marker = model.settings.parameters.extra.get("restart_failure_marker")
    if marker and Path(marker).exists():
        raise ValueError("deliberate replay failure")
    return model


async def pause_replay(model):
    directory = Path(model.settings.parameters.extra["close_replay_barrier"])
    if (directory / "hold").exists():
        (directory / "entered").touch()
        while not (directory / "release").exists():
            await asyncio.sleep(0.01)
    return model


@pytest.mark.parametrize("phase", ["registration", "replay"])
async def test_close_during_native_start_joins_restart_owner(
    settings, sum_model_settings, tmp_path, monkeypatch, phase
):
    settings.parallel_workers = 1
    settings.model_operation_timeout = 15
    settings.worker_start_timeout = 3
    settings.parallel_workers_timeout = 1
    pool = InferencePool(settings, on_worker_load=[pause_replay])
    await pool.start()
    original_workers = list(pool._workers.values())
    from mlserver.parallel import pool as pool_module

    original_spawn = pool_module._spawn_worker
    spawned = []
    spawn_entered, swept = asyncio.Event(), asyncio.Event()
    original_sweep = pool._close_workers

    async def sweep():
        try:
            await original_sweep()
        finally:
            swept.set()

    def spawn(*args):
        worker = original_spawn(*args)
        spawned.append(worker)
        spawn_entered.set()
        return worker

    monkeypatch.setattr(pool_module, "_spawn_worker", spawn)
    monkeypatch.setattr(pool_module.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(pool, "_close_workers", sweep)
    incoming = sum_model_settings.model_copy(deep=True)
    incoming.parameters.extra["close_replay_barrier"] = str(tmp_path)
    try:
        await pool.load_model(SumModel(incoming))
        if phase == "replay":
            (tmp_path / "hold").touch()
        else:
            await pool._dispatcher._worker_starting_lock.acquire()
        victim = original_workers[0]
        victim.kill()
        await asyncio.to_thread(victim.join, 5)
        await pool.on_worker_stop(victim.pid, victim.exitcode)
        await asyncio.wait_for(spawn_entered.wait(), 5)
        if phase == "replay":
            await asyncio.wait_for(
                wait_until(lambda: (tmp_path / "entered").exists()), 5
            )
        pool._settings.model_operation_timeout = 0.05
        close = asyncio.create_task(pool.close())
        if phase == "registration":
            # The child exists before dispatcher registration. Force the
            # first close sweep to finish before registration can continue.
            await asyncio.wait_for(swept.wait(), 5)
            pool._dispatcher._worker_starting_lock.release()
        await asyncio.wait_for(close, 10)
        assert len(spawned) == 1
        assert not pool._restart_tasks
        assert not pool._closing_workers
        assert not pool._dispatcher._ready_workers
        assert all(not worker.is_alive() for worker in [*original_workers, *spawned])
    finally:
        (tmp_path / "release").touch()
        if phase == "registration" and pool._dispatcher._worker_starting_lock.locked():
            pool._dispatcher._worker_starting_lock.release()
        try:
            await pool.close()
        finally:
            for worker in [*original_workers, *spawned]:
                if worker.is_alive():
                    worker.kill()
                await asyncio.to_thread(worker.join, 5)
                await worker.retire_queues(5)


@pytest.mark.parametrize("failure", ["spawn", "replay", "registration-timeout"])
async def test_failed_restart_is_bounded_observable_and_not_retried(
    settings, sum_model_settings, tmp_path, monkeypatch, caplog, failure
):
    settings.parallel_workers = 1
    settings.model_operation_timeout = 15
    settings.worker_start_timeout = 3
    pool = InferencePool(settings, on_worker_load=[reject_replay])
    await pool.start()
    original_workers = list(pool._workers.values())
    marker = tmp_path / "reject-replay"
    incoming = sum_model_settings.model_copy(deep=True)
    incoming.parameters.extra["restart_failure_marker"] = str(marker)
    model = await pool.load_model(SumModel(incoming))
    from mlserver.parallel import pool as pool_module

    original_spawn, original_restart = (
        pool_module._spawn_worker,
        pool._safe_start_worker,
    )
    monkeypatch.setattr(pool_module.random, "uniform", lambda *_: 0)
    finished = asyncio.Event()
    spawned, attempts = [], []

    def spawn(*args):
        attempts.append(None)
        if failure == "spawn":
            raise OSError("deliberate spawn failure")
        worker = original_spawn(*args)
        spawned.append(worker)
        return worker

    async def observe_restart():
        try:
            await original_restart()
        finally:
            finished.set()

    monkeypatch.setattr(pool_module, "_spawn_worker", spawn)
    monkeypatch.setattr(pool, "_safe_start_worker", observe_restart)
    try:
        if failure == "registration-timeout":
            await pool._dispatcher._worker_starting_lock.acquire()
        else:
            marker.touch()
        victim = original_workers[0]
        victim.kill()
        await asyncio.to_thread(victim.join, 5)
        await pool.on_worker_stop(victim.pid, victim.exitcode)
        await asyncio.wait_for(finished.wait(), 10)
        await asyncio.sleep(0)
        assert len(attempts) == 1
        assert "Failed to start replacement worker" in caplog.text
        assert not pool._restart_tasks
        assert not pool._dispatcher._ready_workers
        assert pool.has_model(model.settings)
        for worker in spawned:
            assert worker.exitcode is not None
            assert not worker.is_alive()
            await pool.on_worker_stop(worker.pid, worker.exitcode)
        assert len(attempts) == 1
        assert not pool._restart_tasks
        assert not pool._closing_workers
    finally:
        if (
            failure == "registration-timeout"
            and pool._dispatcher._worker_starting_lock.locked()
        ):
            pool._dispatcher._worker_starting_lock.release()
        await pool.close()
        assert all(not worker.is_alive() for worker in [*original_workers, *spawned])
