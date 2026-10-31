"""A spawned child is owned before dispatcher registration can suspend."""

import asyncio

import pytest

from mlserver.parallel.pool import InferencePool
from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.registry import MultiModelRegistry

from .test_lifecycle_recovery import assert_worker_models, wait_until


@pytest.fixture(params=["queue", "zmq"])
def settings(settings, request):
    if request.param == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_inference_transport = request.param
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    settings.worker_start_timeout = 3
    settings.parallel_workers_timeout = 1
    return settings


def assert_retired(worker):
    for queue in (worker._requests, worker._model_updates):
        assert queue._reader.closed
        assert queue._writer.closed
        thread = getattr(queue, "_thread", None)
        assert thread is None or not thread.is_alive()


@pytest.mark.parametrize("exit_mode", ["kill", "stop"])
@pytest.mark.parametrize(
    "observation", ["known-close", "signal-timeout", "signal-release"]
)
async def test_exit_before_registration_remains_owned(
    settings, monkeypatch, exit_mode, observation
):
    pools = InferencePoolRegistry(settings) if observation != "known-close" else None
    pool = pools._default_pool if pools is not None else InferencePool(settings)
    await (pools or pool).start()
    originals = list(pool._workers.values())
    from mlserver.parallel import pool as pool_module

    original_spawn, original_stop = pool_module._spawn_worker, pool.on_worker_stop
    spawned, exits = [], {}
    entered = asyncio.Event()

    def spawn(*args):
        worker = original_spawn(*args)
        spawned.append(worker)
        entered.set()
        return worker

    async def observe_stop(pid, exit_code):
        try:
            await original_stop(pid, exit_code)
        finally:
            exits[pid] = exit_code

    monkeypatch.setattr(pool_module, "_spawn_worker", spawn)
    monkeypatch.setattr(pool_module.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(pool, "on_worker_stop", observe_stop)
    lock = pool._dispatcher._worker_starting_lock
    try:
        # This deliberately forces a defensive interval which ordinary pool
        # serialization does not currently prolong. It is not a production
        # shutdown reproduction or permission to broadcast to unready children.
        await lock.acquire()
        originals[0].kill()
        if pools is None:
            await asyncio.to_thread(originals[0].join, 5)
            await pool.on_worker_stop(originals[0].pid, originals[0].exitcode)
        await asyncio.wait_for(entered.wait(), 5)
        replacement = spawned[0]
        assert replacement.pid not in pool._workers
        assert replacement.pid not in pool._dispatcher._ready_workers
        if exit_mode == "kill":
            replacement.kill()
            expected_exit = -9
        else:
            await replacement.stop()
            expected_exit = 0
        if pools is None:
            await asyncio.to_thread(replacement.join, 5)
            assert replacement.exitcode == expected_exit
            pool._settings.model_operation_timeout = 0.05
            await asyncio.wait_for(pool.close(), 10)
        else:
            # Do not join: the real SIGCHLD handler must preserve waitpid's
            # result before anything can poll the Worker itself.
            await asyncio.wait_for(wait_until(lambda: replacement.pid in exits), 5)
            assert exits[replacement.pid] == expected_exit
            if observation == "signal-release":
                assert replacement.exitcode == expected_exit
                lock.release()
            await asyncio.wait_for(wait_until(lambda: not pool._restart_tasks), 10)
            assert replacement.exitcode == expected_exit
        assert_retired(replacement)
        assert replacement.pid not in pool._dispatcher._ready_workers
        assert not pool._closing_workers
        if observation == "signal-release":
            # Empty replay must not turn the dead child into a ready worker.
            # Its ordinary registered-death path restores replacement capacity.
            assert len(pool._dispatcher._ready_workers) == 1
        await asyncio.wait_for((pools or pool).close(), 10)
        await asyncio.wait_for((pools or pool).close(), 1)
        assert not pool._dispatcher._ready_workers
        assert not pool._restart_tasks
        for worker in originals + spawned:
            assert worker.exitcode is not None
            assert_retired(worker)
    finally:
        if lock.locked():
            lock.release()
        # On an affected baseline, repair only status already observed by this
        # test's reaper observer, after recording the failed assertion. This
        # allows all test-owned resources to be retired without masking failure.
        for worker in originals + spawned:
            if worker.pid in exits:
                worker.record_exit(exits[worker.pid])
        try:
            await asyncio.wait_for((pools or pool).close(), 15)
        finally:
            for worker in originals + spawned:
                if worker.is_alive():
                    worker.kill()
                await asyncio.to_thread(worker.join, 5)
                assert not worker.is_alive()
                await worker.retire_queues(5)


async def test_ordinary_early_exit_restores_committed_model(
    settings, sum_model_settings, monkeypatch, inference_request
):
    pools = InferencePoolRegistry(settings)
    await pools.start()
    pool = pools._default_pool
    registry = MultiModelRegistry(
        on_model_load=[pools.load_model],
        on_model_unload=[pools.unload_model],
        model_operation_timeout=10,
    )
    originals, replacements = list(pool._workers.values()), []
    from mlserver.parallel import pool as pool_module

    original_spawn = pool_module._spawn_worker

    def spawn(*args):
        worker = original_spawn(*args)
        replacements.append(worker)
        if len(replacements) == 1:
            worker.kill()
        return worker

    try:
        current = await registry.load(sum_model_settings)
        monkeypatch.setattr(pool_module, "_spawn_worker", spawn)
        monkeypatch.setattr(pool_module.random, "uniform", lambda *_: 0)
        originals[0].kill()
        # Neither hold registration nor supply an exit status: use the normal
        # reaper and restart path, including a child which exits during startup.
        await asyncio.wait_for(wait_until(lambda: len(replacements) >= 2), 10)
        await asyncio.wait_for(
            wait_until(
                lambda: replacements[-1].pid in pool._dispatcher._ready_workers
                and not pool._restart_tasks
            ),
            10,
        )
        assert len(replacements) == 2
        assert replacements[0].exitcode == -9
        assert_retired(replacements[0])
        await assert_worker_models(pool, current)
        response = await current.predict(inference_request)
        assert response.model_name == current.name
        assert response.model_dump()["outputs"][0]["data"] == [6]
    finally:
        await pools.close()
        for worker in originals + replacements:
            assert not worker.is_alive()
            assert_retired(worker)


@pytest.mark.parametrize("close_instead", [False, True])
async def test_simultaneous_loss_during_staging_keeps_restart_owners(
    settings, sum_model_settings, monkeypatch, close_instead
):
    settings.parallel_workers = 2
    settings.model_operation_timeout = 15
    pool = InferencePool(settings)
    await pool.start()
    originals = list(pool._workers.values())
    entered, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr("mlserver.parallel.pool.random.uniform", lambda *_: 0)

    async def later_hook(model):
        entered.set()
        await release.wait()
        return model

    registry = MultiModelRegistry(
        on_model_load=[pool.load_model, later_hook],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=15,
    )
    attempt = asyncio.create_task(registry.load(sum_model_settings))
    replacements = []
    try:
        await asyncio.wait_for(entered.wait(), 5)
        for worker in originals:
            worker.kill()
            await asyncio.to_thread(worker.join, 5)
            await pool.on_worker_stop(worker.pid, worker.exitcode)
        await asyncio.wait_for(wait_until(lambda: len(pool._restart_tasks) == 2), 5)
        await asyncio.sleep(settings.worker_start_timeout + 0.2)
        assert len(pool._restart_tasks) == 2
        assert not pool._dispatcher._ready_workers
        if close_instead:
            await asyncio.wait_for(pool.close(), 10)
            release.set()
            with pytest.raises(Exception):
                await attempt
            assert not pool._workers
            assert not pool._restart_tasks
        else:
            release.set()
            current = await asyncio.wait_for(attempt, 5)
            await asyncio.wait_for(
                wait_until(lambda: len(pool._dispatcher._ready_workers) == 2), 10
            )
            replacements = list(pool._workers.values())
            assert await registry.get_model(current.name, current.version) is current
            await assert_worker_models(pool, current)
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        await pool.close()
        for worker in originals + replacements:
            assert not worker.is_alive()
            assert_retired(worker)
