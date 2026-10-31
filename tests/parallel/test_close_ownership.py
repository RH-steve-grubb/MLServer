import asyncio

import pytest

from mlserver.parallel.errors import InferencePoolUnavailable


@pytest.mark.parametrize("during_cleanup", [False, True])
async def test_cancelled_pool_close_is_joinable(
    inference_pool, sum_model, monkeypatch, during_cleanup
):
    pool = inference_pool
    entered, release = asyncio.Event(), asyncio.Event()
    if during_cleanup:
        original = pool._close_workers

        async def close_workers():
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(pool, "_close_workers", close_workers)
    else:
        await pool._operation_lock.acquire()
    close = asyncio.create_task(pool.close())
    repeated = None
    try:
        if during_cleanup:
            await asyncio.wait_for(entered.wait(), 5)
        else:
            await asyncio.sleep(0)
        assert pool._closing
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        repeated = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        assert not repeated.done()
        release.set()
        if not during_cleanup:
            pool._operation_lock.release()
        await asyncio.wait_for(repeated, 30)
        assert not pool._workers
        with pytest.raises(InferencePoolUnavailable):
            await pool.load_model(sum_model)
    finally:
        release.set()
        if not during_cleanup and pool._operation_lock.locked():
            pool._operation_lock.release()
        await asyncio.gather(
            close, *([repeated] if repeated else []), return_exceptions=True
        )
        await pool.close()


async def test_cancelled_registry_close_is_joinable(inference_pool_registry):
    registry = inference_pool_registry
    pool = registry._default_pool
    lock = registry._operation_locks.setdefault(None, asyncio.Lock())
    await lock.acquire()
    close = asyncio.create_task(registry.close())
    repeated = None
    try:
        await asyncio.sleep(0)
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        repeated = asyncio.create_task(registry.close())
        lock.release()
        await asyncio.wait_for(repeated, 30)
        assert registry._default_pool is None
        assert not pool._workers
    finally:
        if lock.locked():
            lock.release()
        await asyncio.gather(
            close, *([repeated] if repeated else []), return_exceptions=True
        )
        await registry.close()


async def test_repeated_registry_close_reports_same_cleanup_failure(
    settings, monkeypatch
):
    from mlserver.parallel.registry import InferencePoolRegistry

    settings.parallel_workers = 0
    registry = InferencePoolRegistry(settings)
    await registry.start()
    pool = registry._default_pool
    failure = RuntimeError("cleanup failed after process exit")
    original = pool._close_workers

    async def fail_after_cleanup():
        await original()
        raise failure

    monkeypatch.setattr(pool, "_close_workers", fail_after_cleanup)
    for _ in range(2):
        with pytest.raises(RuntimeError) as observed:
            await asyncio.wait_for(registry.close(), 10)
        assert observed.value is failure
    assert registry._failed_pools[None] is pool
    assert not pool._workers
    assert not pool._closing_workers


async def test_reaped_worker_exit_remains_observable(inference_pool):
    import os

    pool = inference_pool
    worker = next(iter(pool._workers.values()))
    worker.kill()
    pid, status = await asyncio.to_thread(os.waitpid, worker.pid, 0)
    await pool.on_worker_stop(pid, os.waitstatus_to_exitcode(status))
    assert worker.exitcode == -9
    assert not worker.is_alive()
    await pool.close()
    assert not pool._closing_workers


async def test_registry_close_retains_worker_stop_cleanup(
    inference_pool_registry, monkeypatch
):
    registry = inference_pool_registry
    pool = registry._default_pool
    entered, release, pool_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = pool.close

    async def stop_hook(worker):
        entered.set()
        await release.wait()

    async def close_pool():
        await original()
        pool_closed.set()

    pool._on_worker_stop = [stop_hook]
    monkeypatch.setattr(pool, "close", close_pool)
    worker = next(iter(pool._workers.values()))
    worker.kill()
    close = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert registry._worker_stop_tasks
        close = asyncio.create_task(registry.close())
        # Release hooks for workers shut down by close too. The first handler
        # remains deliberately blocked until the close waiter is interrupted.
        await asyncio.sleep(0)
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        assert registry._worker_stop_tasks
    finally:
        release.set()
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await asyncio.wait_for(registry.close(), 15)
    assert pool_closed.is_set()
    assert not registry._worker_stop_tasks
    assert not pool._workers
