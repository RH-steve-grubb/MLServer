"""Automatic replacement owns and rolls back partially acquired candidates."""

import asyncio

import pytest

from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.parallel.worker import Worker
from tests.fixtures import SumModel

from .test_lifecycle_recovery import wait_until


@pytest.fixture(params=["queue", "zmq"])
def settings(settings, request):
    if request.param == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_inference_transport = request.param
    settings.parallel_workers = 2
    settings.model_operation_timeout = 10
    settings.worker_start_timeout = 3
    settings.parallel_workers_timeout = 1
    return settings


def fail_acquisition(monkeypatch, boundary):
    from mlserver.parallel import worker as worker_module

    candidates, queues = [], []
    primary = OSError("replacement acquisition failed")
    original_start, original_queue = Worker.start, worker_module.Queue

    def start(worker):
        candidates.append(worker)
        if boundary == "pre-start":
            raise primary
        if boundary == "process-start":
            worker._prepare_queues()
            raise primary
        original_start(worker)
        if boundary == "post-start":
            raise primary

    def queue():
        if boundary == "first-queue" or queues:
            raise primary
        value = original_queue()
        queues.append(value)
        return value

    monkeypatch.setattr(Worker, "start", start)
    if boundary in ("first-queue", "second-queue"):
        monkeypatch.setattr(worker_module, "Queue", queue)
    return candidates, primary


@pytest.mark.parametrize(
    "boundary,cleanup_failure",
    [
        (boundary, False)
        for boundary in ("first-queue", "second-queue", "pre-start", "post-start")
    ]
    + [("post-start", True)],
)
async def test_automatic_failed_acquisition_retires_only_candidate(
    settings, sum_model_settings, monkeypatch, caplog, boundary, cleanup_failure
):
    registry = InferencePoolRegistry(settings)
    candidates = []
    try:
        model = await registry.load_model(SumModel(sum_model_settings))
        ready_before = model.ready
        pool = registry._default_pool
        victim, survivor = list(pool._workers.values())
        candidates, primary = fail_acquisition(monkeypatch, boundary)
        original_finish, original_start = pool._finish_restart, pool._start_worker
        original_retire = Worker.retire_queues
        finished = asyncio.Event()
        errors = []
        cleanup = OSError("replacement queues remain owned")

        async def start():
            try:
                return await original_start()
            except BaseException as error:
                errors.append(error)
                raise

        async def finish():
            try:
                return await original_finish()
            finally:
                finished.set()

        async def retire(worker, timeout):
            if worker in candidates:
                raise cleanup
            await original_retire(worker, timeout)

        monkeypatch.setattr(pool, "_start_worker", start)
        monkeypatch.setattr(pool, "_finish_restart", finish)
        monkeypatch.setattr("mlserver.parallel.pool.random.uniform", lambda *_: 0)
        if cleanup_failure:
            monkeypatch.setattr(Worker, "retire_queues", retire)
        victim.kill()
        await asyncio.wait_for(finished.wait(), 10)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert errors == [primary]
        assert "Failed to start replacement worker" in caplog.text
        assert candidate.pid is None or candidate.exitcode is not None
        assert survivor.exitcode is None
        assert pool._dispatcher._ready_workers == {survivor.pid: survivor}
        assert (await model._send("metadata")).name == model.name
        # This hook does not own frontend readiness publication. Preserve its
        # existing value while the surviving worker still serves the model.
        assert pool.has_model(model.settings) and model.ready == ready_before
        if cleanup_failure:
            assert pool._closing_workers[id(candidate)] is candidate
            assert not candidate.queues_retired
            assert getattr(primary, "cleanup_error") is cleanup
            assert pool._cleanup_error is cleanup
        else:
            assert candidate.queues_retired
            assert all(q._reader.closed and q._writer.closed for q in candidate._queues)
            assert candidate not in pool._unstarted_workers
            assert candidate not in pool._closing_workers.values()
        await asyncio.sleep(0)
        assert len(candidates) == 1 and not pool._restart_tasks
    finally:
        monkeypatch.undo()
        await registry.close()
        for candidate in candidates:
            assert candidate.pid is None or candidate.exitcode is not None
            assert candidate.queues_retired


@pytest.mark.parametrize("interruption", ["cancel-restart", "cancel-close-waiter"])
async def test_accepted_replacement_rollback_survives_waiter_cancellation(
    settings, sum_model_settings, monkeypatch, interruption
):
    registry = InferencePoolRegistry(settings)
    candidates = []
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    close_waiter = None
    try:
        model = await registry.load_model(SumModel(sum_model_settings))
        pool = registry._default_pool
        victim = next(iter(pool._workers.values()))
        candidates, primary = fail_acquisition(monkeypatch, "post-start")
        original_retire, original_finish = Worker.retire_queues, pool._finish_restart

        async def retire(worker, timeout):
            if worker in candidates:
                entered.set()
                await release.wait()
            await original_retire(worker, timeout)

        async def finish():
            try:
                return await original_finish()
            finally:
                finished.set()

        monkeypatch.setattr(Worker, "retire_queues", retire)
        monkeypatch.setattr(pool, "_finish_restart", finish)
        monkeypatch.setattr("mlserver.parallel.pool.random.uniform", lambda *_: 0)
        victim.kill()
        await asyncio.wait_for(entered.wait(), 5)
        candidate = candidates[0]
        assert candidate.exitcode is not None and not candidate.queues_retired
        assert pool._closing_workers[id(candidate)] is candidate
        if interruption == "cancel-restart":
            assert next(iter(pool._restart_tasks)).cancel()
        else:
            close_waiter = asyncio.create_task(registry.close())
            await asyncio.wait_for(wait_until(lambda: pool._closing), 5)
            close_waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close_waiter
        release.set()
        await asyncio.wait_for(finished.wait(), 5)
        if interruption == "cancel-close-waiter":
            await asyncio.wait_for(registry.close(), 10)
            assert not pool._dispatcher._ready_workers
        else:
            assert (await model._send("metadata")).name == model.name
        assert len(candidates) == 1 and candidate.queues_retired
        assert not pool._unstarted_workers and not pool._closing_workers
    finally:
        release.set()
        if close_waiter is not None:
            await asyncio.gather(close_waiter, return_exceptions=True)
        monkeypatch.undo()
        await registry.close()
        for candidate in candidates:
            assert candidate.exitcode is not None and candidate.queues_retired
