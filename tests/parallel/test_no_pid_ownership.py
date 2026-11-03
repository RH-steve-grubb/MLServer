"""Receiving shutdown owners must handle partially acquired, never-started workers."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from mlserver.parallel.pool import InferencePool
from mlserver.parallel.worker import Worker
from tests.fixtures import SumModel

from .test_lifecycle_recovery import wait_until
from .test_replacement_ownership import fail_acquisition


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


@pytest.mark.parametrize("boundary", ["second-queue", "process-start"])
@pytest.mark.parametrize("persistent", [False, True])
async def test_close_owns_no_pid_candidate_after_failed_rollback(
    settings, sum_model_settings, monkeypatch, boundary, persistent
):
    stopped = []

    async def on_stop(worker):
        stopped.append(worker)

    pool = InferencePool(settings, on_worker_stop=[on_stop])
    await pool.start()
    healthy = list(pool._workers.values())
    original_retire = Worker.retire_queues
    candidates = []
    try:
        model = await pool.load_model(SumModel(sum_model_settings))
        candidates, primary = fail_acquisition(monkeypatch, boundary)
        cleanup = OSError("no-PID queue retirement failed")

        async def fail(worker, timeout):
            if worker in candidates:
                raise cleanup
            await original_retire(worker, timeout)

        monkeypatch.setattr(Worker, "retire_queues", fail)
        with pytest.raises(OSError) as observed:
            await pool._start_worker()
        candidate = candidates[0]
        assert observed.value is primary
        assert getattr(primary, "cleanup_error") is cleanup
        assert candidate.pid is None and candidate._queues
        assert not candidate.queues_retired
        assert (await model._send("metadata")).name == model.name
        monkeypatch.undo()
        forbidden = [
            Mock(side_effect=AssertionError("never-started process operation"))
            for _ in range(3)
        ]
        for method, mock in zip(("stop", "kill", "join"), forbidden):
            monkeypatch.setattr(candidate, method, mock)
        if persistent:
            monkeypatch.setattr(Worker, "retire_queues", fail)
            with pytest.raises(OSError) as close_error:
                await asyncio.wait_for(pool.close(), 10)
            assert close_error.value is cleanup
            with pytest.raises(OSError) as repeated:
                await pool.close()
            assert repeated.value is cleanup
            assert pool._closing_workers[id(candidate)] is candidate
            assert not candidate.queues_retired
            assert any(not queue._reader.closed for queue in candidate._queues)
        else:
            await asyncio.wait_for(pool.close(), 10)
            await pool.close()
            assert candidate.queues_retired
            assert all(
                queue._reader.closed and queue._writer.closed
                for queue in candidate._queues
            )
            assert all(
                getattr(queue, "_thread", None) is None or not queue._thread.is_alive()
                for queue in candidate._queues
            )
            assert not pool._closing_workers
            if pool._transport_directory is not None:
                assert not Path(pool._transport_directory.name).exists()
        assert all(worker.exitcode is not None for worker in healthy)
        assert set(stopped) == set(healthy) and candidate not in stopped
        for mock in forbidden:
            mock.assert_not_called()
    finally:
        monkeypatch.undo()
        if pool._close_task is None:
            await pool.close()
        else:
            await asyncio.gather(pool._close_task, return_exceptions=True)
        # Explicit test teardown for both the affected baseline and deliberately
        # persistent failures. Production repeats the recorded public result.
        for candidate in candidates:
            await original_retire(candidate, 2)
        assert all(worker.exitcode is not None for worker in healthy)
        if pool._transport_directory is not None:
            pool._transport_directory.cleanup()


async def test_close_takes_over_no_pid_retirement_during_rollback(
    settings, sum_model_settings, monkeypatch
):
    pool = InferencePool(settings)
    await pool.start()
    healthy = list(pool._workers.values())
    original_retire = Worker.retire_queues
    entered, release, taken_over = asyncio.Event(), asyncio.Event(), asyncio.Event()
    candidates = []
    attempt = closing = None
    try:
        await pool.load_model(SumModel(sum_model_settings))
        candidates, primary = fail_acquisition(monkeypatch, "second-queue")
        cleanup = OSError("initial retirement interrupted")
        calls = 0

        async def retire(worker, timeout):
            nonlocal calls
            if worker in candidates:
                calls += 1
                if calls == 1:
                    entered.set()
                    await release.wait()
                    raise cleanup
                taken_over.set()
            await original_retire(worker, timeout)

        monkeypatch.setattr(Worker, "retire_queues", retire)
        attempt = asyncio.create_task(pool._start_worker())
        await asyncio.wait_for(entered.wait(), 5)
        candidate = candidates[0]
        assert candidate.pid is None and candidate in pool._closing_workers.values()
        # Allocation normally does not yield. Hold the first retirement after
        # acquisition, then force bounded lock draining to test the receiving owner.
        pool._settings.model_operation_timeout = 0.05
        closing = asyncio.create_task(pool.close())
        await asyncio.wait_for(wait_until(lambda: pool._closing), 5)
        await asyncio.wait_for(taken_over.wait(), 5)
        release.set()
        with pytest.raises(OSError) as observed:
            await asyncio.wait_for(attempt, 5)
        assert observed.value is primary
        assert getattr(primary, "cleanup_error") is cleanup
        await asyncio.wait_for(closing, 10)
        assert candidate.queues_retired and not pool._closing_workers
        assert all(
            queue._reader.closed and queue._writer.closed for queue in candidate._queues
        )
        assert all(worker.exitcode is not None for worker in healthy)
    finally:
        release.set()
        await asyncio.gather(
            *(t for t in (attempt, closing) if t is not None), return_exceptions=True
        )
        monkeypatch.undo()
        if pool._close_task is None:
            await pool.close()
        else:
            await asyncio.gather(pool._close_task, return_exceptions=True)
        for candidate in candidates:
            await original_retire(candidate, 2)
        if pool._transport_directory is not None:
            pool._transport_directory.cleanup()
