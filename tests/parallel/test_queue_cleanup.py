"""Real pipe backpressure while a worker is blocked in native preparation."""

import asyncio
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from multiprocessing import get_context

import pytest

from mlserver.parallel.messages import ModelRequestMessage
from mlserver.parallel.pool import InferencePool
from mlserver.parallel.worker import Worker

from tests.fixtures import SumModel


async def test_failed_retirement_remains_retryable(monkeypatch):
    # Purpose: Exercise partial parent-queue retirement failure with a supplied
    # confirmed-exit state.
    # Expect: Independent handles close, the failed handle remains owned and a retry
    # completes native endpoint release idempotently.
    # Method: Use real spawn-context queues with a lightweight owner, fault one writer
    # close, restore it and inspect actual readers/writers; this is not reaper proof.
    # Unit-level ownership with actual parent pipe endpoints and a supplied
    # confirmed-exit state. Process/reaper behavior is covered by native tests.
    queues = [get_context("spawn").Queue(), get_context("spawn").Queue()]
    owner = SimpleNamespace(
        exitcode=0,
        pid=0,
        _queues_retired=False,
        _queues=queues,
        _requests=queues[0],
        _model_updates=queues[1],
        close_queues=lambda: [queue.close() for queue in queues],
    )
    writer = queues[0]._writer
    original_close = writer.close
    failure = RuntimeError("deliberate parent endpoint cleanup failure")

    def fail():
        raise failure

    try:
        monkeypatch.setattr(writer, "close", fail)
        with pytest.raises(RuntimeError) as observed:
            await Worker.retire_queues(owner, 1)
        assert observed.value is failure
        assert not owner._queues_retired
        assert not writer.closed
        assert queues[1]._reader.closed and queues[1]._writer.closed
        monkeypatch.setattr(writer, "close", original_close)
        await Worker.retire_queues(owner, 1)
        await Worker.retire_queues(owner, 1)
        assert owner._queues_retired
        assert all(queue._reader.closed and queue._writer.closed for queue in queues)
    finally:
        monkeypatch.setattr(writer, "close", original_close)
        for queue in queues:
            queue.close()
            queue._reader.close()
            queue._writer.close()
            queue.join_thread()


async def test_retirement_does_not_race_feeder_connection_close(
    settings, sum_model_settings, monkeypatch
):
    # Purpose: Exercise competing feeder and retirement closure of one native parent
    # reader.
    # Expect: Exactly one native descriptor close occurs and the feeder terminates after
    # retirement.
    # Method: Pause inside actual descriptor cleanup after a real child exits, observe
    # the competing lock attempt, then release and count native close calls.
    from mlserver.parallel import worker as worker_module

    # Observe both contenders before either can win native descriptor cleanup.
    # The production lock, when present, is still a real threading lock.
    entered, contended, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    locks = []

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.attempts = 0
            locks.append(self)

        def __enter__(self):
            self.attempts += 1
            if self is locks[1] and self.attempts == 2:
                contended.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(worker_module, "Lock", ObservedLock, raising=False)
    settings.parallel_workers = 1
    pool = InferencePool(settings)
    await pool.start()
    worker = next(iter(pool._workers.values()))
    queue = worker._model_updates
    reader = queue._reader
    native_calls = []
    retirement = None

    def native_close():
        # Connection.close reads the descriptor before its native close can
        # release the GIL. Pause that interval to force the competing closer.
        descriptor = reader.fileno()
        native_calls.append(descriptor)
        if len(native_calls) == 1:
            entered.set()
            assert release.wait(10)
        else:
            contended.set()
        os.close(descriptor)

    try:
        await pool.load_model(SumModel(sum_model_settings))
        worker.kill()
        await asyncio.to_thread(worker.join, 5)
        assert worker.exitcode is not None
        monkeypatch.setattr(reader, "_close", native_close)
        queue.close()
        assert await asyncio.to_thread(entered.wait, 5)
        retirement = asyncio.create_task(
            asyncio.to_thread(lambda: asyncio.run(worker.retire_queues(5)))
        )
        assert await asyncio.to_thread(contended.wait, 5)
        assert len(native_calls) == 1
        release.set()
        await asyncio.wait_for(retirement, 10)
        assert not queue._thread.is_alive()
        assert len(native_calls) == 1
    finally:
        release.set()
        if retirement is not None:
            await asyncio.gather(retirement, return_exceptions=True)
        await pool.close()


async def block_worker(model):
    Path(model.settings.parameters.extra["entered_file"]).touch()
    # Deliberately block this process's event loop, as native runtime work can.
    # No shared condition is touched after its waiter is killed.
    threading.Event().wait()
    return model


async def test_close_settles_writers_after_blocked_worker_exit(
    settings, sum_model_settings, tmp_path
):
    # Purpose: Exercise shutdown with a real pipe writer backpressured behind a child
    # blocked like native preparation.
    # Expect: Bounded shutdown confirms child exit and terminates the obsolete parent
    # feeder instead of leaving interpreter-exit cleanup blocked.
    # Method: Use a file entry marker and a frame larger than pipe capacity, establish
    # writer entry by reader readability, then close and inspect exit/feeder state.
    settings.parallel_workers = 1
    settings.parallel_workers_timeout = 1
    settings.model_operation_timeout = 10
    entered = tmp_path / "entered"
    sum_model_settings.parameters.extra["entered_file"] = str(entered)
    pool = InferencePool(settings, on_worker_load=[block_worker])
    await pool.start()
    worker = next(iter(pool._workers.values()))
    attempt = asyncio.create_task(pool.load_model(SumModel(sum_model_settings)))

    async def await_entry():
        while not entered.exists():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(await_entry(), 5)
        worker.send_request(
            ModelRequestMessage(
                model_name=sum_model_settings.name,
                method_name="predict",
                method_kwargs={"payload": b"x" * (2 * 1024 * 1024)},
            )
        )
        # Readability proves that the writer entered the pipe. The blocked
        # consumer cannot drain this frame, which exceeds the pipe capacity.
        while not worker._requests._reader.poll():
            await asyncio.sleep(0.01)
        pool._settings.model_operation_timeout = 0.1
        await asyncio.wait_for(pool.close(), 15)
        await asyncio.gather(attempt, return_exceptions=True)
        assert worker.exitcode is not None
        assert not worker._requests._thread.is_alive()
    finally:
        # Also clean the affected implementation's blocked writer: after its
        # consumer exits, closing the reader forces the obsolete write to fail.
        if worker.is_alive():
            worker.kill()
        await asyncio.to_thread(worker.join, 5)
        for queue in [worker._requests, worker._model_updates]:
            queue.close()
            queue._reader.close()
            thread = getattr(queue, "_thread", None)
            if thread is not None:
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive()
            queue._writer.close()
        await asyncio.gather(attempt, return_exceptions=True)
        await pool.close()
