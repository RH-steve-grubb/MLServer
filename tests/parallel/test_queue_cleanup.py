"""Real pipe backpressure while a worker is blocked in native preparation."""

import asyncio
from pathlib import Path
import threading

from mlserver.parallel.messages import ModelRequestMessage
from mlserver.parallel.pool import InferencePool

from tests.fixtures import SumModel


async def block_worker(model):
    Path(model.settings.parameters.extra["entered_file"]).touch()
    # Deliberately block this process's event loop, as native runtime work can.
    # No shared condition is touched after its waiter is killed.
    threading.Event().wait()
    return model


async def test_close_settles_writers_after_blocked_worker_exit(
    settings, sum_model_settings, tmp_path
):
    settings.parallel_workers = 1
    settings.parallel_workers_timeout = 1
    settings.model_operation_timeout = 10
    entered = tmp_path / "entered"
    sum_model_settings.parameters.extra["entered_file"] = str(entered)
    pool = InferencePool(settings, on_worker_load=[block_worker])
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
