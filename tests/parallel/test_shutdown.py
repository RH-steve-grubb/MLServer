"""Default Queue shutdown regressions, independent of optional IPC transports."""

import asyncio
from threading import Event

from pytest_mock import MockerFixture

from mlserver.parallel.pool import InferencePool
from mlserver.settings import Settings


async def test_close_joins_blocked_response_reader_before_closing_queue(
    settings: Settings, mocker: MockerFixture
) -> None:
    pool = InferencePool(settings)
    workers = list(pool._workers.values())
    dispatcher = pool._dispatcher
    response_task = dispatcher._process_responses_task
    assert response_task is not None
    entered = Event()
    original_get = pool._responses.get

    def blocking_get():
        entered.set()
        return original_get()

    # Construction schedules the response task, but this test has not yielded
    # yet. Observe its real Queue.get(), with no model updates to wake it.
    mocker.patch.object(pool._responses, "get", side_effect=blocking_get)
    original_close = pool._responses.close
    observations = []

    def close_queue() -> None:
        observations.append(
            (
                response_task.done(),
                all(not thread.is_alive() for thread in dispatcher._executor._threads),
            )
        )
        original_close()

    mocker.patch.object(pool._responses, "close", side_effect=close_queue)
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)
        assert not response_task.done()
        await asyncio.wait_for(pool.close(), timeout=10)
        # A cancelled executor Future alone is insufficient: its reader thread
        # must also finish before Queue.close() invalidates the pipe handle.
        assert observations == [(True, True)]
        assert not dispatcher._active
        assert all(not worker.is_alive() for worker in workers)
        await asyncio.wait_for(pool.close(), timeout=1)
        assert observations == [(True, True)]
    finally:
        await asyncio.wait_for(pool.close(), timeout=10)


async def test_closed_dispatcher_does_not_restart_after_response_task_failure(
    inference_pool: InferencePool, mocker: MockerFixture
) -> None:
    await asyncio.wait_for(inference_pool.close(), timeout=10)
    dispatcher = inference_pool._dispatcher
    start = mocker.patch.object(dispatcher, "start")
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(RuntimeError("late response-reader failure"))
    dispatcher._process_responses_cb(failed)
    start.assert_not_called()
    assert not dispatcher._active
