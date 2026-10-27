import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock

import pytest

from mlserver.parallel.dispatcher import AsyncResponses
from mlserver.parallel.messages import ModelRequestMessage, ModelResponseMessage


async def checkpoint() -> None:
    """Run already-scheduled callbacks before resuming the test."""
    loop = asyncio.get_running_loop()
    reached = loop.create_future()
    loop.call_soon(reached.set_result, None)
    await reached


@pytest.fixture
async def callback_errors() -> AsyncIterator[list[dict[str, Any]]]:
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _, context: errors.append(context))
    try:
        yield errors
    finally:
        try:
            await checkpoint()
        finally:
            loop.set_exception_handler(previous)
    assert not errors


@pytest.fixture
def message() -> ModelRequestMessage:
    return ModelRequestMessage(model_name="selected-model", method_name="predict")


def assert_no_pending(responses: AsyncResponses) -> None:
    assert not responses._futures
    assert not responses._futures_map
    # The worker index may retain empty sets for workers still in the pool.
    assert not any(responses._workers_map.values())


async def test_cancellation_after_resolve_before_callback_cleans_request(
    message: ModelRequestMessage, callback_errors: list[dict[str, Any]]
) -> None:
    responses = AsyncResponses()
    future = responses._schedule(message, Mock(pid=1))
    responses.resolve(ModelResponseMessage(id=message.id, return_value="done"))
    future.cancel()
    with pytest.raises(asyncio.CancelledError):
        await responses._wait(message.id)
    await checkpoint()
    assert_no_pending(responses)
    assert not callback_errors


async def test_response_completion_wins_scheduled_worker_failure(
    message: ModelRequestMessage, callback_errors: list[dict[str, Any]]
) -> None:
    responses = AsyncResponses()
    worker = Mock(pid=1)
    future = responses._schedule(message, worker)
    expected = ModelResponseMessage(id=message.id, return_value="completed")
    # Both callbacks are queued while the future is pending. Completion runs
    # first, so the worker-stop callback must leave the result intact.
    responses.resolve(expected)
    responses.cancel(worker, exit_code=255)
    assert await responses._wait(message.id) is expected
    await checkpoint()
    assert future.result() is expected
    assert_no_pending(responses)
    assert not callback_errors


async def test_fail_all_wakes_waiters_and_ignores_completed_or_cancelled_futures(
    message: ModelRequestMessage, callback_errors: list[dict[str, Any]]
) -> None:
    responses = AsyncResponses()
    messages = [
        message.model_copy(update={"id": f"{message.id}-{i}"}) for i in range(4)
    ]
    futures = [responses._schedule(item, Mock(pid=1)) for item in messages]
    expected = ModelResponseMessage(id=messages[2].id, return_value="already done")
    futures[2].set_result(expected)
    futures[3].cancel()
    waiters = [asyncio.create_task(responses._wait(item.id)) for item in messages[:2]]
    await checkpoint()
    assert all(not waiter.done() for waiter in waiters)
    error = RuntimeError("response transport failed")
    # Keep the completed and cancelled futures tracked when fail_all schedules
    # its callbacks; each callback must check the actual completion state.
    responses.fail_all(error)
    assert await responses._wait(messages[2].id) is expected
    with pytest.raises(asyncio.CancelledError):
        await responses._wait(messages[3].id)
    results = await asyncio.wait_for(
        asyncio.gather(*waiters, return_exceptions=True), timeout=1
    )
    assert len(results) == 2 and all(result is error for result in results)
    await checkpoint()
    assert_no_pending(responses)
    assert not callback_errors
