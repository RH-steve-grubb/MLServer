import pytest
import asyncio
from types import SimpleNamespace
from typing import cast

from mlserver.types import InferenceResponse
from mlserver.parallel.errors import WorkerStop, NoWorkersAvailable
from mlserver.parallel.dispatcher import Dispatcher
from mlserver.parallel.messages import (
    ModelResponseMessage,
    ModelUpdateMessage,
    ModelRequestMessage,
)
from mlserver.parallel.worker import Worker
from mlserver.parallel.utils import terminate_queue


async def test_on_worker_stop(dispatcher: Dispatcher):
    worker = list(dispatcher._workers.values())[0]
    await worker.stop()
    dispatcher.on_worker_stop(worker, 255)

    assert worker.pid not in dispatcher._workers
    # Ensure worker is no longer in round robin rotation
    workers_count = len(dispatcher._workers)
    for _ in range(workers_count + 1):
        worker_pid = next(dispatcher._workers_round_robin)
        assert worker_pid != worker.pid


async def test_dispatch(
    dispatcher: Dispatcher,
    load_message: ModelUpdateMessage,
    inference_request_message: ModelRequestMessage,
):
    await dispatcher.dispatch_update(load_message)
    response_message = await dispatcher.dispatch_request(inference_request_message)

    assert response_message.exception is None
    inference_response = response_message.return_value
    assert isinstance(inference_response, InferenceResponse)
    assert len(inference_response.outputs) > 0


async def test_dispatch_request_no_ready_workers(
    dispatcher: Dispatcher,
    inference_request_message: ModelRequestMessage,
):
    dispatcher._ready_workers.clear()
    dispatcher._reset_round_robin()

    with pytest.raises(NoWorkersAvailable):
        await dispatcher.dispatch_request(inference_request_message)


async def test_dispatch_update_load_no_workers(
    responses,
    load_message: ModelUpdateMessage,
):
    dispatcher = Dispatcher({}, responses)
    dispatcher.start()
    try:
        with pytest.raises(NoWorkersAvailable):
            await dispatcher.dispatch_update(load_message)
    finally:
        await terminate_queue(responses)
        await dispatcher.stop()


async def test_cancel(dispatcher: Dispatcher, inference_request_message):
    worker = list(dispatcher._workers.values())[0]
    async_responses = dispatcher._async_responses
    async_responses._schedule(inference_request_message, worker)

    exit_code = 234
    async_responses.cancel(worker, exit_code)

    with pytest.raises(WorkerStop) as err:
        await async_responses._wait(inference_request_message.id)

    assert str(exit_code) in str(err)


async def test_response_loop_restarts_after_unexpected_error(dispatcher, mocker):
    process_responses = asyncio.get_running_loop().create_future()
    process_responses.set_exception(RuntimeError("response loop failed"))
    start = mocker.patch.object(dispatcher, "start")

    dispatcher._process_responses_cb(process_responses)

    start.assert_called_once_with()


async def test_response_loop_does_not_restart_after_cancellation(dispatcher, mocker):
    process_responses = asyncio.get_running_loop().create_future()
    process_responses.cancel()
    start = mocker.patch.object(dispatcher, "start")

    dispatcher._process_responses_cb(process_responses)

    start.assert_not_called()


async def test_on_worker_ready_adds_worker_to_routing(
    responses,
):
    worker_pid = 1234
    worker = cast(Worker, SimpleNamespace(pid=worker_pid))
    dispatcher = Dispatcher({}, responses)

    dispatcher.on_worker_ready(worker)

    selected_worker, selected_pid = dispatcher._get_worker()
    assert selected_worker is worker
    assert selected_pid == worker_pid


async def test_dispatch_update_waits_for_worker_start(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    dispatcher = Dispatcher({}, responses)
    worker_pid = 1234
    worker = cast(Worker, SimpleNamespace(pid=worker_pid))
    dispatcher._workers[worker_pid] = worker
    init_started = asyncio.Event()
    release_init = asyncio.Event()

    async def init_worker():
        init_started.set()
        await release_init.wait()

    async def dispatch_to_worker(_worker, _message):
        return ModelResponseMessage()

    mocker.patch.object(
        dispatcher, "dispatch_update_to_worker", side_effect=dispatch_to_worker
    )

    start_task = asyncio.create_task(dispatcher.on_worker_start(worker, init_worker()))
    await init_started.wait()

    update_task = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await asyncio.sleep(0)
    assert not update_task.done()

    release_init.set()
    await asyncio.gather(start_task, update_task)


async def test_dispatch_update_serializes_updates(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    worker_pid = 1234
    worker = cast(Worker, SimpleNamespace(pid=worker_pid))
    dispatcher = Dispatcher({worker_pid: worker}, responses)
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def dispatch_to_worker(_worker, _message):
        dispatch_started.set()
        await release_dispatch.wait()
        return ModelResponseMessage()

    mocker.patch.object(
        dispatcher, "dispatch_update_to_worker", side_effect=dispatch_to_worker
    )

    first_update = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await dispatch_started.wait()
    second_update = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await asyncio.sleep(0)

    assert not second_update.done()
    release_dispatch.set()
    await asyncio.gather(first_update, second_update)


async def test_dispatch_update_raises_non_worker_error(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, SimpleNamespace(pid=first_pid))
    second_worker = cast(Worker, SimpleNamespace(pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[RuntimeError("update failed"), ModelResponseMessage()],
    )

    with pytest.raises(RuntimeError, match="update failed"):
        await dispatcher.dispatch_update(load_message)


async def test_dispatch_update_raises_when_all_workers_stop_for_load(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, SimpleNamespace(pid=first_pid))
    second_worker = cast(Worker, SimpleNamespace(pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[WorkerStop(1), WorkerStop(2)],
    )

    with pytest.raises(NoWorkersAvailable):
        await dispatcher.dispatch_update(load_message)


async def test_dispatch_update_returns_successes_when_some_workers_stop(
    responses,
    load_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, SimpleNamespace(pid=first_pid))
    second_worker = cast(Worker, SimpleNamespace(pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    success = ModelResponseMessage()
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[WorkerStop(1), success],
    )

    assert await dispatcher.dispatch_update(load_message) == [success]


async def test_dispatch_update_allows_all_workers_to_stop_during_unload(
    responses,
    unload_message: ModelUpdateMessage,
    mocker,
):
    first_pid = 1234
    second_pid = 5678
    first_worker = cast(Worker, SimpleNamespace(pid=first_pid))
    second_worker = cast(Worker, SimpleNamespace(pid=second_pid))
    dispatcher = Dispatcher(
        {first_pid: first_worker, second_pid: second_worker}, responses
    )
    mocker.patch.object(
        dispatcher,
        "dispatch_update_to_worker",
        side_effect=[WorkerStop(1), WorkerStop(2)],
    )

    assert await dispatcher.dispatch_update(unload_message) == []


async def test_on_worker_start_defers_cancellation_until_initialization_finishes(
    responses,
):
    dispatcher = Dispatcher({}, responses)
    worker = cast(Worker, SimpleNamespace(pid=1234))
    init_started = asyncio.Event()
    release_init = asyncio.Event()

    async def init_worker():
        init_started.set()
        await release_init.wait()

    start_task = asyncio.create_task(dispatcher.on_worker_start(worker, init_worker()))
    await init_started.wait()

    start_task.cancel()
    await asyncio.sleep(0)
    assert not start_task.done()

    release_init.set()
    await start_task

    assert dispatcher._workers[worker.pid] is worker


async def test_dispatch_update_defers_cancellation_until_broadcast_finishes(
    responses, load_message, mocker
):
    worker = cast(Worker, SimpleNamespace(pid=1234))
    dispatcher = Dispatcher({worker.pid: worker}, responses)
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def dispatch_to_worker(_worker, _message):
        dispatch_started.set()
        await release_dispatch.wait()
        return ModelResponseMessage()

    mocker.patch.object(
        dispatcher, "dispatch_update_to_worker", side_effect=dispatch_to_worker
    )

    update_task = asyncio.create_task(dispatcher.dispatch_update(load_message))
    await dispatch_started.wait()

    update_task.cancel()
    await asyncio.sleep(0)
    assert not update_task.done()

    release_dispatch.set()
    await update_task


async def test_stop_cleans_up_response_processing(responses, mocker):
    worker_pid = 1234
    worker = cast(Worker, SimpleNamespace(pid=worker_pid))
    workers = {worker_pid: worker}
    dispatcher = Dispatcher(workers, responses)
    shutdown_executor = mocker.patch.object(
        dispatcher._executor, "shutdown", wraps=dispatcher._executor.shutdown
    )
    dispatcher.start()
    response_task = dispatcher._process_responses_task
    await terminate_queue(responses)

    await dispatcher.stop()

    assert not dispatcher._ready_workers
    assert dispatcher._workers == workers
    assert workers == {worker_pid: worker}
    assert response_task is not None
    assert response_task.done()
    shutdown_executor.assert_called_once_with()
    with pytest.raises(NoWorkersAvailable):
        dispatcher._get_worker()
