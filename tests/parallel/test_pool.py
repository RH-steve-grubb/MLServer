import os
import pytest
import asyncio
import numpy as np
from copy import deepcopy
from types import SimpleNamespace

from mlserver.errors import MLServerError
from mlserver.model import MLModel
from mlserver.settings import Settings, ModelSettings
from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec, StringCodec
from mlserver.parallel.pool import InferencePool, WorkerRegistry
from mlserver.parallel.errors import InferencePoolUnavailable

from ..fixtures import ErrorModel, SumModel


def check_pid(pid):
    """
    Check For the existence of a unix pid.

    From https://stackoverflow.com/a/568285/5015573
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


async def _wait_for_workers(pool: InferencePool, expected: int):
    while len(pool._workers) < expected:
        await asyncio.sleep(0.1)


def test_workers_start(inference_pool: InferencePool, settings: Settings):
    assert len(inference_pool._workers) == settings.parallel_workers
    assert (
        inference_pool._dispatcher._model_operation_timeout
        == settings.model_operation_timeout
    )
    assert (
        inference_pool._dispatcher._worker_start_timeout
        == settings.worker_start_timeout
    )

    for worker_pid in inference_pool._workers:
        assert check_pid(worker_pid)


async def test_on_worker_stop(
    settings: Settings, inference_pool: InferencePool, sum_model: MLModel, mocker
):
    # Ensure pool has some loaded models
    await inference_pool.load_model(sum_model)

    prev_workers = list(inference_pool._workers.values())
    stopped_worker = prev_workers[0]
    close_queues = mocker.spy(stopped_worker, "close_queues")

    assert stopped_worker.pid is not None
    stopped_worker.kill()
    stopped_worker.join(settings.parallel_workers_timeout)
    await inference_pool.on_worker_stop(stopped_worker.pid, 23)
    close_queues.assert_called_once_with()

    # Wait for replacement worker (started via create_task in on_worker_stop)
    await asyncio.wait_for(
        _wait_for_workers(inference_pool, settings.parallel_workers),
        timeout=5,
    )

    # Make sure worker is taken out of the rota and a new worker is started
    new_workers = list(inference_pool._workers.values())
    assert len(new_workers) == settings.parallel_workers
    for worker in new_workers:
        assert worker.pid != stopped_worker.pid


async def test_start_worker(
    settings: Settings,
    inference_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    # Ensure pool has some loaded models
    model = await inference_pool.load_model(sum_model)

    # Assert no traffic errors while new worker is starting
    start_worker_task = asyncio.create_task(inference_pool._start_worker())
    while not start_worker_task.done():
        inference_response = await model.predict(inference_request)
        assert len(inference_response.outputs) == 1

    await start_worker_task

    # Make a last pass through all workers
    for _ in range(settings.parallel_workers + 2):
        inference_response = await model.predict(inference_request)
        assert len(inference_response.outputs) == 1


async def test_start_worker_new_model(
    settings: Settings,
    inference_pool: InferencePool,
    sum_model: MLModel,
    simple_model: MLModel,
):
    # Ensure pool has some loaded models
    await inference_pool.load_model(sum_model)

    # Assert new models make their way through to the new worker
    start_worker_task = asyncio.create_task(inference_pool._start_worker())
    new_model = await inference_pool.load_model(simple_model)
    inference_request = InferenceRequest(
        inputs=[
            NumpyCodec.encode_input("foo", np.array([[1, 2]], dtype=np.int32)),
            StringCodec.encode_input("bar", ["asd", "qwe"]),
        ]
    )
    while not start_worker_task.done():
        inference_response = await new_model.predict(inference_request)
        assert len(inference_response.outputs) == 1

    await start_worker_task

    # Make a last pass through all workers
    for _ in range(settings.parallel_workers + 2):
        inference_response = await new_model.predict(inference_request)
        assert len(inference_response.outputs) == 1


async def test_close_is_terminal(inference_pool: InferencePool, sum_model: MLModel):
    worker_pids = [pid for pid in inference_pool._workers]
    inference_pool._worker_registry.add(sum_model.settings)
    inference_pool._pending_reload[inference_pool._model_key(sum_model.settings)] = (
        sum_model
    )

    await inference_pool.close()

    assert len(inference_pool._workers) == 0
    assert inference_pool.empty()
    assert not inference_pool._pending_reload
    for worker_pid in worker_pids:
        assert not check_pid(worker_pid)

    with pytest.raises(InferencePoolUnavailable):
        await inference_pool.load_model(sum_model)
    with pytest.raises(InferencePoolUnavailable):
        await inference_pool.unload_model(sum_model)
    assert await inference_pool._start_worker() is None

    # Test idempotency
    await inference_pool.close()


async def test_close_clears_tracking_when_worker_cleanup_fails(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    inference_pool._worker_registry.add(sum_model.settings)
    inference_pool._pending_reload[inference_pool._model_key(sum_model.settings)] = (
        sum_model
    )
    close_workers = inference_pool._close_workers

    async def cleanup_then_fail():
        await close_workers()
        raise RuntimeError("worker cleanup failed")

    mocker.patch.object(inference_pool, "_close_workers", new=cleanup_then_fail)

    with pytest.raises(RuntimeError, match="worker cleanup failed"):
        await inference_pool.close()

    assert inference_pool.empty()
    assert not inference_pool._pending_reload


def test_worker_registry_keeps_model_name_and_version_keys_distinct(
    sum_model_settings: ModelSettings,
):
    first_settings = deepcopy(sum_model_settings)
    assert first_settings.parameters is not None
    first_settings.name = "foo-bar"
    first_settings.parameters.version = "v1"
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.name = "foo"
    second_settings.parameters.version = "bar-v1"

    registry = WorkerRegistry()
    registry.add(first_settings)
    registry.add(second_settings)

    assert len(registry) == 2
    assert registry.has_model(first_settings)
    assert registry.has_model(second_settings)

    registry.remove(first_settings)

    assert len(registry) == 1
    assert not registry.has_model(first_settings)
    assert registry.has_model(second_settings)


async def test_pending_reload_keeps_model_name_and_version_keys_distinct(
    inference_pool: InferencePool,
    sum_model_settings: ModelSettings,
):
    first_settings = deepcopy(sum_model_settings)
    assert first_settings.parameters is not None
    first_settings.name = "foo-bar"
    first_settings.parameters.version = "v1"
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.name = "foo"
    second_settings.parameters.version = "bar-v1"
    first_model = SumModel(first_settings)
    second_model = SumModel(second_settings)

    inference_pool._pending_reload[inference_pool._model_key(first_settings)] = (
        first_model
    )
    inference_pool._pending_reload[inference_pool._model_key(second_settings)] = (
        second_model
    )

    assert len(inference_pool._pending_reload) == 2
    assert (
        inference_pool._pending_reload[inference_pool._model_key(first_settings)]
        is first_model
    )
    assert (
        inference_pool._pending_reload[inference_pool._model_key(second_settings)]
        is second_model
    )

    del inference_pool._pending_reload[inference_pool._model_key(first_settings)]

    assert len(inference_pool._pending_reload) == 1
    assert (
        inference_pool._pending_reload[inference_pool._model_key(second_settings)]
        is second_model
    )


async def test_load(
    inference_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    sum_model.settings.name = "foo"
    assert len(inference_pool._worker_registry) == 0
    model = await inference_pool.load_model(sum_model)
    assert len(inference_pool._worker_registry) == 1

    # NOTE: This should leverage the worker inference_pool, after wrapping the
    # model
    inference_response = await model.predict(inference_request)

    assert inference_response.id == inference_request.id
    assert inference_response.model_name == sum_model.settings.name
    assert len(inference_response.outputs) == 1

    await inference_pool.unload_model(sum_model)
    assert len(inference_pool._worker_registry) == 0


async def test_load_error(
    inference_pool: InferencePool,
    load_error_model: MLModel,
):
    assert len(inference_pool._worker_registry) == 0
    with pytest.raises(MLServerError) as excinfo:
        await inference_pool.load_model(load_error_model)

    assert len(inference_pool._worker_registry) == 0
    expected_msg = f"mlserver.errors.MLServerError: {ErrorModel.error_message}"
    assert str(excinfo.value) == expected_msg


async def test_start_worker_waits_for_load_lock(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    startup_started = asyncio.Event()
    worker = SimpleNamespace(pid=987654)

    async def stop_worker():
        return None

    worker.stop = stop_worker
    worker.join = lambda _timeout: None
    worker.exitcode = 0

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    mocker.patch("mlserver.parallel.pool._spawn_worker", return_value=worker)

    async def replay_worker(_worker):
        return None

    mocker.patch.object(inference_pool, "_replay_worker", new=replay_worker)

    original_on_worker_start = inference_pool._dispatcher.on_worker_start

    async def controlled_start(worker, init_coro):
        startup_started.set()
        await original_on_worker_start(worker, init_coro)

    mocker.patch.object(
        inference_pool._dispatcher, "on_worker_start", side_effect=controlled_start
    )

    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()
    start_worker_task = asyncio.create_task(inference_pool._start_worker())
    await asyncio.sleep(0)

    assert not start_worker_task.done()
    assert not startup_started.is_set()
    release_dispatch.set()
    await asyncio.gather(load_task, start_worker_task)

    assert startup_started.is_set()


async def test_load_model_serializes(
    inference_pool: InferencePool, sum_model_settings: ModelSettings, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    dispatch_count = 0

    async def controlled_dispatch(_message):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    first_settings = deepcopy(sum_model_settings)
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.name = "second-model"
    first = SumModel(first_settings)
    second = SumModel(second_settings)

    first_load = asyncio.create_task(inference_pool.load_model(first))
    await dispatch_started.wait()
    second_load = asyncio.create_task(inference_pool.load_model(second))
    await asyncio.sleep(0)

    assert not second_load.done()
    release_dispatch.set()
    await asyncio.gather(first_load, second_load)

    assert len(inference_pool._worker_registry) == 2


async def test_load_and_unload_serialize(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    dispatch_count = 0

    async def controlled_dispatch(_message):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )

    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()
    unload_task = asyncio.create_task(inference_pool.unload_model(sum_model))
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_dispatch.set()
    await asyncio.gather(load_task, unload_task)

    assert inference_pool.empty()


async def test_load_and_close_serialize(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )

    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()
    close_task = asyncio.create_task(inference_pool.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    assert not inference_pool._closing

    release_dispatch.set()
    await asyncio.gather(load_task, close_task)

    assert inference_pool._closing
    assert inference_pool.empty()


async def test_start_worker_cancellation_settles_startup(
    inference_pool: InferencePool, mocker
):
    worker = SimpleNamespace(pid=987654)
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()

    async def stop_worker():
        return None

    worker.stop = stop_worker
    worker.join = lambda _timeout: None
    worker.exitcode = 0

    async def controlled_replay(_worker):
        replay_started.set()
        await release_replay.wait()

    mocker.patch("mlserver.parallel.pool._spawn_worker", return_value=worker)
    mocker.patch.object(inference_pool, "_replay_worker", side_effect=controlled_replay)

    start_task = asyncio.create_task(inference_pool._start_worker())
    await replay_started.wait()
    start_task.cancel()
    await asyncio.sleep(0)

    assert not start_task.done()
    release_replay.set()
    await start_task

    assert worker.pid in inference_pool._dispatcher._ready_workers


async def test_start_worker_replay_failure_kills_replacement(
    inference_pool: InferencePool, mocker
):
    worker = SimpleNamespace(pid=987654)
    worker.kill = mocker.Mock()
    worker.exitcode = 0

    async def stop_worker():
        return None

    worker.stop = stop_worker
    worker.join = lambda _timeout: None

    async def replay_worker(_worker):
        raise RuntimeError("replay failed")

    mocker.patch("mlserver.parallel.pool._spawn_worker", return_value=worker)
    mocker.patch.object(inference_pool, "_replay_worker", new=replay_worker)

    with pytest.raises(RuntimeError, match="replay failed"):
        await inference_pool._start_worker()

    worker.kill.assert_called_once_with()
    assert worker.pid not in inference_pool._dispatcher._ready_workers


async def test_load_model_cancellation_settles_and_preserves_tracking(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    load_task = asyncio.create_task(inference_pool.load_model(sum_model))
    await dispatch_started.wait()

    load_task.cancel()
    await asyncio.sleep(0)
    assert not load_task.done()
    release_dispatch.set()

    await load_task

    assert inference_pool.has_model(sum_model.settings)


async def test_reload_cancellation_settles_and_preserves_pending_state(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    inference_pool._worker_registry.add(sum_model.settings)
    replacement = SumModel(deepcopy(sum_model.settings))

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    reload_task = asyncio.create_task(inference_pool.load_model(replacement))
    await dispatch_started.wait()

    reload_task.cancel()
    await asyncio.sleep(0)
    assert not reload_task.done()
    release_dispatch.set()

    await reload_task

    model_key = inference_pool._model_key(replacement.settings)
    assert inference_pool.has_model(sum_model.settings)
    assert inference_pool._pending_reload[model_key] is not None

    await inference_pool.unload_model(replacement)
    assert inference_pool.has_model(sum_model.settings)
    assert model_key not in inference_pool._pending_reload


async def test_unload_model_cancellation_settles_and_clears_tracking(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()
    inference_pool._worker_registry.add(sum_model.settings)

    async def controlled_dispatch(_message):
        dispatch_started.set()
        await release_dispatch.wait()
        return []

    mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=controlled_dispatch,
    )
    unload_task = asyncio.create_task(inference_pool.unload_model(sum_model))
    await dispatch_started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()
    release_dispatch.set()

    await unload_task

    assert inference_pool.empty()


async def test_failed_reload_preserves_committed_model_for_rollback(
    inference_pool: InferencePool, sum_model: MLModel, mocker
):
    inference_pool._worker_registry.add(sum_model.settings)
    replacement = SumModel(deepcopy(sum_model.settings))
    dispatch = mocker.patch.object(
        inference_pool._dispatcher,
        "dispatch_update",
        side_effect=RuntimeError("replacement load failed"),
    )

    with pytest.raises(RuntimeError, match="replacement load failed"):
        await inference_pool.load_model(replacement)

    model_key = inference_pool._model_key(replacement.settings)
    assert inference_pool.has_model(sum_model.settings)
    assert inference_pool._pending_reload[model_key] is replacement

    dispatch.side_effect = None
    dispatch.return_value = []
    await inference_pool.unload_model(replacement)

    assert inference_pool.has_model(sum_model.settings)
    assert model_key not in inference_pool._pending_reload
