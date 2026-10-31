"""Acquisition failures are owned through the existing async registry API."""

import asyncio
from contextlib import nullcontext
from pathlib import Path
import sys

import pytest

from mlserver.parallel.dispatcher import Dispatcher
from mlserver.parallel.errors import InferencePoolUnavailable
from mlserver.parallel.pool import InferencePool
from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.parallel.worker import Worker

from .test_startup_ownership import assert_retired
from .test_lifecycle_recovery import wait_until
from . import test_environment_ownership as environments

local_environment_tarball = environments.local_environment_tarball


@pytest.fixture(params=["queue", "zmq"])
def settings(settings, request):
    if request.param == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_inference_transport = request.param
    settings.parallel_workers = 2
    settings.parallel_workers_timeout = 1
    return settings


async def dispose(workers):
    # Baseline failures must not orphan the very children under investigation.
    for worker in workers:
        if worker.pid is not None:
            if worker.exitcode is None:
                worker.kill()
            await asyncio.to_thread(worker.join, 5)
            await worker.retire_queues(2)


@pytest.mark.parametrize(
    "failure", ["first", "later", "construction", "post-start", "dispatcher"]
)
async def test_dynamic_activation_failure_releases_children(
    settings, sum_model, monkeypatch, failure
):
    registry = InferencePoolRegistry(settings)
    workers = []
    original = Worker.start
    original_init = Worker.__init__
    primary = OSError("injected acquisition failure")

    def start(worker):
        workers.append(worker)
        if failure == "first" or (failure == "later" and len(workers) == 2):
            raise primary
        original(worker)
        if failure == "post-start":
            raise primary

    def dispatcher_start(dispatcher):
        raise primary

    def construct(worker, *args, **kwargs):
        if workers:
            raise primary
        original_init(worker, *args, **kwargs)

    try:
        # Warm an unrelated pool before the injected named-pool acquisition.
        healthy = await registry.load_model(sum_model)
        monkeypatch.setattr(Worker, "start", start)
        if failure == "construction":
            monkeypatch.setattr(Worker, "__init__", construct)
        if failure == "dispatcher":
            monkeypatch.setattr(Dispatcher, "start", dispatcher_start)
        failed = sum_model.settings.model_copy(deep=True)
        failed.name = "failed-acquisition"
        failed.parameters.inference_pool_gid = "new-pool"
        with pytest.raises(OSError) as observed:
            await registry.load_model(type(sum_model)(failed))
        assert observed.value is primary
        assert all(w.exitcode is not None for w in workers if w.pid is not None)
        for worker in workers:
            if worker.pid is not None:
                assert_retired(worker)
        assert "new-pool" not in registry._pools
        assert "new-pool" not in registry._failed_pools
        assert (await healthy.metadata()).name == sum_model.name
        monkeypatch.undo()
        retry = await registry.load_model(type(sum_model)(failed))
        assert (await retry.metadata()).name == failed.name
    finally:
        monkeypatch.undo()
        await dispose(workers)
        await registry.close()


async def test_server_start_owns_default_pool_acquisition_failure(
    settings, monkeypatch, prometheus_registry
):
    from mlserver.server import MLServer

    original = Worker.start
    workers = []

    def start(worker):
        original(worker)
        workers.append(worker)
        raise OSError("default worker exposed then failed")

    monkeypatch.setattr(Worker, "start", start)
    server = MLServer(settings)
    assert not server._inference_pool_registry._default_pool._workers
    try:
        with pytest.raises(OSError, match="default worker exposed"):
            await server.start()
        assert not server._live
        assert not server._server_tasks
        assert all(worker.exitcode is not None for worker in workers)
        for worker in workers:
            assert_retired(worker)
    finally:
        monkeypatch.undo()
        await dispose(workers)
        await server._inference_pool_registry.close()


@pytest.mark.parametrize("boundary", ["directory", "signal", "worker"])
async def test_default_activation_failure_is_owned_and_retryable(
    settings, monkeypatch, boundary
):
    import signal
    from mlserver.parallel import registry as registry_module

    previous = signal.getsignal(signal.SIGCHLD)
    registry = InferencePoolRegistry(settings)
    pool = registry._default_pool
    assert not pool._workers
    failure = OSError("default activation failed")
    workers = []
    original = Worker.start

    def fail_start(worker):
        original(worker)
        workers.append(worker)
        raise failure

    def fail(*args, **kwargs):
        raise failure

    if boundary == "directory":
        monkeypatch.setattr(registry_module.os, "makedirs", fail)
    elif boundary == "signal":
        monkeypatch.setattr(registry_module.signal, "signal", fail)
    else:
        monkeypatch.setattr(Worker, "start", fail_start)
    try:
        with pytest.raises(OSError) as observed:
            await registry.start()
        assert observed.value is failure
        assert registry._default_pool is None
        assert not registry._failed_pools
        assert signal.getsignal(signal.SIGCHLD) is previous
        assert all(worker.exitcode is not None for worker in workers)
        for worker in workers:
            assert_retired(worker)
        monkeypatch.undo()
        await registry.start()
        assert len(registry._default_pool._dispatcher._ready_workers) == 2
    finally:
        monkeypatch.undo()
        await dispose(workers)
        await registry.close()


@pytest.mark.parametrize("outcome", ["cancel", "deadline", "close"])
async def test_owned_activation_cannot_publish_after_close(
    settings, sum_model, monkeypatch, outcome
):
    registry = InferencePoolRegistry(settings)
    await registry.load_model(sum_model)
    original = InferencePool._activate
    entered, release = asyncio.Event(), asyncio.Event()
    owned = []

    async def activate(pool):
        await original(pool)
        if pool.pool_gid == "delayed":
            owned.append(pool)
            entered.set()
            await release.wait()

    monkeypatch.setattr(InferencePool, "_activate", activate)
    sum_model.settings.parameters.inference_pool_gid = "delayed"
    if outcome == "deadline":
        settings.model_operation_timeout = 0.1
    attempt = asyncio.create_task(registry.load_model(sum_model))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        workers = list(owned[0]._workers.values())
        if outcome == "cancel":
            attempt.cancel()
            await asyncio.sleep(0)
            assert not attempt.done()
            release.set()
            assert (await asyncio.wait_for(attempt, 10)).name == sum_model.name
        elif outcome == "deadline":
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(attempt, 10)
            assert "delayed" not in registry._pools
            assert all(w.exitcode is not None for w in workers)
            for worker in workers:
                assert_retired(worker)
        else:
            # Close the actual pool while initialization holds its lock.
            closing = asyncio.create_task(owned[0].close())
            await wait_until(lambda: owned[0]._closing)
            release.set()
            with pytest.raises(InferencePoolUnavailable):
                await asyncio.wait_for(attempt, 10)
            await asyncio.wait_for(closing, 10)
            assert not owned[0]._dispatcher._ready_workers
            assert all(w.exitcode is not None for w in workers)
    finally:
        release.set()
        await asyncio.gather(
            attempt, *([closing] if closing else []), return_exceptions=True
        )
        await registry.close()


async def test_failed_rollback_retains_owner_and_receives_reaped_exit(
    settings, sum_model, monkeypatch
):
    registry = InferencePoolRegistry(settings)
    await registry.load_model(sum_model)
    original_start, original_kill = Worker.start, Worker.kill
    primary = OSError("post-start failure")
    children = []
    settings.parallel_workers_timeout = 0.05

    def start(worker):
        original_start(worker)
        children.append(worker)
        raise primary

    async def stop(worker):
        pass

    monkeypatch.setattr(Worker, "start", start)
    monkeypatch.setattr(Worker, "stop", stop)
    monkeypatch.setattr(Worker, "kill", lambda worker: None)
    sum_model.settings.parameters.inference_pool_gid = "retained"
    pool = None
    try:
        with pytest.raises(OSError) as observed:
            await registry.load_model(sum_model)
        assert observed.value is primary
        pool = registry._failed_pools["retained"]
        assert primary.cleanup_error is pool._cleanup_error
        assert children[0].is_alive()
        if pool._transport_directory is not None:
            assert Path(pool._transport_directory.name).exists()
        with pytest.raises(InferencePoolUnavailable):
            await registry.load_model(sum_model)
        for _ in range(2):
            with pytest.raises(type(primary.cleanup_error)) as repeated:
                await pool.close()
            assert repeated.value is primary.cleanup_error
        # The production reaper must notify failed owners as well as active
        # pools. Do not poll/join first: that would consume the exit ourselves.
        original_kill(children[0])
        await asyncio.wait_for(
            wait_until(lambda: children[0]._popen.returncode is not None), 5
        )
        assert children[0].exitcode == -9
    finally:
        monkeypatch.undo()
        await dispose(children)
        if pool is not None:
            # Test-only recovery after the injected unrecoverable close. Public
            # callers continue to receive its retained error and blocked key.
            await pool._close()
        with pytest.raises(Exception) if pool is not None else nullcontext():
            await registry.close()


@pytest.mark.parametrize("boundary", ["context", "socket", "bind"])
async def test_parent_transport_setup_failure_has_a_cleanup_owner(
    settings, sum_model, monkeypatch, boundary
):
    if settings.parallel_inference_transport != "zmq":
        pytest.skip("ZeroMQ acquisition boundary")
    from mlserver.parallel import zmq as transport

    registry = InferencePoolRegistry(settings)
    await registry.load_model(sum_model)
    contexts = []
    original_context, original_endpoint = (
        transport.zmq.asyncio.Context,
        transport.create_endpoint,
    )

    def context(*args, **kwargs):
        if boundary == "context":
            raise OSError("context setup failed")
        value = original_context(*args, **kwargs)
        contexts.append(value)
        return value

    def endpoint(directory, kind):
        if kind == "r":
            return f"ipc://{directory / 'missing' / 'r.sock'}"
        return original_endpoint(directory, kind)

    monkeypatch.setattr(transport.zmq.asyncio, "Context", context)
    if boundary == "socket":

        def fail(*args):
            raise OSError("socket setup failed")

        monkeypatch.setattr(transport, "_socket", fail)
    elif boundary == "bind":
        monkeypatch.setattr(transport, "create_endpoint", endpoint)
    sum_model.settings.parameters.inference_pool_gid = "transport-failure"
    try:
        with pytest.raises((OSError, transport.zmq.ZMQError)):
            await registry.load_model(sum_model)
        assert contexts == [] or all(value.closed for value in contexts)
        assert "transport-failure" not in registry._pools
        assert "transport-failure" not in registry._failed_pools
    finally:
        monkeypatch.undo()
        await registry.close()


@pytest.mark.parametrize("borrowed", [False, True])
async def test_failed_environment_pool_activation_releases_only_owned_files(
    settings,
    sum_model,
    local_environment_tarball,
    development_mode,
    monkeypatch,
    borrowed,
):
    registry = InferencePoolRegistry(settings)
    await registry.load_model(sum_model)
    original = Worker.start
    pools, children = [], []

    def start(worker):
        pools.extend(registry._pools.values())
        original(worker)
        children.append(worker)
        raise OSError("environment worker start failed")

    monkeypatch.setattr(Worker, "start", start)
    parameters = sum_model.settings.parameters
    if borrowed:
        parameters.environment_path = sys.prefix
    else:
        parameters.environment_tarball = str(local_environment_tarball)
    try:
        with pytest.raises(OSError, match="environment worker start"):
            await registry.load_model(sum_model)
        path = Path(pools[0]._env._env_path)
        assert path.exists() is borrowed
        assert all(w.exitcode is not None for w in children)
        for worker in children:
            assert_retired(worker)
    finally:
        monkeypatch.undo()
        await dispose(children)
        await registry.close()


async def test_transport_close_failure_still_joins_response_reader(
    settings, monkeypatch
):
    if settings.parallel_inference_transport != "zmq":
        pytest.skip("ZeroMQ transport cleanup boundary")
    pool = InferencePool(settings)
    await pool.start()
    transport = pool._dispatcher._transport
    original = transport.close
    failure = OSError("transport cleanup failure")

    async def close():
        await original()
        raise failure

    monkeypatch.setattr(transport, "close", close)
    try:
        with pytest.raises(OSError) as observed:
            await pool.close()
        assert observed.value is failure
        assert pool._dispatcher._process_responses_task.done()
        assert all(not t.is_alive() for t in pool._dispatcher._executor._threads)
        assert not pool._workers
        with pytest.raises(OSError) as repeated:
            await pool.close()
        assert repeated.value is failure
    finally:
        monkeypatch.undo()
        await pool._close()


async def test_close_owns_extraction_before_transfer_to_a_pool(
    settings, sum_model, local_environment_tarball, development_mode, monkeypatch
):
    registry = InferencePoolRegistry(settings)
    await registry.load_model(sum_model)
    original = registry._prepare_environment
    entered, release = asyncio.Event(), asyncio.Event()
    prepared = []

    async def prepare(*args):
        env = await original(*args)
        prepared.append(env)
        entered.set()
        await release.wait()
        return env

    monkeypatch.setattr(registry, "_prepare_environment", prepare)
    sum_model.settings.parameters.environment_tarball = str(local_environment_tarball)
    attempt = asyncio.create_task(registry.load_model(sum_model))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        path = Path(prepared[0]._env_path)
        assert path.exists()
        closing = asyncio.create_task(registry.close())
        await asyncio.wait_for(wait_until(lambda: registry._closing), 5)
        release.set()
        with pytest.raises(InferencePoolUnavailable):
            await asyncio.wait_for(attempt, 10)
        await asyncio.wait_for(closing, 10)
        assert not path.exists()
        assert not registry._environment_tasks
    finally:
        release.set()
        await asyncio.gather(
            attempt, *([closing] if closing else []), return_exceptions=True
        )
        await registry.close()


async def test_partial_worker_queue_allocation_is_released(
    settings, sum_model, monkeypatch
):
    from mlserver.parallel import worker as worker_module

    registry = InferencePoolRegistry(settings)
    acquired = []
    original = worker_module.Queue

    def queue():
        if acquired:
            raise OSError("second worker queue failed")
        value = original()
        acquired.append(value)
        return value

    try:
        await registry.load_model(sum_model)
        monkeypatch.setattr(worker_module, "Queue", queue)
        sum_model.settings.parameters.inference_pool_gid = "partial-queue"
        with pytest.raises(OSError, match="second worker queue"):
            await registry.load_model(sum_model)
        assert acquired[0]._reader.closed
        assert acquired[0]._writer.closed
    finally:
        monkeypatch.undo()
        for value in acquired:
            value.close()
            value._reader.close()
            value._writer.close()
            value.join_thread()
        await registry.close()
