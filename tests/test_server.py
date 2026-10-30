import asyncio

import pytest

from mlserver import MLServer
from mlserver.settings import Settings


async def test_server_start_is_single_flight(mocker, prometheus_registry):
    server = MLServer(Settings())
    transports_started = asyncio.Event()
    transports_released = asyncio.Event()
    started_count = 0
    transport_count = 2 + int(server._metrics_server is not None)

    async def transport():
        nonlocal started_count
        started_count += 1
        if started_count == transport_count:
            transports_started.set()
        await transports_released.wait()

    async def stop_transport(*args, **kwargs):
        transports_released.set()

    for transport_server in (
        server._rest_server,
        server._grpc_server,
        server._metrics_server,
    ):
        if transport_server is not None:
            mocker.patch.object(transport_server, "start", new=transport)
            mocker.patch.object(transport_server, "stop", new=stop_transport)

    startup_task = asyncio.create_task(server.start())
    await transports_started.wait()

    assert len(server._server_tasks) == transport_count
    assert not server._startup_model_tasks
    assert server._stop_task is None
    assert server._live

    await server.start()
    assert len(server._server_tasks) == transport_count
    assert not startup_task.done()

    await server.stop()
    await startup_task

    assert not server._server_tasks
    assert not server._startup_model_tasks
    assert server._stop_task is None
    assert not server._live


async def test_start_returns_while_shutdown_is_in_progress(mocker, prometheus_registry):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def stop_resources(_sig=None):
        cleanup_started.set()
        await release_cleanup.wait()

    mocker.patch.object(server, "_stop_resources", new=stop_resources)

    try:
        stop_task = asyncio.create_task(server.stop())
        await cleanup_started.wait()

        await server.start()
        assert not server._live
    finally:
        release_cleanup.set()
        await stop_task
        server._stop_resources = original_stop_resources
        await server.stop()


async def test_stop_cancels_server_tasks_when_resource_cleanup_fails(
    mocker, prometheus_registry
):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    task_started = asyncio.Event()

    async def server_task_body():
        task_started.set()
        await asyncio.Future()

    server_task = asyncio.create_task(server_task_body())
    server._server_tasks.append(server_task)
    await task_started.wait()

    async def failing_stop_resources(_sig=None):
        raise RuntimeError("resource cleanup failed")

    mocker.patch.object(server, "_stop_resources", new=failing_stop_resources)

    try:
        await server.stop()

        assert server_task.cancelled()
        assert server._stop_task is None
    finally:
        server._stop_resources = original_stop_resources
        await server.stop()


async def test_stop_cancels_startup_model_tasks(mocker, prometheus_registry):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    task_started = asyncio.Event()

    async def model_task_body():
        task_started.set()
        await asyncio.Future()

    model_task = asyncio.create_task(model_task_body())
    server._startup_model_tasks.append(model_task)
    await task_started.wait()
    mocker.patch.object(server, "_stop_resources", new=mocker.AsyncMock())

    try:
        await server.stop()

        assert model_task.cancelled()
        assert server._stop_task is None
    finally:
        server._stop_resources = original_stop_resources
        await server.stop()


async def test_start_validation_failure_resets_live(mocker, prometheus_registry):
    server = MLServer(Settings())
    mocker.patch(
        "mlserver.server.log_runtime_security_mode",
        side_effect=RuntimeError("invalid trusted runtimes"),
    )

    try:
        with pytest.raises(RuntimeError, match="invalid trusted runtimes allowlist"):
            await server.start()

        assert not server._live
        assert server._stop_task is None
    finally:
        await server.stop()


async def test_server_propagates_model_operation_timeout(prometheus_registry):
    model_operation_timeout = 37
    server = MLServer(Settings(model_operation_timeout=model_operation_timeout))

    try:
        assert (
            server._model_registry._model_operation_timeout == model_operation_timeout
        )
        assert (
            server._model_repository_handlers._model_operation_timeout
            == model_operation_timeout
        )
    finally:
        await server.stop()


async def test_repeated_stop_does_not_start_duplicate_cleanup(
    mocker, prometheus_registry
):
    server = MLServer(Settings())
    original_stop_resources = server._stop_resources
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_calls = 0

    async def stop_resources(_sig=None):
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_started.set()
        await release_cleanup.wait()

    mocker.patch.object(server, "_stop_resources", new=stop_resources)

    try:
        first_stop = asyncio.create_task(server.stop())
        await cleanup_started.wait()
        await server.stop()

        assert cleanup_calls == 1
    finally:
        release_cleanup.set()
        await first_stop
        server._stop_resources = original_stop_resources
        await server.stop()
