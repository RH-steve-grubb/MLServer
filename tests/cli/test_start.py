import asyncio
import os
import pytest
import signal
import sys

from aiohttp.client_exceptions import ClientResponseError
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from subprocess import Popen, TimeoutExpired

from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest

from ..utils import (
    RESTClient,
    TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV,
    get_available_ports,
)
from .test_start_cases import case_custom_module, case_sum_model


def _spawn_mlserver(folder: str) -> Popen:
    # Use the same interpreter as the running test env so imports resolve
    # consistently across tox and local runs.
    # This fixture depends on repository-root `conftest.py` bootstrap setup,
    # which pre-populates PYTHONPATH and trusted-runtime artifact env for
    # spawned subprocesses.
    repo_root = str(Path(__file__).resolve().parents[2])
    subprocess_env = {
        key: value for key, value in os.environ.items() if key != "PYTHONHOME"
    }
    if "PYTHONPATH" not in subprocess_env:
        raise RuntimeError("Missing PYTHONPATH test bootstrap env.")
    if TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV not in subprocess_env:
        raise RuntimeError("Missing trusted-runtimes test artifact env.")
    return Popen(
        [sys.executable, "-m", "mlserver.cli.main", "start", folder],
        cwd=repo_root,
        start_new_session=True,
        env=subprocess_env,
    )


def _stop_mlserver(process: Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process may have already exited before fixture teardown runs.
        pass
    try:
        process.wait(timeout=10)
    except TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except TimeoutExpired:
            pass  # Give up; process is unkillable (zombie or kernel issue)


@pytest.fixture
def settings(settings: Settings, free_ports: tuple[int, int, int]) -> Settings:
    http_port, grpc_port, metrics_port = free_ports

    settings.http_port = http_port
    settings.grpc_port = grpc_port
    settings.metrics_port = metrics_port

    return settings


@pytest.fixture
def mlserver_start_sum_model(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Generator[Popen, None, None]:
    # Baseline scenario: importable runtime (`tests.fixtures.SumModel`).
    sum_model_folder = case_sum_model(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(sum_model_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
def mlserver_start_custom_module(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Generator[Popen, None, None]:
    # Security scenario: model-folder module (`custom.SumModel`) should not load.
    custom_module_folder = case_custom_module(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(custom_module_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
async def rest_client(settings: Settings) -> AsyncGenerator[RESTClient, None]:
    http_server = f"127.0.0.1:{settings.http_port}"
    client = RESTClient(http_server)

    yield client

    await client.close()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_live(rest_client: RESTClient):
    await rest_client.wait_until_live()
    is_live = await rest_client.live()
    assert is_live

    # Assert that the server is live, but some models are still loading
    with pytest.raises(ClientResponseError):
        await rest_client.ready()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_infer(
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    await rest_client.wait_until_model_ready(sum_model_settings.name)
    response = await rest_client.infer(sum_model_settings.name, inference_request)

    assert len(response.outputs) == 1


async def test_custom_module_fails_closed(
    mlserver_start_custom_module: Popen,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # Fail closed when runtime points to a non-allowlisted model-folder module.
    # This also proves spawned workers (parallel_workers=2) are in PRODUCTION mode
    # with the allowlist enforced. If worker bootstrap failed and workers degraded
    # to DEVELOPMENT mode, custom.SumModel would load successfully.
    # See: test_spawned_workers_load_allowlisted_runtime_via_bootstrap (positive test)
    await rest_client.wait_until_live()
    with pytest.raises(ClientResponseError):
        await rest_client.wait_until_model_ready(sum_model_settings.name)


async def test_custom_module_loads_in_development_mode(
    development_mode,
    mlserver_start_custom_module: Popen,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # In DEVELOPMENT mode, custom.SumModel should load successfully
    await rest_client.wait_until_live()
    await rest_client.wait_until_model_ready(sum_model_settings.name)


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_spawned_workers_load_allowlisted_runtime_via_bootstrap(
    settings: Settings,
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # This verifies the trusted-runtime bootstrap is applied inside spawned
    # worker processes (parallel_workers > 0). Without that bootstrap, workers
    # would fall back to DEVELOPMENT mode and accept any runtime.
    #
    # Positive test: allowlisted runtime loads successfully in workers.
    # For proof that workers are in PRODUCTION mode (not DEVELOPMENT fallback),
    # see: test_custom_module_fails_closed (negative test - rejects non-allowlisted)
    assert settings.parallel_workers > 0
    await rest_client.wait_until_live()
    await rest_client.wait_until_model_ready(sum_model_settings.name)


async def test_concurrent_mlserver_start_spawns_workers(
    tmp_path: str,
    settings: Settings,
    sum_model_settings: ModelSettings,
):
    # Start multiple MLServer instances concurrently to stress worker spawn
    # and trusted-runtime bootstrap application under parallel startup.
    instances: list[tuple[Popen, RESTClient]] = []
    used_ports: set[int] = set()

    try:
        for idx in range(2):
            instance_settings = settings.model_copy(deep=True)
            instance_ports = get_available_ports(3)
            if used_ports.intersection(instance_ports):
                raise AssertionError(
                    "Concurrent test allocated duplicate ports across instances."
                )
            used_ports.update(instance_ports)
            instance_settings.http_port = instance_ports[0]
            instance_settings.grpc_port = instance_ports[1]
            instance_settings.metrics_port = instance_ports[2]

            folder = os.path.join(str(tmp_path), f"instance-{idx}")
            os.makedirs(folder, exist_ok=True)
            case_sum_model(folder, instance_settings, sum_model_settings)

            process = _spawn_mlserver(folder)
            client = RESTClient(f"127.0.0.1:{instance_settings.http_port}")
            instances.append((process, client))

        await asyncio.gather(*[client.wait_until_live() for _, client in instances])
        await asyncio.gather(
            *[
                client.wait_until_model_ready(sum_model_settings.name)
                for _, client in instances
            ]
        )
    finally:
        await asyncio.gather(*[client.close() for _, client in instances])
        for process, _ in instances:
            _stop_mlserver(process)


def test_server_startup_aborts_with_corrupted_allowlist(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings, monkeypatch
):
    """
    Test that server.start() aborts when trusted-runtimes.json is corrupted.
    Verifies that system-level failures (corrupted artifact) cause immediate
    server shutdown rather than just logging errors.
    """
    # Create corrupted trusted-runtimes.json artifact
    corrupted_artifact = Path(str(tmp_path)) / "corrupted-runtimes.json"
    corrupted_artifact.write_text("{invalid-json", encoding="utf-8")

    # Override artifact path via environment variable
    monkeypatch.setenv(TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV, str(corrupted_artifact))

    # Create model folder
    folder = case_sum_model(tmp_path, settings, sum_model_settings)

    # Spawn MLServer (should fail to start)
    process = _spawn_mlserver(folder)

    try:
        # Wait for process to exit (should fail fast during server.start())
        exit_code = process.wait(timeout=10)
        # Server should exit with non-zero code due to corrupted allowlist
        assert (
            exit_code != 0
        ), "Server should have failed to start with corrupted allowlist"
    except TimeoutExpired:
        _stop_mlserver(process)
        pytest.fail("Server did not exit within timeout - should have failed fast")
    finally:
        _stop_mlserver(process)


async def test_server_marks_startup_complete_after_successful_load(
    settings: Settings,
    sum_model_settings: ModelSettings,
    rest_client: RESTClient,
    prometheus_registry,
):
    """
    Test that server.start() has completed startup loading
    after successful load at startup. This ensures the health check stops
    reporting not ready because of the inital startup loading sequence
    once it completes.
    """
    from mlserver import MLServer

    server = MLServer(settings)

    # Start server with models
    server_task = asyncio.create_task(server.start([sum_model_settings]))

    try:
        # Wait for model to be ready via REST API
        await rest_client.wait_until_model_ready(sum_model_settings.name)

        # Should be marked complete after successful load
        assert server._data_plane.is_startup_complete

        # Verify server task is still running
        assert not server_task.done(), "Server task completed unexpectedly"
    finally:
        # Cleanup - let exceptions propagate to catch cleanup regressions
        await server.stop()
        await server_task


async def test_server_keeps_startup_incomplete_after_load_failure(
    settings: Settings,
    load_error_model_settings: ModelSettings,
    prometheus_registry,
    mocker,
):
    """
    Test that server.start() does NOT complete startup loading when model
    load fails. This ensures health check returns False during shutdown
    after startup failure, preventing false positive health checks during the
    race condition window.
    """
    from mlserver import MLServer
    from mlserver.errors import MLServerError, ModelNotFound

    server = MLServer(settings)
    transports_released = asyncio.Event()

    async def transport():
        await transports_released.wait()

    async def stop_transport(*args, **kwargs):
        return None

    original_stop = server.stop

    async def stop(*args, **kwargs):
        transports_released.set()
        await original_stop(*args, **kwargs)

    mocker.patch.object(server._rest_server, "start", new=transport)
    mocker.patch.object(server._rest_server, "stop", new=stop_transport)
    mocker.patch.object(server._grpc_server, "start", new=transport)
    mocker.patch.object(server._grpc_server, "stop", new=stop_transport)
    if server._metrics_server:
        mocker.patch.object(server._metrics_server, "start", new=transport)
        mocker.patch.object(server._metrics_server, "stop", new=stop_transport)
    mocker.patch.object(server, "stop", new=stop)

    # Start server with failing model - triggers shutdown
    server_task = asyncio.create_task(server.start([load_error_model_settings]))

    # The original model-load error must be preserved through shutdown.
    with pytest.raises(MLServerError, match="something really bad happened"):
        await server_task

    # Verify task actually completed (not hanging)
    assert server_task.done(), "Server startup task did not complete"

    # Verify the model was removed from registry on failure
    with pytest.raises(ModelNotFound):
        await server._model_registry.get_model(load_error_model_settings.name)

    # Should still be False (startup failed)
    assert not server._data_plane.is_startup_complete
    assert not server._startup_model_tasks
    assert not server._server_tasks
    assert not server._live
    assert server._stop_task is None


async def test_server_startup_cancellation_waits_for_model_load_cleanup(
    settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry,
    mocker,
):
    from mlserver import MLServer

    server = MLServer(settings)
    load_started = asyncio.Event()
    load_cancelled = asyncio.Event()
    release_load = asyncio.Event()
    load_settled = asyncio.Event()
    transports_released = asyncio.Event()

    async def load(_model_settings: ModelSettings):
        load_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            load_cancelled.set()
            await release_load.wait()
            load_settled.set()

    async def transport():
        await transports_released.wait()

    async def stop_transport(*args, **kwargs):
        return None

    original_stop = server.stop

    async def stop(*args, **kwargs):
        transports_released.set()
        await original_stop(*args, **kwargs)

    mocker.patch.object(server._model_registry, "load", new=load)
    mocker.patch.object(server._rest_server, "start", new=transport)
    mocker.patch.object(server._rest_server, "stop", new=stop_transport)
    mocker.patch.object(server._grpc_server, "start", new=transport)
    mocker.patch.object(server._grpc_server, "stop", new=stop_transport)
    if server._metrics_server:
        mocker.patch.object(server._metrics_server, "start", new=transport)
        mocker.patch.object(server._metrics_server, "stop", new=stop_transport)
    mocker.patch.object(server, "stop", new=stop)

    startup_task = asyncio.create_task(server.start([sum_model_settings]))
    await load_started.wait()
    startup_task.cancel()
    await load_cancelled.wait()
    await asyncio.sleep(0)
    assert not startup_task.done()
    release_load.set()

    with pytest.raises(asyncio.CancelledError):
        await startup_task

    assert load_settled.is_set()
    assert transports_released.is_set()
    assert not server._startup_model_tasks
    assert not server._server_tasks
    assert not server._live
    assert server._stop_task is None


async def test_server_startup_preserves_primary_model_error(
    settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry,
    mocker,
):
    from mlserver import MLServer

    server = MLServer(settings)
    transports_released = asyncio.Event()
    primary_error = RuntimeError("model startup failed")
    secondary_error = RuntimeError("transport failed")

    async def load(_model_settings: ModelSettings):
        raise primary_error

    async def failing_transport():
        await transports_released.wait()
        raise secondary_error

    async def transport():
        await transports_released.wait()

    async def stop_transport(*args, **kwargs):
        return None

    original_stop = server.stop

    async def stop(*args, **kwargs):
        transports_released.set()
        await original_stop(*args, **kwargs)

    mocker.patch.object(server._model_registry, "load", new=load)
    mocker.patch.object(server._rest_server, "start", new=failing_transport)
    mocker.patch.object(server._rest_server, "stop", new=stop_transport)
    mocker.patch.object(server._grpc_server, "start", new=transport)
    mocker.patch.object(server._grpc_server, "stop", new=stop_transport)
    if server._metrics_server:
        mocker.patch.object(server._metrics_server, "start", new=transport)
        mocker.patch.object(server._metrics_server, "stop", new=stop_transport)
    mocker.patch.object(server, "stop", new=stop)

    with pytest.raises(RuntimeError) as error:
        await server.start([sum_model_settings])

    assert error.value is primary_error
    assert error.value is not secondary_error
    assert transports_released.is_set()


async def test_server_startup_preserves_primary_error_when_stop_is_cancelled(
    settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry,
    mocker,
):
    from mlserver import MLServer

    server = MLServer(settings)
    primary_error = RuntimeError("model startup failed")

    async def load(_model_settings: ModelSettings):
        raise primary_error

    async def transport():
        return None

    async def cancelled_stop(*args, **kwargs):
        raise asyncio.CancelledError()

    original_stop = server.stop
    mocker.patch.object(server._model_registry, "load", new=load)
    mocker.patch.object(server._rest_server, "start", new=transport)
    mocker.patch.object(server._grpc_server, "start", new=transport)
    if server._metrics_server:
        mocker.patch.object(server._metrics_server, "start", new=transport)
    mocker.patch.object(server, "stop", new=cancelled_stop)

    try:
        with pytest.raises(RuntimeError) as error:
            await server.start([sum_model_settings])

        assert error.value is primary_error
    finally:
        await original_stop()


async def test_server_startup_cancels_transport_tasks_when_resource_cleanup_fails(
    settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry,
    mocker,
):
    from mlserver import MLServer

    server = MLServer(settings)
    transport_count = 2 + int(server._metrics_server is not None)
    transports_started = asyncio.Event()
    transports_cancelled = asyncio.Event()
    started_count = 0
    cancelled_count = 0
    primary_error = RuntimeError("model startup failed")

    async def load(_model_settings: ModelSettings):
        await transports_started.wait()
        raise primary_error

    async def blocked_transport():
        nonlocal started_count, cancelled_count
        started_count += 1
        if started_count == transport_count:
            transports_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled_count += 1
            if cancelled_count == transport_count:
                transports_cancelled.set()
            raise

    mocker.patch.object(server._model_registry, "load", new=load)
    mocker.patch.object(server._rest_server, "start", new=blocked_transport)
    mocker.patch.object(server._grpc_server, "start", new=blocked_transport)
    if server._metrics_server:
        mocker.patch.object(server._metrics_server, "start", new=blocked_transport)

    async def failing_stop_resources(*args, **kwargs):
        raise RuntimeError("resource cleanup failed")

    original_stop_resources = server._stop_resources
    mocker.patch.object(server, "_stop_resources", new=failing_stop_resources)

    try:
        with pytest.raises(RuntimeError) as error:
            await asyncio.wait_for(server.start([sum_model_settings]), timeout=1)

        assert transports_started.is_set()
        assert transports_cancelled.is_set()
        assert error.value is primary_error
    finally:
        mocker.patch.object(server, "_stop_resources", new=original_stop_resources)
        await server.stop()


async def test_server_startup_cleans_up_blocked_transport_after_sibling_failure(
    settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry,
    mocker,
):
    from mlserver import MLServer

    server = MLServer(settings)
    assert server._inference_pool_registry is not None
    original_registry_close = server._inference_pool_registry.close
    registry_close = mocker.patch.object(
        server._inference_pool_registry,
        "close",
        new=mocker.AsyncMock(),
    )
    transport_count = 2 + int(server._metrics_server is not None)
    transports_started = asyncio.Event()
    transport_failed = asyncio.Event()
    transports_released = asyncio.Event()
    transports_settled = asyncio.Event()
    started_count = 0
    settled_count = 0
    transport_error = RuntimeError("transport failed")

    async def load(_model_settings: ModelSettings):
        await transport_failed.wait()

    async def mark_started():
        nonlocal started_count
        started_count += 1
        if started_count == transport_count:
            transports_started.set()

    async def failing_transport():
        nonlocal settled_count
        await mark_started()
        await transports_started.wait()
        transport_failed.set()
        try:
            raise transport_error
        finally:
            settled_count += 1
            if settled_count == transport_count:
                transports_settled.set()

    async def blocked_transport():
        nonlocal settled_count
        await mark_started()
        try:
            await transports_released.wait()
        finally:
            settled_count += 1
            if settled_count == transport_count:
                transports_settled.set()

    async def stop_transport(*args, **kwargs):
        transports_released.set()

    mocker.patch.object(server._model_registry, "load", new=load)
    mocker.patch.object(server._rest_server, "start", new=failing_transport)
    mocker.patch.object(server._rest_server, "stop", new=stop_transport)
    mocker.patch.object(server._grpc_server, "start", new=blocked_transport)
    mocker.patch.object(server._grpc_server, "stop", new=stop_transport)
    if server._metrics_server:
        mocker.patch.object(server._metrics_server, "start", new=blocked_transport)
        mocker.patch.object(server._metrics_server, "stop", new=stop_transport)
    try:
        with pytest.raises(RuntimeError) as error:
            await asyncio.wait_for(server.start([sum_model_settings]), timeout=1)

        assert transports_settled.is_set()
        assert error.value is transport_error
        registry_close.assert_awaited_once_with()
    finally:
        mocker.patch.object(
            server._inference_pool_registry,
            "close",
            new=original_registry_close,
        )
        await server.stop()


async def test_server_startup_with_no_models(
    settings: Settings,
    rest_client: RESTClient,
    prometheus_registry,
):
    """
    Test that server completes startup even with no models configured.
    This tests the edge case where an empty model list should still
    complete startup successfully and mark startup as complete.
    """
    from mlserver import MLServer

    server = MLServer(settings)
    server_task = asyncio.create_task(server.start([]))  # Empty list

    try:
        # Wait for server to be live
        await rest_client.wait_until_live()

        # Startup should complete even with no models
        assert server._data_plane.is_startup_complete

        # Health check should use empty_registry_readiness setting
        is_ready = await rest_client.ready()
        assert is_ready == settings.empty_registry_readiness

        # Verify server task is still running
        assert not server_task.done(), "Server task completed unexpectedly"
    finally:
        # Cleanup
        await server.stop()
        await server_task
