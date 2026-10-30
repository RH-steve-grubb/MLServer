import pytest
import asyncio
import platform

from unittest.mock import patch

from mlserver.utils import (
    defer_cancellation,
    get_model_uri,
    extract_headers,
    insert_headers,
    get_normalized_version,
    install_uvloop_event_loop,
    with_operation_lock,
)
from mlserver.version import __version__
from mlserver.types import InferenceRequest, InferenceResponse, Parameters
from mlserver.settings import ModelSettings, ModelParameters
from .fixtures import SumModel

test_get_model_uri_paramaters = [
    ("s3://bucket/key", None, "s3://bucket/key"),
    ("s3://bucket/key", "/mnt/models/model-settings.json", "s3://bucket/key"),
]
for scheme in ["", "file:"]:
    for uri, source, expected in [
        ("my-model.bin", None, "my-model.bin"),
        (
            "my-model.bin",
            "./my-model-folder/model-settings.json",
            "my-model-folder/my-model.bin",
        ),
        (
            "my-model.bin",
            "./my-model-folder/../model-settings.json",
            "my-model.bin",
        ),
        (
            "/an/absolute/path/my-model.bin",
            "/mnt/models/model-settings.json",
            "/an/absolute/path/my-model.bin",
        ),
    ]:
        test_get_model_uri_paramaters.append((scheme + uri, source, expected))


@pytest.mark.parametrize(
    "uri, source, expected",
    test_get_model_uri_paramaters,
)
async def test_get_model_uri(uri: str, source: str | None, expected: str):
    model_settings = ModelSettings(
        name="test-model", implementation=SumModel, parameters=ModelParameters(uri=uri)
    )
    model_settings._source = source
    with patch("os.path.isfile", return_value=True):
        model_uri = await get_model_uri(model_settings)

    assert model_uri == expected


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        Parameters(),
        Parameters(headers={"foo": "bar2"}),
        Parameters(headers={"bar": "foo"}),
    ],
)
def test_insert_headers(parameters: Parameters):
    inference_request = InferenceRequest(inputs=[], parameters=parameters)
    headers = {"foo": "bar", "hello": "world"}
    insert_headers(inference_request, headers)

    assert inference_request.parameters is not None
    assert inference_request.parameters.headers == headers


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (None, None),
        (Parameters(), None),
        (Parameters(headers={}), {}),
        (Parameters(headers={"foo": "bar"}), {"foo": "bar"}),
    ],
)
def test_extract_headers(parameters: Parameters, expected: dict[str, str]):
    inference_response = InferenceResponse(
        model_name="foo", outputs=[], parameters=parameters
    )
    headers = extract_headers(inference_response)

    assert headers == expected
    if inference_response.parameters:
        assert inference_response.parameters.headers is None


def _check_uvloop_availability():
    avail = True
    try:
        import uvloop  # noqa: F401
    except ImportError:  # pragma: no cover
        avail = False
    return avail


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.7.1+rhaiv.8", "1.7.1"),
        ("1.7.1", "1.7.1"),
        ("1.7.0.dev0", "1.7.0.dev0"),
    ],
)
def test_get_normalized_version(version: str | None, expected: str):
    assert get_normalized_version(version) == expected


def test_get_normalized_version_default_uses_current_version():
    assert get_normalized_version() == __version__.split("+", 1)[0]


def test_uvloop_auto_install():
    uvloop_available = _check_uvloop_availability()
    install_uvloop_event_loop()
    policy = asyncio.get_event_loop_policy()

    if uvloop_available:
        assert type(policy).__module__.startswith("uvloop")
    else:
        if platform.system() == "Windows":
            assert isinstance(policy, asyncio.WindowsProactorEventLoopPolicy)
        elif platform.python_implementation() != "CPython":
            assert isinstance(policy, asyncio.DefaultEventLoopPolicy)


async def test_defer_cancellation_waits_for_operation_to_settle():
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()

    async def operation() -> None:
        started.set()
        await finish.wait()
        completed.set()

    task = asyncio.create_task(defer_cancellation()(operation)())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    finish.set()

    await task

    assert completed.is_set()
    if hasattr(task, "cancelling"):
        assert task.cancelling() == 0


async def test_defer_cancellation_defers_repeated_cancellation():
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()

    async def operation() -> None:
        started.set()
        await finish.wait()
        completed.set()

    task = asyncio.create_task(defer_cancellation()(operation)())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    finish.set()

    await task

    assert completed.is_set()
    if hasattr(task, "cancelling"):
        assert task.cancelling() == 0


async def test_defer_cancellation_clears_coalesced_requests():
    started = asyncio.Event()
    finish = asyncio.Event()

    async def operation() -> None:
        started.set()
        await finish.wait()

    task = asyncio.create_task(defer_cancellation()(operation)())
    await started.wait()
    task.cancel()
    task.cancel()
    finish.set()

    await task

    if hasattr(task, "cancelling"):
        assert task.cancelling() == 0


async def test_defer_cancellation_propagates_failure_after_cancellation():
    started = asyncio.Event()
    fail = asyncio.Event()

    async def operation() -> None:
        started.set()
        await fail.wait()
        raise RuntimeError("operation failed")

    task = asyncio.create_task(defer_cancellation()(operation)())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    fail.set()

    with pytest.raises(RuntimeError, match="operation failed"):
        await task


async def test_defer_cancellation_returns_operation_result():
    async def operation() -> str:
        return "complete"

    result = await defer_cancellation()(operation)()

    assert result == "complete"


async def test_defer_cancellation_propagates_failure_without_cancellation():
    error = RuntimeError("operation failed")

    async def operation() -> None:
        raise error

    with pytest.raises(RuntimeError, match="operation failed"):
        await defer_cancellation()(operation)()


async def test_defer_cancellation_timeout_cancels_operation():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @defer_cancellation(timeout=0.01)
    async def operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(operation())
    await started.wait()

    with pytest.raises(asyncio.TimeoutError):
        await task

    assert cancelled.is_set()


async def test_defer_cancellation_resolves_callable_timeout():
    @defer_cancellation(lambda timeout: timeout)
    async def operation(timeout: float) -> None:
        await asyncio.sleep(timeout * 2)

    with pytest.raises(asyncio.TimeoutError):
        await operation(0.01)


async def test_defer_cancellation_timeout_takes_precedence_over_cancellation():
    started = asyncio.Event()

    @defer_cancellation(timeout=0.01)
    async def operation() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(operation())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.TimeoutError):
        await task


async def test_defer_cancellation_outer_timeout_cancels_nested_operation():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    def unexpected_inner_timeout(*args, **kwargs) -> float:
        raise AssertionError("nested timeout was resolved")

    @defer_cancellation(timeout=unexpected_inner_timeout)
    async def inner_operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    @defer_cancellation(timeout=0.01)
    async def outer_operation() -> None:
        await inner_operation()

    task = asyncio.create_task(outer_operation())
    await started.wait()

    with pytest.raises(asyncio.TimeoutError):
        await task

    assert cancelled.is_set()


async def test_defer_cancellation_inner_timeout_is_ignored_for_nested_operation():
    @defer_cancellation(timeout=0.01)
    async def inner_operation() -> None:
        await asyncio.sleep(0.02)

    @defer_cancellation(timeout=0.1)
    async def outer_operation() -> str:
        await inner_operation()
        await asyncio.sleep(0.005)
        return "complete"

    assert await outer_operation() == "complete"


async def test_defer_cancellation_deadlines_are_isolated_between_tasks():
    @defer_cancellation(timeout=0.01)
    async def short_operation() -> None:
        await asyncio.sleep(1)

    @defer_cancellation(timeout=0.1)
    async def long_operation() -> str:
        await asyncio.sleep(0.02)
        return "complete"

    short_task, long_task = await asyncio.gather(
        short_operation(),
        long_operation(),
        return_exceptions=True,
    )

    assert isinstance(short_task, asyncio.TimeoutError)
    assert long_task == "complete"


async def test_defer_cancellation_resets_nested_operation_state():
    second_started = asyncio.Event()
    release_second = asyncio.Event()

    @defer_cancellation(timeout=0.01)
    async def first_operation() -> None:
        await asyncio.Event().wait()

    @defer_cancellation(timeout=10.0)
    async def second_operation() -> None:
        second_started.set()
        await release_second.wait()

    # A timed-out top-level operation must restore the caller's context.
    with pytest.raises(asyncio.TimeoutError):
        await first_operation()

    # The new task inherits the reset context and gets its own cancellation
    # boundary. Cancellation remains deferred until the operation settles.
    task = asyncio.create_task(second_operation())
    await second_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release_second.set()
    await task


async def test_with_operation_lock_serializes_same_lock_operations():
    lock = asyncio.Lock()
    active = 0
    maximum_active = 0

    @with_operation_lock(lambda: lock)
    async def operation() -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1

    await asyncio.gather(operation(), operation())

    assert maximum_active == 1


async def test_with_operation_lock_allows_independent_locks_and_resolvers():
    locks = {"sync": asyncio.Lock(), "async": asyncio.Lock()}
    active = 0
    maximum_active = 0

    def resolve_sync_lock(key: str) -> asyncio.Lock:
        return locks[key]

    async def resolve_async_lock(key: str) -> asyncio.Lock:
        return locks[key]

    @with_operation_lock(resolve_sync_lock)
    async def operation_with_sync_resolver(key: str) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1

    @with_operation_lock(resolve_async_lock)
    async def operation_with_async_resolver(key: str) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1

    await asyncio.gather(
        operation_with_sync_resolver("sync"),
        operation_with_async_resolver("async"),
    )

    assert maximum_active == 2


async def test_composed_lock_and_deferred_cancellation_cancel_while_waiting():
    lock = asyncio.Lock()
    lock_resolved = asyncio.Event()
    entered = asyncio.Event()
    await lock.acquire()

    def lock_for() -> asyncio.Lock:
        lock_resolved.set()
        return lock

    @with_operation_lock(lock_for)
    @defer_cancellation()
    async def operation() -> None:
        entered.set()

    task = asyncio.create_task(operation())
    await lock_resolved.wait()
    task.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        lock.release()

    assert not entered.is_set()


async def test_composed_lock_and_deferred_cancellation_after_lock_acquisition():
    lock = asyncio.Lock()
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()

    @with_operation_lock(lambda: lock)
    @defer_cancellation()
    async def operation() -> None:
        started.set()
        await finish.wait()
        completed.set()

    task = asyncio.create_task(operation())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    finish.set()

    await task

    assert completed.is_set()
    assert not lock.locked()


async def test_deferred_timeout_starts_after_lock_acquisition():
    lock = asyncio.Lock()
    lock_waiting = asyncio.Event()
    entered = asyncio.Event()
    finish = asyncio.Event()
    await lock.acquire()

    def lock_for() -> asyncio.Lock:
        lock_waiting.set()
        return lock

    @with_operation_lock(lock_for)
    @defer_cancellation(timeout=0.01)
    async def operation() -> None:
        entered.set()
        await finish.wait()

    task = asyncio.create_task(operation())
    await lock_waiting.wait()
    assert not entered.is_set()

    lock.release()
    await entered.wait()

    with pytest.raises(asyncio.TimeoutError):
        await task

    assert not lock.locked()
