import asyncio
from collections.abc import Iterator
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
from mlserver.parallel import zmq as transport  # noqa: E402
from mlserver.parallel.errors import WorkerError  # noqa: E402
from mlserver.parallel.messages import (  # noqa: E402
    ModelRequestMessage,
    ModelResponseMessage,
)
from mlserver.parallel.worker import Worker  # noqa: E402


@pytest.fixture
def directory() -> Iterator[Path]:
    # Pytest's long test-specific directories can exceed Unix socket limits.
    with tempfile.TemporaryDirectory(prefix="mls-zmq-") as path:
        yield Path(path)


def worker(directory: Path, reply: str) -> Mock:
    return Mock(
        spec=Worker,
        pid=42,
        _zmq_request_endpoint=transport.create_endpoint(directory, "q"),
        _zmq_reply_endpoint=reply,
    )


def request_message() -> ModelRequestMessage:
    return ModelRequestMessage(model_name="selected", method_name="predict")


async def stop(task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)


def test_endpoints_are_unique_short_strings_and_reject_unsafe_paths(
    directory: Path,
) -> None:
    first = transport.create_endpoint(directory, "q")
    second = transport.create_endpoint(directory, "q")
    assert first != second and first.startswith("ipc://")
    assert len(first.removeprefix("ipc://").encode()) < 100
    assert ForkingPickler.loads(ForkingPickler.dumps({"endpoint": first})) == {
        "endpoint": first
    }
    for path, kind in (
        (Path("relative"), "q"),
        (directory / "missing", "q"),
        (directory, "../r"),
    ):
        with pytest.raises(ValueError):
            transport.create_endpoint(path, kind)
    long = directory / ("x" * 100)
    long.mkdir()
    with pytest.raises(ValueError, match="shorter"):
        transport.create_endpoint(long, "q")


@pytest.mark.parametrize("error", [False, True])
async def test_real_ipc_preserves_request_response_arrays_bytes_and_errors(
    directory: Path, error: bool
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    resolved = asyncio.get_running_loop().create_future()
    failed = Mock()
    parent.start(resolved.set_result, failed)
    target = worker(directory, endpoint)
    bits = np.array([0x80000000, 0x7FC00001, 1], dtype="<u4")
    message = ModelRequestMessage(
        model_name="chosen-model",
        model_version="v2",
        method_name="predict",
        method_args=[bits.view("<f4"), b"request bytes"],
        method_kwargs={"selection": "retained"},
    )

    async def process(request: ModelRequestMessage) -> ModelResponseMessage:
        assert request.id == message.id
        assert (request.model_name, request.model_version) == ("chosen-model", "v2")
        assert request.method_name == "predict"
        np.testing.assert_array_equal(request.method_args[0].view("<u4"), bits)
        assert request.method_args[1] == b"request bytes"
        assert request.method_kwargs == {"selection": "retained"}
        return ModelResponseMessage(
            id=request.id,
            return_value=None if error else (request.method_args[0], b"reply bytes"),
            exception=WorkerError(ValueError("model failed")) if error else None,
        )

    target._process_request = AsyncMock(side_effect=process)
    task = asyncio.create_task(transport.serve_requests(target))
    try:
        await asyncio.wait_for(parent.send(target, message), timeout=2)
        response = await asyncio.wait_for(resolved, timeout=2)
        assert response.id == message.id
        if error:
            assert isinstance(response.exception, WorkerError)
            assert str(response.exception) == "builtins.ValueError: model failed"
        else:
            np.testing.assert_array_equal(response.return_value[0].view("<u4"), bits)
            assert response.return_value[1] == b"reply bytes"
        failed.assert_not_called()
    finally:
        await parent.close()
        await stop(task)


async def test_wire_uses_queues_default_pickle_and_does_not_modify_worker_state(
    directory: Path,
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    target = worker(directory, endpoint)
    context = zmq.asyncio.Context()
    peer = context.socket(zmq.PULL)
    peer.bind(getattr(target, "_zmq_request_endpoint"))
    message = ModelRequestMessage(
        model_name="selected", method_name="predict", method_args=[b"pixels"]
    )
    state = dict(target.__dict__)
    parent.start(Mock(), Mock())
    try:
        await asyncio.wait_for(parent.send(target, message), timeout=2)
        wire = await asyncio.wait_for(peer.recv(), timeout=2)
        assert wire == bytes(ForkingPickler.dumps(message))
        assert target.__dict__ == state
        sock = parent._requests[42]
        assert sock.getsockopt(zmq.SNDHWM) == sock.getsockopt(zmq.RCVHWM) == 0
        assert sock.getsockopt(zmq.IMMEDIATE) == 1
        assert sock.getsockopt(zmq.LINGER) == 0
    finally:
        await parent.close()
        peer.close(linger=0)
        context.term()


async def test_async_model_requests_can_overlap(directory: Path) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    target = worker(directory, endpoint)
    entered: set[str] = set()
    both, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    responses = []

    async def process(request: ModelRequestMessage) -> ModelResponseMessage:
        entered.add(request.id)
        if len(entered) == 2:
            both.set()
        await release.wait()
        return ModelResponseMessage(id=request.id, return_value="done")

    def resolve(response: ModelResponseMessage) -> None:
        responses.append(response)
        if len(responses) == 2:
            completed.set()

    target._process_request = AsyncMock(side_effect=process)
    parent.start(resolve, Mock())
    task = asyncio.create_task(transport.serve_requests(target))
    messages = [
        ModelRequestMessage(model_name="async-model", method_name="predict")
        for _ in range(2)
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(parent.send(target, message) for message in messages)), 2
        )
        # A serial worker would block on the first request before reaching two.
        await asyncio.wait_for(both.wait(), timeout=2)
        release.set()
        await asyncio.wait_for(completed.wait(), timeout=2)
        assert {response.id for response in responses} == {
            message.id for message in messages
        }
    finally:
        await parent.close()
        await stop(task)


async def test_send_from_another_threads_loop_uses_owner_transport(
    directory: Path,
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    resolved = asyncio.get_running_loop().create_future()
    parent.start(resolved.set_result, Mock())
    target = worker(directory, endpoint)
    target._process_request = AsyncMock(
        side_effect=lambda request: ModelResponseMessage(id=request.id)
    )
    task = asyncio.create_task(transport.serve_requests(target))
    message = request_message()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(lambda: asyncio.run(parent.send(target, message))),
            timeout=2,
        )
        response = await asyncio.wait_for(resolved, timeout=2)
        assert response.id == message.id
    finally:
        await parent.close()
        await stop(task)


async def test_cross_loop_send_cancellation_reaches_owner_operation(
    directory: Path,
) -> None:
    parent = transport.ParentTransport(transport.create_endpoint(directory, "r"))
    parent.start(Mock(), Mock())
    cancelled = asyncio.Event()

    async def pending(worker: Worker, message: ModelRequestMessage) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def caller() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                parent.send(worker(directory, "unused"), request_message()), 0.05
            )

    try:
        with patch.object(parent, "_send", side_effect=pending):
            await asyncio.wait_for(asyncio.to_thread(lambda: asyncio.run(caller())), 2)
            await asyncio.wait_for(cancelled.wait(), 2)
    finally:
        await parent.close()


async def test_receiver_failure_reports_once_and_rejects_future_sends(
    directory: Path,
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    failure = asyncio.get_running_loop().create_future()
    resolved = Mock()
    parent.start(resolved, failure.set_result)
    context = zmq.asyncio.Context()
    peer = context.socket(zmq.PUSH)
    peer.connect(endpoint)
    try:
        await peer.send(ForkingPickler.dumps("wrong response"))
        error = await asyncio.wait_for(failure, timeout=2)
        assert isinstance(error, TypeError)
        resolved.assert_not_called()
        with pytest.raises(RuntimeError, match="unavailable"):
            await parent.send(worker(directory, endpoint), request_message())
    finally:
        await parent.close()
        peer.close(linger=0)
        context.term()


@pytest.mark.parametrize("failure", ["worker-stop", "receiver-failure", "close"])
async def test_transport_failures_release_send_waiting_for_worker_connection(
    directory: Path, failure: str
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    reported = asyncio.get_running_loop().create_future()
    parent.start(Mock(), reported.set_result)
    # No worker binds its endpoint. IMMEDIATE must keep this send pending.
    target = worker(directory, endpoint)
    sending = asyncio.create_task(parent.send(target, request_message()))
    context = zmq.asyncio.Context()
    peer = context.socket(zmq.PUSH)
    try:
        await asyncio.sleep(0.03)
        assert not sending.done()
        if failure == "worker-stop":
            parent.forget_worker(42)
        elif failure == "receiver-failure":
            peer.connect(endpoint)
            await peer.send(ForkingPickler.dumps("invalid response"))
            assert isinstance(await asyncio.wait_for(reported, 2), TypeError)
        else:
            await parent.close()
        (result,) = await asyncio.wait_for(
            asyncio.gather(sending, return_exceptions=True), timeout=2
        )
        assert isinstance(result, BaseException)
        assert not parent._requests
    finally:
        await parent.close()
        await stop(sending)
        peer.close(linger=0)
        context.term()


@pytest.mark.parametrize("bad_request", [True, False])
async def test_worker_protocol_failure_terminates_request_loop(
    directory: Path, bad_request: bool
) -> None:
    target = worker(directory, transport.create_endpoint(directory, "r"))
    target._process_request = AsyncMock(return_value="wrong worker result")
    context = zmq.asyncio.Context()
    peer = context.socket(zmq.PUSH)
    peer.connect(getattr(target, "_zmq_request_endpoint"))
    task = asyncio.create_task(transport.serve_requests(target))
    try:
        message = "wrong request" if bad_request else request_message()
        await peer.send(ForkingPickler.dumps(message))
        with pytest.raises(TypeError, match="request|response"):
            await asyncio.wait_for(task, timeout=2)
    finally:
        await stop(task)
        peer.close(linger=0)
        context.term()


async def test_shutdown_cancels_inflight_requests_even_if_cancellation_is_wrapped(
    directory: Path,
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    parent.start(Mock(), Mock())
    target = worker(directory, endpoint)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def process(request: ModelRequestMessage) -> ModelResponseMessage:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            cancelled.set()
            # The stock worker converts cancellation to a response envelope.
            return ModelResponseMessage(id=request.id, exception=WorkerError(error))
        raise AssertionError("Unreleased request unexpectedly completed")

    target._process_request = AsyncMock(side_effect=process)
    task = asyncio.create_task(transport.serve_requests(target))
    try:
        await asyncio.wait_for(parent.send(target, request_message()), timeout=2)
        await asyncio.wait_for(entered.wait(), timeout=2)
        await parent.close()
        await stop(task)
        assert cancelled.is_set()
    finally:
        await parent.close()
        await stop(task)


async def test_worker_replacement_forgets_socket_and_reconnects(
    directory: Path,
) -> None:
    endpoint = transport.create_endpoint(directory, "r")
    parent = transport.ParentTransport(endpoint)
    received: asyncio.Queue = asyncio.Queue()
    parent.start(received.put_nowait, Mock())
    tasks = []
    try:
        previous = None
        for _ in range(2):
            target = worker(directory, endpoint)
            target._process_request = AsyncMock(
                side_effect=lambda request: ModelResponseMessage(id=request.id)
            )
            tasks.append(asyncio.create_task(transport.serve_requests(target)))
            await asyncio.wait_for(parent.send(target, request_message()), timeout=2)
            await asyncio.wait_for(received.get(), timeout=2)
            sock = parent._requests[42]
            assert sock is not previous
            parent.forget_worker(42)
            assert sock.closed and 42 not in parent._requests
            parent.forget_worker(42)
            previous = sock
    finally:
        await parent.close()
        for task in tasks:
            await stop(task)


async def test_parent_constructor_delays_receiver_and_close_releases_descriptors(
    directory: Path,
) -> None:
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(4):
        parent = transport.ParentTransport(transport.create_endpoint(directory, "r"))
        assert parent._receiver is None
        with pytest.raises(RuntimeError, match="unavailable"):
            await parent.send(worker(directory, "unused"), request_message())
        parent.start(Mock(), Mock())
        receiver = parent._receiver
        parent.start(Mock(), Mock())
        assert parent._receiver is receiver
        await parent.close()
        await parent.close()
        assert parent._context.closed and parent._replies.closed
        with pytest.raises(RuntimeError, match="unavailable"):
            parent.start(Mock(), Mock())
    assert len(list(Path("/proc/self/fd").iterdir())) <= before


def test_parent_bind_failure_closes_sockets_and_context(directory: Path) -> None:
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises(zmq.ZMQError):
        transport.ParentTransport(f"ipc://{directory / 'missing' / 'reply.sock'}")
    assert len(list(Path("/proc/self/fd").iterdir())) <= before


def test_parent_socket_creation_failure_releases_context() -> None:
    context = Mock()
    with patch.object(
        transport.zmq.asyncio, "Context", return_value=context
    ), patch.object(transport, "_socket", side_effect=RuntimeError("socket failed")):
        with pytest.raises(RuntimeError, match="socket failed"):
            transport.ParentTransport("unused")
    context.term.assert_called_once()


async def test_worker_partial_socket_setup_is_cleaned(directory: Path) -> None:
    context, requests = Mock(), Mock()
    with patch.object(
        transport.zmq.asyncio, "Context", return_value=context
    ), patch.object(
        transport, "_socket", side_effect=[requests, RuntimeError("socket failed")]
    ):
        with pytest.raises(RuntimeError, match="socket failed"):
            await transport.serve_requests(worker(directory, "unused"))
    requests.close.assert_called_once()
    context.term.assert_called_once()


def test_socket_option_failure_closes_partial_socket() -> None:
    context, sock = Mock(), Mock()
    context.socket.return_value = sock
    sock.setsockopt.side_effect = RuntimeError("option failed")
    with pytest.raises(RuntimeError, match="option failed"):
        transport._socket(context, zmq.PUSH)
    sock.close.assert_called_once_with(linger=0)
