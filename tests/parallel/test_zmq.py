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


async def test_failed_socket_retirement_keeps_owner_and_attempts_other_sockets(
    directory, monkeypatch
):
    # Purpose: Exercise native request-socket close failure during worker revocation and
    # transport close.
    # Expect: The worker stays revoked, the failed socket/context remain owned, other
    # sockets close and repeated public close reports the same error.
    # Method: Register two real sockets, fault one close, attempt an old send and
    # inspect actual closed flags plus retained socket identity before test-only
    # teardown.
    parent = transport.ParentTransport(transport.create_endpoint(directory, "r"))
    parent.start(lambda _: None, lambda _: None)
    workers = [worker(directory, parent._reply_endpoint) for _ in range(2)]
    sockets = [transport._socket(parent._context, zmq.PUSH, set()) for _ in workers]
    parent._requests.update(zip(workers, sockets))
    cleanup = OSError("socket release did not complete")

    def fail():
        raise cleanup

    monkeypatch.setattr(sockets[0], "close", fail)
    try:
        with pytest.raises(OSError) as observed:
            parent.forget_worker(workers[0], 0)
        assert observed.value is cleanup
        assert parent._requests[workers[0]] is sockets[0]
        assert not sockets[0].closed
        from mlserver.parallel.errors import WorkerStop

        with pytest.raises(WorkerStop):
            await parent.send(workers[0], request_message())
        with pytest.raises(OSError) as observed:
            await parent.close()
        assert observed.value is cleanup
        assert sockets[1].closed
        assert not sockets[0].closed and not parent._context.closed
        assert parent._requests == {workers[0]: sockets[0]}
        with pytest.raises(OSError) as repeated:
            await parent.close()
        assert repeated.value is cleanup
    finally:
        monkeypatch.undo()
        # Also release the affected baseline's socket, which it no longer owns.
        for sock in sockets:
            sock.close()
        if parent._close_task is None:
            await parent.close()
        else:
            # Test-only cleanup after the recorded terminal close failure.
            await parent._close()


async def test_receive_failure_notifies_waiters_despite_socket_cleanup_failure(
    directory, monkeypatch
):
    # Purpose: Combine receive failure with failed close of a socket whose real native
    # send remains pending.
    # Expect: Failure notification still settles that send and retains the primary
    # receive error plus secondary cleanup error and socket owner.
    # Method: Observe send entry without a connected peer, inject a malformed reply
    # through a real socket and inspect waiter completion and the callback's error.
    failures = []
    parent = transport.ParentTransport(transport.create_endpoint(directory, "r"))
    parent.start(lambda _: None, failures.append)
    selected = worker(directory, parent._reply_endpoint)
    sock = transport._socket(parent._context, zmq.PUSH, set())
    sock.connect(selected._zmq_request_endpoint)
    parent._requests[selected] = sock
    peer = transport._socket(parent._context, zmq.PUSH, set())
    peer.connect(parent._reply_endpoint)
    cleanup = OSError("request socket cleanup failed")

    def fail():
        raise cleanup

    entered = asyncio.Event()
    original_send = sock.send

    def send(*args, **kwargs):
        result = original_send(*args, **kwargs)
        entered.set()
        return result

    monkeypatch.setattr(sock, "close", fail)
    monkeypatch.setattr(sock, "send", send)
    sending = asyncio.create_task(parent.send(selected, request_message()))
    try:
        # IMMEDIATE keeps this real send pending without a connected peer.
        # The callback proves entry before the malformed reply fails reception.
        await asyncio.wait_for(entered.wait(), 5)
        assert parent._pending_sends[selected]
        await peer.send(b"invalid pickle")
        await asyncio.wait_for(
            asyncio.gather(parent._receiver, return_exceptions=True), 5
        )
        assert failures == [parent._failure]
        assert getattr(parent._failure, "cleanup_error") is cleanup
        assert parent._requests[selected] is sock and not sock.closed
        done, _ = await asyncio.wait({sending}, timeout=1)
        assert sending in done, "socket cleanup failure stranded a pending send"
        failure = parent._failure
        assert isinstance(failure, Exception)
        with pytest.raises(type(failure)):
            sending.result()
    finally:
        sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
        monkeypatch.undo()
        peer.close()
        sock.close()
        await parent.close()


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
        sock = parent._requests[target]
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
            parent.forget_worker(target)
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
            sock = parent._requests[target]
            assert sock is not previous
            parent.forget_worker(target)
            assert sock.closed and target not in parent._requests
            parent.forget_worker(target)
            previous = sock
    finally:
        await parent.close()
        for task in tasks:
            await stop(task)


async def test_parent_constructor_delays_receiver_and_close_releases_descriptors(
    directory: Path,
) -> None:
    # Purpose: Check lightweight transport construction, explicit activation and
    # terminal repeated close.
    # Expect: Construction starts no receiver; repeated start shares it, close releases
    # native descriptors and a closed transport cannot restart.
    # Method: Repeat real context/socket activation and closure, compare live descriptor
    # counts and inspect receiver identity and actual closed flags.
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
        assert parent._context is not None
        assert parent._context.closed and parent._replies.closed
        with pytest.raises(RuntimeError, match="unavailable"):
            parent.start(Mock(), Mock())
    assert len(list(Path("/proc/self/fd").iterdir())) <= before


async def test_parent_bind_failure_closes_sockets_and_context(directory: Path) -> None:
    # Purpose: Exercise reply bind failure after native context/socket allocation.
    # Expect: The transport's retained owner must close partial setup without leaving
    # native descriptors.
    # Method: Bind under a missing IPC directory, close in finally and compare
    # descriptor counts before and after the real failure.
    before = len(list(Path("/proc/self/fd").iterdir()))
    parent = transport.ParentTransport(f"ipc://{directory / 'missing' / 'reply.sock'}")
    try:
        with pytest.raises(zmq.ZMQError):
            parent.start(Mock(), Mock())
    finally:
        await parent.close()
    assert len(list(Path("/proc/self/fd").iterdir())) <= before


async def test_parent_socket_creation_failure_releases_context() -> None:
    # Purpose: Exercise parent setup failing before a socket helper returns a handle.
    # Expect: The already retained context still receives its termination call.
    # Method: Use a context double and failing socket helper, then call public close and
    # inspect termination; native partial-handle behavior has separate probes.
    context = Mock()
    with patch.object(
        transport.zmq.asyncio, "Context", return_value=context
    ), patch.object(transport, "_socket", side_effect=RuntimeError("socket failed")):
        parent = transport.ParentTransport("unused")
        try:
            with pytest.raises(RuntimeError, match="socket failed"):
                parent.start(Mock(), Mock())
        finally:
            await parent.close()
    context.term.assert_called_once()


async def test_parent_close_waiter_cancellation_retains_native_cleanup(directory):
    # Purpose: Exercise caller cancellation while parent transport close waits for
    # receiver cleanup.
    # Expect: A repeated close joins the retained operation and eventually releases the
    # real reply socket/context.
    # Method: Replace the receiver with an event-held finally block, cancel only the
    # first waiter and release it after the second joins.
    parent = transport.ParentTransport(transport.create_endpoint(directory, "r"))
    parent.start(Mock(), Mock())
    parent._receiver.cancel()
    await asyncio.gather(parent._receiver, return_exceptions=True)
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def receiver():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    parent._receiver = asyncio.create_task(receiver())
    await entered.wait()
    first = asyncio.create_task(parent.close())
    repeated = None
    try:
        await asyncio.wait_for(cleaning.wait(), 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        repeated = asyncio.create_task(parent.close())
        await asyncio.sleep(0)
        assert not repeated.done()
        release.set()
        await asyncio.wait_for(repeated, 5)
        assert parent._context is not None and parent._context.closed
        assert parent._replies.closed
    finally:
        release.set()
        await asyncio.gather(
            first, *([repeated] if repeated else []), return_exceptions=True
        )
        await parent.close()


@pytest.mark.parametrize("term_fails", [False, True])
async def test_worker_partial_socket_setup_is_cleaned(
    directory: Path, term_fails
) -> None:
    # Purpose: Exercise worker reply-socket setup failure after request-socket
    # acquisition, including a context-termination error.
    # Expect: The initiating setup error stays primary, acquired sockets close and
    # failed context cleanup remains owned with its secondary error.
    # Method: Use ownership-aware socket/context doubles, fail the second acquisition
    # and optional term, then inspect close calls, error identity and the worker
    # resource owner.
    context, requests = Mock(), Mock()
    primary = RuntimeError("socket failed")
    cleanup = RuntimeError("context retirement failed")
    target = worker(directory, "unused")
    if term_fails:
        context.term.side_effect = cleanup

    def socket(context, kind, owned):
        if kind == zmq.PULL:
            owned.add(requests)
            return requests
        raise primary

    with patch.object(
        transport.zmq.asyncio, "Context", return_value=context
    ), patch.object(transport, "_socket", side_effect=socket):
        with pytest.raises(RuntimeError) as observed:
            await transport.serve_requests(target)
    assert observed.value is primary
    if term_fails:
        assert getattr(primary, "cleanup_error") is cleanup
        assert target._zmq_resources[0] is context
    else:
        assert target._zmq_resources is None
    requests.close.assert_called_once_with(linger=0)
    context.term.assert_called_once()


def test_socket_option_failure_closes_partial_socket() -> None:
    # Purpose: Exercise helper rollback when options fail on a returned socket.
    # Expect: The setup error must propagate after requesting non-lingering socket
    # close.
    # Method: Use a socket double whose setsockopt raises and verify close(linger=0);
    # the subprocess matrix establishes actual native release behavior.
    context, sock = Mock(), Mock()
    context.socket.return_value = sock
    sock.setsockopt.side_effect = RuntimeError("option failed")
    with pytest.raises(RuntimeError, match="option failed"):
        transport._socket(context, zmq.PUSH, set())
    sock.close.assert_called_once_with(linger=0)
