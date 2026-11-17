"""Native setup/rollback failures must not strand sockets in Context.term()."""

import asyncio
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("zmq")


async def socket_probe(case, directory, status):
    import zmq
    import zmq.asyncio
    from multiprocessing.reduction import ForkingPickler
    from mlserver.parallel import zmq as transport
    from mlserver.parallel.messages import ModelRequestMessage
    from mlserver.parallel.worker import Worker

    owner, boundary, mode = case
    loop = asyncio.get_running_loop()
    state = {
        "owner": owner,
        "boundary": boundary,
        "mode": mode,
        "loop_class": f"{type(loop).__module__}.{type(loop).__name__}",
    }

    def record(**values):
        state.update(values)
        status.write_text(json.dumps(state))

    primary, cleanup = OSError("socket setup failed"), OSError("socket rollback failed")
    original_context = zmq.asyncio.Context
    original_socket = original_context.socket
    contexts, native, acquired = [], [], []
    terms = {}
    inject = owner == "worker" or boundary == "reply-options"
    tearing_down = False
    close_calls = 0
    peer_context = original_context()
    peers = []

    def context(*args, **kwargs):
        value = original_context(*args, **kwargs)
        contexts.append(value)
        terms[value] = value.term

        def term():
            record(
                close_started=True,
                term_entered=True,
                term_open_sockets=[
                    not sock.closed for ctx, sock in native if ctx is value
                ],
            )
            # This is real native termination; only the parent process can
            # interrupt a stuck event loop. Capture its stack before the watchdog.
            faulthandler.dump_traceback_later(1)
            try:
                terms[value]()
            finally:
                faulthandler.cancel_dump_traceback_later()
                record(term_returned=True)

        value.term = term
        return value

    def socket(context, kind, *args, **kwargs):
        nonlocal close_calls
        value = original_socket(context, kind, *args, **kwargs)
        if context is peer_context:
            return value
        native.append((context, value))
        selected = (
            boundary in ("reply-options", "options")
            and kind == zmq.PULL
            or boundary in ("request-options", "connect")
            and kind == zmq.PUSH
        )
        if inject and selected and not acquired:
            acquired.append(value)
            original_close = value.close

            def fail(*args, **kwargs):
                raise primary

            def close(*args, **kwargs):
                nonlocal close_calls
                close_calls += 1
                if not tearing_down and (
                    mode == "persistent" or mode == "transient" and close_calls == 1
                ):
                    raise cleanup
                return original_close(*args, **kwargs)

            value.close = close
            if boundary.endswith("options"):
                value.setsockopt = fail
            else:
                value.connect = fail
        return value

    original_context.socket = socket
    zmq.asyncio.Context = context
    parent = None
    target = Mock(
        spec=Worker,
        pid=42,
        _zmq_request_endpoint=transport.create_endpoint(directory, "q"),
        _zmq_reply_endpoint=transport.create_endpoint(directory, "r"),
    )
    healthy = Mock(
        spec=Worker,
        pid=43,
        _zmq_request_endpoint=transport.create_endpoint(directory, "q"),
    )

    def peer(worker):
        value = peer_context.socket(zmq.PULL)
        value.setsockopt(zmq.LINGER, 0)
        value.bind(worker._zmq_request_endpoint)
        peers.append(value)
        return value

    async def deliver(worker, receiver, label):
        await parent.send(
            worker, ModelRequestMessage(model_name=label, method_name="metadata")
        )
        message = ForkingPickler.loads(await receiver.recv())
        assert message.model_name == label

    try:
        observed = None
        if owner == "parent":
            parent = transport.ParentTransport(
                transport.create_endpoint(directory, "r")
            )
            try:
                parent.start(lambda _: None, lambda _: None)
                first, second = peer(healthy), peer(target)
                await deliver(healthy, first, "healthy-before")
                healthy_socket = parent._requests[healthy]
                inject = True
                await parent.send(
                    target,
                    ModelRequestMessage(model_name="failed", method_name="metadata"),
                )
            except BaseException as error:
                observed = error
            assert acquired and observed is not None
            owned_before = (
                acquired[0] is parent._replies
                or acquired[0] in parent._requests.values()
                or acquired[0] in getattr(parent, "_unregistered_sockets", ())
            )
            if boundary != "reply-options":
                assert target not in parent._requests
                await deliver(healthy, first, "healthy-after")
                assert parent._requests[healthy] is healthy_socket
                # A retry must create a usable socket rather than reuse the
                # earlier unconnected handle retained for resource cleanup.
                await deliver(target, second, "retry")
                assert parent._requests[target] is not acquired[0]
            record(
                close_started=True,
                primary_preserved=observed is primary,
                owned_before=owned_before,
                acquired_closed=acquired[0].closed,
                close_calls=close_calls,
            )
            close_error = None
            try:
                await parent.close()
            except BaseException as error:
                close_error = error
            record(
                close_returned=True,
                close_error=repr(close_error),
                close_calls=close_calls,
            )
            if mode == "persistent":
                assert close_error is cleanup
                try:
                    await parent.close()
                except BaseException as repeated:
                    assert repeated is close_error
                else:
                    raise AssertionError("repeated close lost its recorded failure")
                assert not parent._context.closed and not acquired[0].closed
                assert acquired[0] in parent._unregistered_sockets
                assert not state.get("term_entered")
            else:
                assert close_error is None
                assert parent._context.closed and acquired[0].closed
                assert not getattr(parent, "_unregistered_sockets", ())
                assert not any(state["term_open_sockets"])
                await parent.close()
        else:
            try:
                await transport.serve_requests(target)
            except BaseException as error:
                observed = error
            assert acquired and observed is not None
            resources = getattr(target, "_zmq_resources", None)
            owned_before = resources is not None and acquired[0] in resources[1]
            record(
                close_returned=True,
                primary_preserved=observed is primary,
                owned_before=owned_before,
                acquired_closed=acquired[0].closed,
                close_calls=close_calls,
            )
            if acquired[0].closed:
                assert contexts[0].closed and resources is None
            else:
                assert owned_before and not contexts[0].closed
                assert not state.get("term_entered")
        assert observed is primary
        if mode != "success":
            assert getattr(primary, "cleanup_error") is cleanup
        if mode == "success":
            assert acquired[0].closed
        else:
            assert owned_before or acquired[0].closed
        record(passed=True)
    finally:
        # This is test-owned cleanup, including the affected baseline's lost
        # handles. A successful assertion must come from the production path.
        tearing_down = True
        for _, sock in native:
            sock.close(linger=0)
        for value in peers:
            value.close(linger=0)
        peer_context.term()
        if parent is not None and parent._receiver is not None:
            parent._receiver.cancel()
            await asyncio.gather(parent._receiver, return_exceptions=True)
        for value in contexts:
            terms[value]()
        original_context.socket = original_socket
        zmq.asyncio.Context = original_context


@pytest.mark.parametrize(
    "owner,boundary",
    [
        ("parent", "reply-options"),
        ("parent", "request-options"),
        ("parent", "connect"),
        ("worker", "options"),
        ("worker", "connect"),
    ],
)
@pytest.mark.parametrize("mode", ["success", "transient", "persistent"])
def test_socket_setup_retains_release_ownership(owner, boundary, mode, tmp_path):
    # Purpose: Exercise parent/worker socket options or connect failure with successful,
    # transiently failed or persistently failed rollback.
    # Expect: Incomplete native sockets remain owned outside usable routing;
    # primary/cleanup errors survive, retry is isolated and close releases resources or
    # returns an owned error.
    # Method: Run real contexts/sockets in a disposable subprocess, verify its loop
    # policy and recorded native state, and supervise close externally because a blocked
    # Context.term prevents asyncio watchdogs from running. Only test-owned children are
    # killed on failure.
    policy = os.environ.get("MLSERVER_TEST_SOCKET_LOOP_POLICY", "asyncio")
    folder = Path(os.environ.get("MLSERVER_TEST_SOCKET_EVIDENCE", str(tmp_path)))
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{policy}-{owner}-{boundary}-{mode}"
    status, log = folder / (stem + ".json"), folder / (stem + ".log")
    with tempfile.TemporaryDirectory(prefix="mls-acquire-") as directory:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "probe",
            json.dumps([owner, boundary, mode]),
            directory,
            str(status),
        ]
        with log.open("w") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                state = {}
                while time.monotonic() < deadline:
                    if status.exists():
                        try:
                            state = json.loads(status.read_text())
                        except json.JSONDecodeError:
                            continue
                        if state.get("close_started") or state.get("close_returned"):
                            break
                    if child.poll() is not None:
                        break
                    time.sleep(0.01)
                assert state, log.read_text()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pytest.fail(f"native close blocked: {state}\n{log.read_text()}")
                state = json.loads(status.read_text())
                assert child.returncode == 0, f"{state}\n{log.read_text()}"
                assert state.get("passed"), state
                expected = (
                    "uvloop.Loop"
                    if policy == "uvloop"
                    else "asyncio.unix_events._UnixSelectorEventLoop"
                )
                assert state["loop_class"] == expected
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    if os.environ.get("MLSERVER_TEST_SOCKET_LOOP_POLICY") == "uvloop":
        import uvloop

        uvloop.install()
    else:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(
        socket_probe(json.loads(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    )
