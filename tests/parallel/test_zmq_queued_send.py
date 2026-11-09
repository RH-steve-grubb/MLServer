"""PR #356: real sockets and a foreign caller loop, with simulated workers.

The barriers establish the ordering, not its frequency in native crashes.
"""

import asyncio
import gc
from concurrent.futures import Future
from multiprocessing import get_context
from pathlib import Path
import tempfile
from threading import Event, Thread
from unittest.mock import Mock
import weakref

import pytest

from mlserver.parallel.dispatcher import Dispatcher
from mlserver.parallel.errors import WorkerStop
from mlserver.parallel.messages import ModelRequestMessage
from mlserver.parallel.worker import Worker


async def wait_until(predicate):
    while not predicate():
        await asyncio.sleep(0)


@pytest.mark.parametrize("transport", ["queue", "zmq"])
@pytest.mark.parametrize("exit_code", [23, 0])
@pytest.mark.parametrize(
    "order", ["worker-only", "caller-only", "caller-first", "worker-first"]
)
async def test_stop_before_forwarded_send_starts(
    monkeypatch, transport, exit_code, order
):
    # Purpose: Exercise stop/cancellation orderings after foreign-loop request
    # submission but before the owner-loop ZeroMQ send begins.
    # Expect: Stop alone returns WorkerStop even for exit zero; genuine cancellation
    # remains cancellation, and no old socket, waiter or caller thread survives.
    # Method: Use Worker doubles, real IPC sockets and a foreign caller loop;
    # callback/thread barriers queue stop and cancellation before send resumes. The
    # Queue variants are compatibility controls.
    owner = asyncio.get_running_loop()
    responses = get_context("spawn").Queue()
    done, response_done = Event(), asyncio.Event()
    caller_paused, release_caller = Event(), Event()
    caller, outcomes, forwarded, response_futures, errors = {}, [], [], [], []
    cancellations = []
    previous_handler = owner.get_exception_handler()
    owner.set_exception_handler(lambda _, context: errors.append(context))
    with tempfile.TemporaryDirectory(prefix="mls-forward-") as directory:
        values = dict(spec=Worker, pid=42)
        if transport == "zmq":
            pytest.importorskip("zmq")
            from mlserver.parallel.zmq import create_endpoint

            endpoint = create_endpoint(Path(directory), "r")
            setattr(responses, "_zmq_reply_endpoint", endpoint)
            values["_zmq_request_endpoint"] = create_endpoint(Path(directory), "q")
        worker = Mock(**values)
        dispatcher = Dispatcher({42: worker}, responses)
        dispatcher.start()
        original_schedule = dispatcher._async_responses._schedule
        run = asyncio.run_coroutine_threadsafe

        def cancel_caller():
            cancellations.append(caller["task"].cancel())

        def pause_caller():
            caller_paused.set()
            assert release_caller.wait(5)

        def stop_and_cancel():
            try:
                if transport == "queue" and order in ("caller-first", "worker-first"):
                    # Queue sends do not have the forwarded-send handoff below.
                    # Hold the caller until both callbacks are queued: otherwise
                    # WorkerStop may finish its task before cancellation arrives.
                    caller["loop"].call_soon_threadsafe(pause_caller)
                    assert caller_paused.wait(5)
                if order in ("caller-only", "caller-first"):
                    caller["loop"].call_soon_threadsafe(cancel_caller)
                if order != "caller-only":
                    dispatcher.on_worker_stop(worker, exit_code)
                if order == "worker-first":
                    caller["loop"].call_soon_threadsafe(cancel_caller)
            finally:
                release_caller.set()

        def schedule(message, target):
            future = original_schedule(message, target)
            response_futures.append(future)
            future.add_done_callback(
                lambda _: owner.call_soon_threadsafe(response_done.set)
            )
            if transport == "queue":
                owner.call_soon_threadsafe(stop_and_cancel)
            return future

        def enqueue_before_stop(operation, loop):
            handoff = Future()

            def submit():
                try:
                    # Both actions happen in one owner callback. Submission
                    # queues _send; stop completes before its first task step.
                    submitted = run(operation, loop)
                    forwarded.append(submitted)
                    stop_and_cancel()
                    handoff.set_result(submitted)
                except BaseException as error:
                    handoff.set_exception(error)

            loop.call_soon_threadsafe(submit)
            return handoff.result(5)

        monkeypatch.setattr(dispatcher._async_responses, "_schedule", schedule)
        if transport == "zmq":
            monkeypatch.setattr(
                asyncio, "run_coroutine_threadsafe", enqueue_before_stop
            )

        async def invoke():
            caller.update(loop=asyncio.get_running_loop(), task=asyncio.current_task())
            caller["loop"].set_exception_handler(
                lambda _, context: errors.append(context)
            )
            try:
                result = await dispatcher.dispatch_request(
                    ModelRequestMessage(model_name="selected", method_name="predict")
                )
                outcomes.append(result)
            except BaseException as error:
                outcomes.append(error)

        def thread_main():
            try:
                asyncio.run(invoke())
            finally:
                done.set()

        thread = Thread(target=thread_main)
        thread.start()
        try:
            await asyncio.wait_for(response_done.wait(), 5)
            if order == "worker-only":
                # Inspect without retrieving the error: production must still
                # observe/consume it, and the caller must receive WorkerStop.
                assert isinstance(
                    getattr(response_futures[0], "_exception"), WorkerStop
                )
                if transport == "zmq":
                    await asyncio.wait_for(
                        wait_until(
                            lambda: forwarded[0].done()
                            or dispatcher._transport._requests
                        ),
                        5,
                    )
                    assert (
                        not dispatcher._transport._requests
                    ), "queued send recreated stopped worker socket"
            assert await asyncio.to_thread(
                done.wait, 5
            ), "caller remained stuck awaiting forwarded send"
            assert len(outcomes) == 1
            if order == "worker-only":
                assert isinstance(outcomes[0], WorkerStop)
                assert f"exit code {exit_code}" in str(outcomes[0])
            else:
                assert isinstance(outcomes[0], asyncio.CancelledError)
                assert cancellations == [True]
            await asyncio.wait_for(
                wait_until(lambda: not dispatcher._async_responses._futures), 5
            )
            if transport == "zmq":
                await asyncio.wait_for(
                    wait_until(lambda: not dispatcher._transport._pending_sends), 5
                )
            assert not dispatcher._async_responses._futures_map
            assert not any(dispatcher._async_responses._workers_map.values())
        finally:
            release_caller.set()
            if not done.is_set() and "loop" in caller:
                caller["loop"].call_soon_threadsafe(caller["task"].cancel)
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
            for submitted in forwarded:
                if not submitted.done():
                    submitted.cancel()
            await dispatcher.stop(timeout=2)
            responses.close()
            await asyncio.to_thread(responses.join_thread)
            owner.set_exception_handler(previous_handler)
            assert not errors


@pytest.mark.parametrize("exit_code", [23, 0])
async def test_cancellation_after_worker_error_completion_preserves_result(
    responses, monkeypatch, exit_code
):
    # Purpose: Check cancellation attempted only after a request has already completed
    # with WorkerStop.
    # Expect: The later cancel is rejected and cannot replace the completed error or
    # retain response bookkeeping.
    # Method: Observe response registration, report worker exit zero/nonzero, await the
    # error and then inspect cancel's return value and the stored result.
    worker = Mock(spec=Worker, pid=42)
    dispatcher = Dispatcher({42: worker}, responses)
    registered = asyncio.Event()
    original = dispatcher._async_responses._schedule

    def schedule(message, selected):
        future = original(message, selected)
        registered.set()
        return future

    monkeypatch.setattr(dispatcher._async_responses, "_schedule", schedule)
    caller = asyncio.create_task(
        dispatcher.dispatch_request(
            ModelRequestMessage(model_name="selected", method_name="predict")
        )
    )
    try:
        await asyncio.wait_for(registered.wait(), 5)
        dispatcher.on_worker_stop(worker, exit_code)
        with pytest.raises(WorkerStop, match=f"exit code {exit_code}"):
            await asyncio.wait_for(caller, 5)
        assert not caller.cancel()
        with pytest.raises(WorkerStop, match=f"exit code {exit_code}"):
            caller.result()
        assert not dispatcher._async_responses._futures
        assert not dispatcher._async_responses._futures_map
        assert not any(dispatcher._async_responses._workers_map.values())
    finally:
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        await dispatcher.stop(2)


async def test_old_identity_cannot_send_or_close_replacement_with_same_pid():
    # Purpose: Exercise distinct worker incarnations sharing a numeric PID with real
    # ZeroMQ sockets.
    # Expect: Old sends fail with the retained exit-zero status while duplicate/stale
    # stop cannot close the replacement's working socket.
    # Method: Run the production child socket coroutine on a Worker double, stop a
    # blocked old send and exchange actual replies with the replacement before and after
    # stale cleanup.
    pytest.importorskip("zmq")
    from mlserver.parallel.zmq import ParentTransport, create_endpoint, serve_requests
    from mlserver.parallel.messages import ModelResponseMessage
    from unittest.mock import AsyncMock

    with tempfile.TemporaryDirectory(prefix="mls-incarnation-") as directory:
        path = Path(directory)
        endpoint = create_endpoint(path, "r")
        parent = ParentTransport(endpoint)
        responses = asyncio.Queue()
        parent.start(responses.put_nowait, lambda error: None)
        old, replacement = (
            Mock(
                spec=Worker,
                pid=42,
                _zmq_request_endpoint=create_endpoint(path, "q"),
                _zmq_reply_endpoint=endpoint,
            )
            for _ in range(2)
        )
        replacement._process_request = AsyncMock(
            side_effect=lambda message: ModelResponseMessage(
                id=message.id, return_value="replacement"
            )
        )
        serving = asyncio.create_task(serve_requests(replacement))
        pending = asyncio.create_task(
            parent.send(
                old, ModelRequestMessage(model_name="old", method_name="predict")
            )
        )
        try:
            await asyncio.wait_for(wait_until(lambda: bool(parent._pending_sends)), 5)
            parent.forget_worker(old, 0)
            parent.forget_worker(old)  # generic cleanup cannot erase exit zero
            message = ModelRequestMessage(
                model_name="replacement", method_name="predict"
            )
            await asyncio.wait_for(parent.send(replacement, message), 5)
            with pytest.raises(WorkerStop, match="exit code 0"):
                await asyncio.wait_for(pending, 5)
            response = await asyncio.wait_for(responses.get(), 5)
            assert response.id == message.id and response.return_value == "replacement"
            parent.forget_worker(old, -1)
            with pytest.raises(WorkerStop, match="exit code 0"):
                await asyncio.wait_for(parent.send(old, message), 5)
            await asyncio.wait_for(parent.send(replacement, message), 5)
            assert (
                await asyncio.wait_for(responses.get(), 5)
            ).return_value == "replacement"
            assert old not in parent._requests
            assert not parent._pending_sends
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await parent.close()
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


async def test_delayed_old_stop_preserves_replacement_routing_and_response():
    # Purpose: Exercise stale worker-stop delivery after a same-PID replacement is
    # selected for routing.
    # Expect: Only the old incarnation's waiter fails; the replacement remains routable
    # and receives its own response.
    # Method: Schedule old/new message UUIDs on distinct Worker doubles, install
    # replacement routing and deliver old stop followed by the new response.
    from mlserver.parallel.messages import ModelResponseMessage

    responses = get_context("spawn").Queue()
    old, replacement = Mock(spec=Worker, pid=42), Mock(spec=Worker, pid=42)
    dispatcher = Dispatcher({42: old}, responses)
    pending = dispatcher._async_responses
    before = ModelRequestMessage(model_name="old", method_name="predict")
    after = ModelRequestMessage(model_name="new", method_name="predict")
    pending._schedule(before, old)
    before_wait = asyncio.create_task(pending._wait(before.id))
    pending._schedule(after, replacement)
    after_wait = asyncio.create_task(pending._wait(after.id))
    dispatcher._workers[42] = replacement
    dispatcher.on_worker_ready(replacement)
    try:
        dispatcher.on_worker_stop(old, 0)
        with pytest.raises(WorkerStop):
            await asyncio.wait_for(before_wait, 5)
        assert not after_wait.done()
        assert dispatcher._get_worker()[0] is replacement
        pending.resolve(ModelResponseMessage(id=after.id, return_value="new"))
        assert (await asyncio.wait_for(after_wait, 5)).return_value == "new"
        assert not any(pending._workers_map.values()) and not pending._futures_map
    finally:
        before_wait.cancel()
        after_wait.cancel()
        await asyncio.gather(before_wait, after_wait, return_exceptions=True)
        await dispatcher.stop(2)
        responses.close()
        await asyncio.to_thread(responses.join_thread)


async def test_stopped_identity_bookkeeping_expires_after_owned_work_settles():
    # Purpose: Check that completed request and revocation bookkeeping does not
    # permanently retain old worker identities.
    # Expect: Weak worker records disappear after settled requests and the last strong
    # reference are released.
    # Method: Cycle 100 Worker doubles through response completion and stop, collect
    # them and inspect weak references plus response/socket maps.
    pytest.importorskip("zmq")
    from mlserver.parallel.dispatcher import AsyncResponses
    from mlserver.parallel.messages import ModelResponseMessage
    from mlserver.parallel.zmq import ParentTransport, create_endpoint

    with tempfile.TemporaryDirectory(prefix="mls-revoke-") as directory:
        parent = ParentTransport(create_endpoint(Path(directory), "r"))
        parent.start(lambda response: None, lambda error: None)
        refs = []
        responses = AsyncResponses()
        try:
            for _ in range(100):
                worker = Mock(spec=Worker, pid=42)
                refs.append(weakref.ref(worker))
                message = ModelRequestMessage(model_name="churn", method_name="predict")
                future = responses._schedule(message, worker)
                future.set_result(ModelResponseMessage(id=message.id))
                responses._clear_message(message.id)
                parent.forget_worker(worker, 0)
            del worker
            gc.collect()
            assert all(ref() is None for ref in refs)
            assert not parent._stopped_workers
            assert not parent._requests
            assert not parent._pending_sends
            assert not responses._workers_map
            assert not responses._futures_map and not responses._futures
        finally:
            await parent.close()


@pytest.mark.parametrize("failure", ["close", "receiver"])
async def test_queued_cross_loop_send_observes_transport_failure(monkeypatch, failure):
    # Purpose: Exercise transport close or receive failure before a queued foreign-loop
    # send takes its first step.
    # Expect: The caller receives the transport error, not cancellation, and queued
    # sends, response waiters and caller threads settle.
    # Method: Intercept forwarding to finish real close or inject a malformed native
    # reply before send begins; inspect completion and captured loop errors.
    zmq = pytest.importorskip("zmq")
    from mlserver.parallel.zmq import create_endpoint

    owner = asyncio.get_running_loop()
    responses = get_context("spawn").Queue()
    outcomes, forwarded, errors = [], [], []
    completed = Event()
    previous_handler = owner.get_exception_handler()
    owner.set_exception_handler(lambda _, context: errors.append(context))
    with tempfile.TemporaryDirectory(prefix="mls-queued-close-") as directory:
        endpoint = create_endpoint(Path(directory), "r")
        setattr(responses, "_zmq_reply_endpoint", endpoint)
        worker = Mock(
            spec=Worker,
            pid=42,
            _zmq_request_endpoint=create_endpoint(Path(directory), "q"),
        )
        dispatcher = Dispatcher({42: worker}, responses)
        dispatcher.start()
        parent = dispatcher._transport
        failed = asyncio.Event()
        original_failed = parent._receive
        # A real malformed reply will drive the production receive failure.
        original_fail_all = dispatcher._async_responses.fail_all

        def fail_all(error):
            original_fail_all(error)
            failed.set()

        # start() bound the original callback; replace the receiver before it
        # takes its first step, retaining the production receive implementation.
        parent._receiver.cancel()
        await asyncio.gather(parent._receiver, return_exceptions=True)
        parent._receiver = asyncio.create_task(
            original_failed(dispatcher._async_responses.resolve, fail_all)
        )
        peer = parent._context.socket(zmq.PUSH)
        peer.setsockopt(zmq.LINGER, 0)
        peer.connect(endpoint)
        submit = asyncio.run_coroutine_threadsafe

        async def settle_then_send(operation):
            try:
                if failure == "close":
                    # External peer must close before term() joins the context.
                    peer.close()
                    await parent.close()
                else:
                    await peer.send(b"invalid pickle")
                    await failed.wait()
                await operation
            finally:
                operation.close()

        def enqueue(operation, loop):
            result = submit(settle_then_send(operation), loop)
            forwarded.append(result)
            return result

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", enqueue)

        def caller():
            async def invoke():
                try:
                    await dispatcher.dispatch_request(
                        ModelRequestMessage(
                            model_name="selected", method_name="predict"
                        )
                    )
                except BaseException as error:
                    outcomes.append(error)

            try:
                asyncio.run(invoke())
            finally:
                completed.set()

        thread = Thread(target=caller)
        thread.start()
        try:
            assert await asyncio.to_thread(completed.wait, 5)
            assert isinstance(outcomes[0], RuntimeError)
            assert not isinstance(outcomes[0], asyncio.CancelledError)
            assert not parent._requests and not parent._pending_sends
            assert not dispatcher._async_responses._futures
            assert all(future.done() for future in forwarded)
        finally:
            for future in forwarded:
                future.cancel()
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
            peer.close()
            await dispatcher.stop(2)
            responses.close()
            await asyncio.to_thread(responses.join_thread)
            owner.set_exception_handler(previous_handler)
            assert not errors


async def test_real_sigchld_invalidates_the_selected_worker_instance(
    settings, sum_model, monkeypatch
):
    # Purpose: Check integration between real child death and transport incarnation
    # revocation.
    # Expect: The installed reaper passes the exact Worker and confirmed status into
    # invalidation, and replacement serving succeeds.
    # Method: Observe forget_worker on a real one-worker ZeroMQ registry, kill the
    # selected child and await replacement routing before querying worker metadata.
    pytest.importorskip("zmq")
    from mlserver.parallel.registry import InferencePoolRegistry

    settings.parallel_inference_transport = "zmq"
    settings.parallel_workers = 1
    registry = InferencePoolRegistry(settings)
    observed, called = [], asyncio.Event()
    try:
        model = await registry.load_model(sum_model)
        pool = registry._default_pool
        worker = next(iter(pool._workers.values()))
        transport = pool._dispatcher._transport
        forget = transport.forget_worker

        def invalidate(selected, exit_code=-1):
            observed.append((selected, exit_code))
            forget(selected, exit_code)
            called.set()

        monkeypatch.setattr(transport, "forget_worker", invalidate)
        worker.kill()
        await asyncio.wait_for(called.wait(), 5)
        assert observed[0] == (worker, -9)
        await asyncio.wait_for(
            wait_until(lambda: bool(pool._dispatcher._ready_workers)), 10
        )
        assert (await model._send("metadata")).name == model.name
        assert worker.pid not in pool._dispatcher._ready_workers
    finally:
        await registry.close()


async def test_failed_queue_retirement_cannot_swallow_replacement_exit(
    settings, monkeypatch
):
    # Purpose: Exercise same-PID replacement exit while an older incarnation remains
    # retained for failed queue cleanup.
    # Expect: The replacement's exit status and revocation cannot be consumed as another
    # notification for the old cleanup owner.
    # Method: Use two Worker doubles with one numeric PID, fail old retirement, install
    # replacement routing and deliver its distinct exit status.
    from mlserver.parallel.pool import InferencePool

    settings.parallel_workers = 0
    pool = InferencePool(settings)
    await pool.start()
    old = Mock(spec=Worker, pid=42, exitcode=0, queues_retired=False)
    replacement = Mock(spec=Worker, pid=42, exitcode=None, queues_retired=False)
    replacement.record_exit.side_effect = lambda code: setattr(
        replacement, "exitcode", code
    )
    old.retire_queues.side_effect = OSError("retirement not confirmed")

    async def pending_restart():
        await asyncio.Event().wait()

    monkeypatch.setattr(pool, "_safe_start_worker", pending_restart)
    try:
        pool._workers[42] = old
        pool._dispatcher.on_worker_ready(old)
        await pool.on_worker_stop(42, 0)
        assert old in pool._closing_workers.values()
        pool._workers[42] = replacement
        pool._dispatcher.on_worker_ready(replacement)
        await pool.on_worker_stop(42, 23)
        assert replacement.exitcode == 23
        assert 42 not in pool._dispatcher._ready_workers
        assert old in pool._closing_workers.values()
    finally:
        old.retire_queues.side_effect = None
        replacement.exitcode = 23
        await pool.close()


async def test_revocation_cleanup_failure_still_confirms_all_native_exits(
    settings, monkeypatch
):
    # Purpose: Exercise explicit pool close with a request-socket revocation error for
    # one of two real workers.
    # Expect: Independent child/queue cleanup continues for both workers, routing ends
    # and repeated close preserves the initiating error.
    # Method: Wrap real forget_worker to raise once after revocation, close the pool and
    # inspect actual exit codes, queue endpoints/feeders and the repeated result.
    pytest.importorskip("zmq")
    from mlserver.parallel.pool import InferencePool
    from .test_startup_ownership import assert_retired

    settings.parallel_inference_transport = "zmq"
    settings.parallel_workers = 2
    settings.parallel_workers_timeout = 1
    pool = InferencePool(settings)
    await pool.start()
    workers = list(pool._workers.values())
    transport = pool._dispatcher._transport
    original = transport.forget_worker
    failure = OSError("request socket close failed")
    calls = []

    def forget(*args):
        original(*args)
        calls.append(args)
        if len(calls) == 1:
            raise failure

    monkeypatch.setattr(transport, "forget_worker", forget)
    try:
        with pytest.raises(OSError) as observed:
            await pool.close()
        assert observed.value is failure
        assert all(worker.exitcode is not None for worker in workers)
        for worker in workers:
            assert_retired(worker)
        assert not pool._dispatcher._ready_workers
        with pytest.raises(OSError) as repeated:
            await pool.close()
        assert repeated.value is failure
    finally:
        monkeypatch.undo()
        await pool._close()


async def test_response_group_survives_foreign_schedule_during_last_cleanup():
    # Purpose: Exercise foreign-loop response registration while cleanup observes the
    # previous worker group as empty.
    # Expect: A subsequent stop still finds and fails the newly registered response
    # without stranding its caller thread.
    # Method: Use an instrumented set and thread events to force scheduling across the
    # empty observation, then report stop and inspect completion on the caller loop.
    from mlserver.parallel.dispatcher import AsyncResponses
    from mlserver.parallel.messages import ModelResponseMessage

    pending = AsyncResponses()
    worker = Mock(spec=Worker, pid=42)
    before = ModelRequestMessage(model_name="before", method_name="predict")
    after = ModelRequestMessage(model_name="after", method_name="predict")
    older = pending._schedule(before, worker)
    older.set_result(ModelResponseMessage(id=before.id))
    entered, release, scheduled, checked, done = (Event() for _ in range(5))
    caller, outcomes, observations = {}, [], []

    class InterleavedGroup(set):
        def __bool__(self):
            nonempty = len(self) != 0
            if not nonempty and not scheduled.is_set():
                # Force the foreign schedule after an empty observation and
                # before a conditional deletion. Event waits are watchdogs;
                # this observed transition determines the ordering.
                release.set()
                assert scheduled.wait(5)
            return nonempty

    pending._workers_map[worker] = InterleavedGroup({before.id})

    def run():
        async def invoke():
            caller.update(loop=asyncio.get_running_loop(), task=asyncio.current_task())
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)
            caller["future"] = pending._schedule(after, worker)
            scheduled.set()
            try:
                await pending._wait(after.id)
            except BaseException as error:
                outcomes.append(error)

        try:
            asyncio.run(invoke())
        finally:
            done.set()

    def inspect_stop():
        observations.append(caller["future"].done())
        checked.set()

    thread = Thread(target=run)
    thread.start()
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        pending._clear_message(before.id)
        release.set()  # A retained weak group needs no conditional deletion.
        assert await asyncio.to_thread(scheduled.wait, 5)
        pending.cancel(worker, 0)
        caller["loop"].call_soon_threadsafe(inspect_stop)
        assert await asyncio.to_thread(checked.wait, 5)
        assert observations == [True], "worker stop lost the newly scheduled response"
        assert await asyncio.to_thread(done.wait, 5)
        assert len(outcomes) == 1 and isinstance(outcomes[0], WorkerStop)
        assert not pending._futures and not pending._futures_map
        assert not any(pending._workers_map.values())
    finally:
        release.set()
        if not done.is_set() and "loop" in caller:
            caller["loop"].call_soon_threadsafe(caller["task"].cancel)
        await asyncio.to_thread(thread.join, 5)
        assert not thread.is_alive()
