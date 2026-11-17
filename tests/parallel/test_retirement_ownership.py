"""Failure composition must preserve responses and native retirement owners."""

import asyncio
import os
import signal

import pytest

from mlserver.parallel.pool import InferencePool
from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.parallel.worker import Worker
from mlserver.parallel.messages import ModelRequestMessage
from mlserver.parallel.errors import WorkerStop

from tests.fixtures import SumModel
from .test_lifecycle_recovery import wait_until


@pytest.fixture
def settings(settings):
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    settings.worker_start_timeout = 3
    settings.parallel_workers_timeout = 1
    return settings


@pytest.mark.parametrize("queue_cleanup_failure", [False, True])
async def test_sigchld_revocation_error_preserves_native_and_response_owners(
    settings, sum_model_settings, monkeypatch, queue_cleanup_failure
):
    # Purpose: Combine normal SIGCHLD handling with transport revocation error and
    # optional parent-queue retirement error.
    # Expect: The exact worker's response fails, replacement capacity returns, and
    # unresolved queue cleanup/error remains owned without disabling a healthy sibling.
    # Method: Establish real ZeroMQ traffic, schedule a controlled response waiter, kill
    # a child and observe the installed reaper before checking serving and native
    # retirement.
    pytest.importorskip("zmq")
    settings.parallel_inference_transport = "zmq"
    settings.parallel_workers = 2
    monkeypatch.setattr("mlserver.parallel.pool.random.uniform", lambda *_: 0)
    registry = InferencePoolRegistry(settings)
    pool = None
    future = None
    victim = None
    try:
        model = await registry.load_model(SumModel(sum_model_settings))
        pool = registry._default_pool
        victim = next(iter(pool._workers.values()))
        # Establish a real socket before injecting its revocation failure.
        assert (await model._send("metadata")).name == model.name
        dispatcher = pool._dispatcher
        message = ModelRequestMessage(model_name=model.name, method_name="predict")
        future = dispatcher._async_responses._schedule(message, victim)
        stopped = asyncio.Event()
        original_forget = dispatcher._transport.forget_worker
        original_stop = pool.on_worker_stop
        failures = []

        def forget(worker, exit_code=-1):
            original_forget(worker, exit_code)
            if worker is victim:
                raise OSError("request socket retirement failed")

        async def stop(pid, exit_code):
            try:
                return await original_stop(pid, exit_code)
            except BaseException as error:
                failures.append(error)
                raise
            finally:
                if pid == victim.pid:
                    stopped.set()

        monkeypatch.setattr(dispatcher._transport, "forget_worker", forget)
        monkeypatch.setattr(pool, "on_worker_stop", stop)
        queue_error = OSError("queue release did not complete")

        async def fail_retirement(timeout):
            raise queue_error

        if queue_cleanup_failure:
            monkeypatch.setattr(victim, "retire_queues", fail_retirement)
        victim.kill()
        # Exercise the installed registry SIGCHLD reaper, not manual waitpid.
        await asyncio.wait_for(stopped.wait(), 5)
        await asyncio.sleep(0)
        owned = (
            victim in pool._workers.values()
            or victim in pool._starting_workers.values()
            or victim in pool._unstarted_workers
            or victim in pool._closing_workers.values()
        )
        print(
            "SIGCHLD_FAILURE_STATE",
            dict(
                exitcode=victim.exitcode,
                response_done=future.done(),
                queues_retired=victim.queues_retired,
                queue_readers_closed=[q._reader.closed for q in victim._queues],
                worker_retained=owned,
                restart_owners=len(pool._restart_tasks),
                errors=[repr(error) for error in failures],
            ),
        )
        assert victim.queues_retired or owned, "reaped worker lost before queue cleanup"
        assert (
            future.done()
        ), "response completion skipped after transport cleanup error"
        with pytest.raises(WorkerStop):
            future.result()
        assert (await model._send("metadata")).name == model.name
        await asyncio.wait_for(
            wait_until(lambda: len(pool._dispatcher._ready_workers) == 2), 10
        )
        assert len(failures) == 1
        assert pool._cleanup_error is failures[0]
        if queue_cleanup_failure:
            assert pool._closing_workers[id(victim)] is victim
            assert not victim.queues_retired
            assert getattr(failures[0], "cleanup_error") is queue_error
            with pytest.raises(OSError) as close_error:
                await registry.close()
            with pytest.raises(OSError) as repeated:
                await registry.close()
            assert repeated.value is close_error.value
        else:
            assert victim.queues_retired
            assert all(q._reader.closed and q._writer.closed for q in victim._queues)
    finally:
        monkeypatch.undo()
        if future is not None:
            future.cancel()
            if not future.cancelled():
                future.exception()
            pool._dispatcher._async_responses._clear_message(message.id)
        if registry._close_task is not None and registry._close_task.exception():
            with pytest.raises(OSError):
                await registry.close()
            # Test-only resource cleanup after observing terminal close failure.
            await pool._close()
        else:
            await registry.close()
        # The failed implementation drops this worker from all pool maps, so
        # reviewer teardown must retire its real parent queues explicitly.
        if victim is not None:
            if victim.exitcode is None:
                victim.kill()
                await asyncio.to_thread(victim.join, 5)
            await victim.retire_queues(2)


async def test_revocation_failure_settles_actual_dispatched_request(
    settings, sum_model_settings, monkeypatch
):
    # Purpose: Exercise a request already sent to a real worker when revocation cleanup
    # raises after its death.
    # Expect: The request completes with WorkerStop without caller cancellation being
    # needed to make it finish.
    # Method: Warm its ZeroMQ socket, SIGSTOP the peer, observe response-wait entry
    # after send, kill it and inspect completion after the installed SIGCHLD handler.
    pytest.importorskip("zmq")
    settings.parallel_inference_transport = "zmq"
    registry = InferencePoolRegistry(settings)
    request = None
    victim = None
    try:
        model = await registry.load_model(SumModel(sum_model_settings))
        pool = registry._default_pool
        victim = next(iter(pool._workers.values()))
        assert (await model._send("metadata")).name == model.name
        dispatcher = pool._dispatcher
        waiting, stopped = asyncio.Event(), asyncio.Event()
        original_wait = dispatcher._async_responses._wait
        original_forget = dispatcher._transport.forget_worker
        original_stop = pool.on_worker_stop

        async def wait(message_id):
            waiting.set()
            return await original_wait(message_id)

        def forget(worker, exit_code=-1):
            original_forget(worker, exit_code)
            raise OSError("request socket retirement failed")

        async def stop(pid, code):
            try:
                return await original_stop(pid, code)
            finally:
                if pid == victim.pid:
                    stopped.set()

        monkeypatch.setattr(dispatcher._async_responses, "_wait", wait)
        monkeypatch.setattr(dispatcher._transport, "forget_worker", forget)
        monkeypatch.setattr(pool, "on_worker_stop", stop)
        # Hold the actual peer so the request is sent on its established
        # socket but cannot complete before the native crash/reaper boundary.
        os.kill(victim.pid, signal.SIGSTOP)
        request = asyncio.create_task(model._send("metadata"))
        await asyncio.wait_for(waiting.wait(), 5)
        victim.kill()
        await asyncio.wait_for(stopped.wait(), 5)
        # This watchdog is not used to cancel the request or declare success.
        done, _ = await asyncio.wait({request}, timeout=0.1)
        print(
            "ACTUAL_REQUEST_AFTER_REVOKE_FAILURE",
            dict(done=bool(done), exitcode=victim.exitcode),
        )
        assert (
            request in done
        ), "real dispatched request still waiting after confirmed worker exit"
        from mlserver.parallel.errors import WorkerStop

        with pytest.raises(WorkerStop):
            request.result()
    finally:
        monkeypatch.undo()
        if request is not None:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        if victim is not None and victim.exitcode is None:
            victim.kill()
        await registry.close()
        if victim is not None:
            await asyncio.to_thread(victim.join, 5)
            await victim.retire_queues(2)


@pytest.mark.parametrize("cancelled", [False, True])
async def test_replay_failure_plus_revocation_error_keeps_child_owned(
    settings, sum_model_settings, monkeypatch, cancelled
):
    # Purpose: Combine failed/cancelled replacement replay with failed transport
    # revocation.
    # Expect: The initiating failure survives with its cleanup error; the real candidate
    # cannot lose its retirement owner or affect existing serving capacity.
    # Method: Spawn a candidate, inject replay and forget_worker errors, then inspect
    # confirmed exit, native queues, ownership and the surviving model's metadata.
    pytest.importorskip("zmq")
    settings.parallel_inference_transport = "zmq"
    pool = InferencePool(settings)
    await pool.start()
    model = await pool.load_model(SumModel(sum_model_settings))
    candidates = []
    original_start = Worker.start
    original_forget = pool._dispatcher._transport.forget_worker
    primary = (
        asyncio.CancelledError() if cancelled else OSError("replay preparation failed")
    )
    cleanup = OSError("replay revocation failed")

    def start(worker):
        original_start(worker)
        candidates.append(worker)

    async def replay(worker):
        raise primary

    def forget(worker, code=-1):
        original_forget(worker, code)
        if worker in candidates:
            raise cleanup

    monkeypatch.setattr(Worker, "start", start)
    monkeypatch.setattr(pool, "_replay_worker", replay)
    monkeypatch.setattr(pool._dispatcher._transport, "forget_worker", forget)
    try:
        with pytest.raises(type(primary)) as observed:
            await pool._start_worker()
        candidate = candidates[0]
        owned = (
            candidate in pool._workers.values()
            or candidate in pool._starting_workers.values()
            or candidate in pool._closing_workers.values()
            or candidate in pool._unstarted_workers
        )
        print(
            "REPLAY_REVOKE_FAILURE_STATE",
            dict(
                exitcode=candidate.exitcode,
                queues_retired=candidate.queues_retired,
                worker_retained=owned,
                primary_preserved=observed.value is primary,
                observed_error=repr(observed.value),
            ),
        )
        assert (
            candidate.queues_retired or owned
        ), "replay rollback lost a live worker owner"
        assert observed.value is primary
        assert getattr(primary, "cleanup_error") is cleanup
        assert pool._cleanup_error is cleanup
        assert candidate.exitcode is not None and candidate.queues_retired
        assert (await model._send("metadata")).name == model.name
    finally:
        monkeypatch.undo()
        await pool.close()
        for candidate in candidates:
            if candidate.exitcode is None:
                candidate.kill()
            await asyncio.to_thread(candidate.join, 5)
            await candidate.retire_queues(2)
