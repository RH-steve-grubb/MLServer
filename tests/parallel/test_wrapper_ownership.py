"""Worker acknowledgements do not publish a candidate whose wrapper failed."""

import asyncio

import pytest

from mlserver.parallel.errors import WorkerError
from mlserver.parallel.pool import InferencePool
from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.errors import ModelLoadError, ModelUnloadError
from mlserver.registry import MultiModelRegistry
from tests.fixtures import SumModel


def fail_candidate_wrapper(pool, monkeypatch, *, persistent=False):
    original = pool._wrap
    failed = False

    def wrap(model):
        nonlocal failed
        if model.settings.platform == "abandoned" and (persistent or not failed):
            failed = True
            raise OSError("frontend wrapper construction failed")
        return original(model)

    monkeypatch.setattr(pool, "_wrap", wrap)
    return original


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_wrapper_failure_preserves_frontend_and_replay(
    settings, sum_model_settings, monkeypatch, cleanup_fails
):
    # Purpose: Exercise frontend wrapper failure after simulated worker preparation
    # acknowledgements, including failed rollback wrapping.
    # Expect: The old frontend identity and committed replay settings survive;
    # initiating and cleanup errors remain observable.
    # Method: Use a zero-worker pool with controlled acknowledgements, fault candidate
    # wrapping and compare frontend/replay state before restoring cleanup.
    settings.parallel_workers = 0
    pool = InferencePool(settings)
    await pool.start()

    async def acknowledge(message):
        pass

    monkeypatch.setattr(pool._dispatcher, "dispatch_update", acknowledge)
    frontend = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
    )
    settings_before = sum_model_settings.model_copy(deep=True)
    settings_before.platform = "committed"
    original_wrap = pool._wrap
    try:
        old = await frontend.load(settings_before)
        fail_candidate_wrapper(pool, monkeypatch, persistent=cleanup_fails)
        candidate = settings_before.model_copy(deep=True)
        candidate.platform = "abandoned"
        expected = ModelLoadError if cleanup_fails else OSError
        with pytest.raises(expected) as failure:
            await frontend.load(candidate)
        if cleanup_fails:
            assert isinstance(failure.value.__cause__, OSError)
            assert isinstance(failure.value.cleanup_error, ModelUnloadError)
            assert isinstance(failure.value.cleanup_error.__cause__, OSError)
        assert await frontend.get_model(old.name, old.version) is old
        assert old.ready
        assert pool.has_model(settings_before)
        assert [model.platform for model in pool._worker_registry.models] == [
            "committed"
        ]
    finally:
        monkeypatch.setattr(pool, "_wrap", original_wrap)
        await frontend.unload(sum_model_settings.name)
        await pool.close()


@pytest.mark.parametrize("transport", ["queue", "zmq"])
@pytest.mark.parametrize("replacement", [False, True])
async def test_native_wrapper_failure_cannot_publish_through_retry_or_unload(
    settings, sum_model_settings, monkeypatch, transport, replacement
):
    # Purpose: Exercise post-acknowledgement wrapper failure with a real Queue/ZeroMQ
    # worker for fresh and replacement loads.
    # Expect: Acknowledged rollback excludes the abandoned candidate from later unload,
    # replay and retry publication.
    # Method: Mark committed/abandoned/retried settings, fault wrapping, await recovery
    # and query actual worker metadata across unload or retry.
    if transport == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_inference_transport = transport
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    pools = InferencePoolRegistry(settings)
    await pools.start()
    pool = pools._default_pool
    before = sum_model_settings.model_copy(deep=True)
    before.platform = "committed"
    try:
        old = await pools.load_model(SumModel(before)) if replacement else None
        fail_candidate_wrapper(pool, monkeypatch)
        candidate = before.model_copy(deep=True)
        candidate.platform = "abandoned"
        with pytest.raises(OSError, match="wrapper construction"):
            await pools.load_model(SumModel(candidate))
        # Rollback acknowledgement, rather than a sleep, establishes safety.
        await asyncio.wait_for(pool._settle_recovery(), 10)
        assert not pool._pending_reload
        if old is not None:
            assert pool.has_model(before)
            assert (await old._send("metadata")).platform == "committed"
            await pools.unload_model(old)
            with pytest.raises(WorkerError):
                await asyncio.wait_for(old._send("metadata"), 5)
        else:
            assert not pool.has_model(candidate)
            retry_settings = before.model_copy(deep=True)
            retry_settings.platform = "retried"
            retry = await pools.load_model(SumModel(retry_settings))
            assert (await retry._send("metadata")).platform == "retried"
        assert not pool._staged
    finally:
        await pools.close()
