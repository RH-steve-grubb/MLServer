"""Pool retirement must follow all model owners, not just replay inventory."""

import asyncio
import gc

import pytest

from mlserver.errors import ModelNotFound
from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.registry import MultiModelRegistry
from mlserver.settings import ModelSettings, ModelParameters
from tests.fixtures import ErrorModel


async def exercise_staged_owner(settings, first, outcome, inference_request):
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    pools = InferencePoolRegistry(settings)
    await pools.start()
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    first = first.model_copy(deep=True)
    first.name = "committed-owner"
    second = first.model_copy(deep=True)
    second.name = "staged-owner"

    async def later_hook(model):
        if model.name == second.name:
            entered.set()
            await release.wait()
            if outcome == "failure":
                raise ValueError("later preparation failed")
        return model

    async def cleanup_done(model):
        if model.name == second.name:
            cleaned.set()
        return model

    registry = MultiModelRegistry(
        on_model_load=[pools.load_model, later_hook],
        on_model_unload=[pools.unload_model, cleanup_done],
        model_operation_timeout=10,
    )
    attempt = None
    try:
        old = await registry.load(first)
        pool = await pools._find(old)
        workers = list(pool._workers.values())
        if outcome == "timeout":
            registry._model_operation_timeout = 2
        attempt = asyncio.create_task(registry.load(second))
        await asyncio.wait_for(entered.wait(), 10)
        await registry.unload(first.name)
        # The workers have prepared B but it is not yet eligible for replay.
        # Removing A must not destroy this accepted owner of the same pool.
        assert not pool._closing
        assert not pool.has_model(second)
        if outcome != "timeout":
            release.set()
        if outcome == "success":
            current = await asyncio.wait_for(asyncio.shield(attempt), 10)
            assert current.ready
            metadata = await current._send("metadata")
            assert metadata.name == second.name
            response = await current.predict(inference_request)
            assert response.model_name == second.name
            registry._model_operation_timeout = 10
            await registry.unload(second.name)
        else:
            expected = ValueError if outcome == "failure" else asyncio.TimeoutError
            with pytest.raises(expected):
                await asyncio.wait_for(asyncio.shield(attempt), 10)
            release.set()
            with pytest.raises(ModelNotFound):
                await registry.get_model(second.name)
        await asyncio.wait_for(cleaned.wait(), 10)
        if pool is pools._default_pool:
            assert not pool._closing  # The default pool remains available.
            assert pool.empty()
        else:
            assert pool not in pools._pools.values()
            assert pool._closing
            assert all(not worker.is_alive() for worker in workers)
    finally:
        release.set()
        if attempt is not None:
            await asyncio.gather(attempt, return_exceptions=True)
        await pools.close()


@pytest.mark.parametrize("pool_gid", [None, "shared-group"])
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
async def test_staged_owner_controls_pool_retirement(
    settings, sum_model_settings, inference_request, pool_gid, outcome
):
    # Purpose: Exercise unload of the last committed model while another model is staged
    # in the same default or named pool.
    # Expect: Accepted staging keeps the pool alive until publication or acknowledged
    # failure/timeout cleanup; final empty named pools then retire.
    # Method: The shared helper pauses a later frontend hook with events, unloads the
    # first model and checks real serving or cleanup for each outcome.
    first = sum_model_settings.model_copy(deep=True)
    first.parameters.inference_pool_gid = pool_gid
    await exercise_staged_owner(settings, first, outcome, inference_request)


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
async def test_staged_owner_controls_custom_environment_retirement(
    settings, existing_env_model_settings, inference_request, outcome
):
    # Purpose: Exercise the staged-owner retirement contract in a custom-environment
    # pool.
    # Expect: Removing its last committed model cannot destroy accepted preparation;
    # success/failure/timeout eventually releases the appropriate pool owner.
    # Method: Run the shared event-controlled staging scenario through the existing
    # approved environment fixture and observe serving and confirmed child exit.
    # Reuse the repository's existing custom-environment fixture and trust mode.
    try:
        await exercise_staged_owner(
            settings, existing_env_model_settings, outcome, inference_request
        )
    finally:
        # Collect model/exception cycles while fixture logging is still open.
        gc.collect()


async def test_failed_fresh_pool_load_can_retire_its_recovery_owner(settings):
    # Purpose: Exercise a failed fresh load that is the only model owner of a named
    # pool.
    # Expect: Recovery ownership must permit eventual automatic retirement rather than
    # keeping an empty pool and workers indefinitely.
    # Method: Use the trusted ErrorModel, retain the actual acquired workers, await the
    # failed registry load and inspect pool removal and child liveness.
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    pools = InferencePoolRegistry(settings)
    await pools.start()
    model = ErrorModel(
        ModelSettings(
            name="failed-only-owner",
            implementation=ErrorModel,
            parameters=ModelParameters(
                inference_pool_gid="failed-only-group", load_error=True
            ),
        )
    )
    try:
        pool = await pools._get_or_create(model)
        await pool.start()
        workers = list(pool._workers.values())
        with pytest.raises(Exception):
            await pools.load_model(model)
        assert pool not in pools._pools.values()
        assert pool._closing
        assert all(not worker.is_alive() for worker in workers)
    finally:
        await pools.close()
