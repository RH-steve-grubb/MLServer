import pytest
import os
import asyncio
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import cast
from unittest.mock import patch

from mlserver.env import Environment, compute_hash_of_file
from mlserver.model import MLModel
from mlserver.settings import Settings, ModelSettings, ModelParameters
from mlserver.types import InferenceRequest
from mlserver.codecs import StringCodec
from mlserver.parallel.errors import EnvironmentNotFound, InferencePoolUnavailable
from mlserver.parallel.pool import InferencePool
from mlserver.parallel.registry import (
    InferencePoolRegistry,
    _set_environment_hash,
    _get_environment_hash,
    _append_gid_environment_hash,
    ENV_HASH_ATTR,
)

from ..fixtures import SumModel, EnvModel


# Test hook - must be module-level to be pickle-able for multiprocessing
async def _test_load_hook_raises(model: MLModel) -> MLModel:
    """Test hook that raises to prove it executed in worker process"""
    raise RuntimeError("test_hook_executed_in_worker")


@pytest.fixture
async def env_model(
    inference_pool_registry: InferencePoolRegistry, env_model_settings: ModelSettings
) -> AsyncGenerator[MLModel, None]:
    env_model = EnvModel(env_model_settings)
    model = await inference_pool_registry.load_model(env_model)

    yield model

    await inference_pool_registry.unload_model(model)


@pytest.fixture
async def existing_env_model(
    inference_pool_registry: InferencePoolRegistry,
    existing_env_model_settings: ModelSettings,
) -> AsyncGenerator[MLModel, None]:
    env_model = EnvModel(existing_env_model_settings)
    model = await inference_pool_registry.load_model(env_model)

    yield model

    await inference_pool_registry.unload_model(model)


def test_set_environment_hash(sum_model: MLModel):
    env_hash = "0e46fce1decb7a89a8b91c71d8b6975630a17224d4f00094e02e1a732f8e95f3"
    _set_environment_hash(sum_model, env_hash)

    assert hasattr(sum_model, ENV_HASH_ATTR)
    assert getattr(sum_model, ENV_HASH_ATTR) == env_hash


@pytest.mark.parametrize(
    "env_hash",
    ["0e46fce1decb7a89a8b91c71d8b6975630a17224d4f00094e02e1a732f8e95f3", None],
)
def test_get_environment_hash(sum_model: MLModel, env_hash: str | None):
    _set_environment_hash(sum_model, env_hash)
    assert _get_environment_hash(sum_model) == env_hash


def test_get_environment_hash_not_set(sum_model: MLModel):
    with pytest.raises(AttributeError):
        _get_environment_hash(sum_model)


async def test_default_pool(
    inference_pool_registry: InferencePoolRegistry, settings: Settings
):
    assert inference_pool_registry._default_pool is not None

    worker_count = len(inference_pool_registry._default_pool._workers)
    assert worker_count == settings.parallel_workers


@pytest.mark.parametrize(
    ("first_gid", "second_gid", "same_lock"),
    [
        (None, None, True),
        ("shared-gid", "shared-gid", True),
        (None, "keyed-gid", False),
        ("first-gid", "second-gid", False),
    ],
    ids=[
        "same-default-pool",
        "same-keyed-pool",
        "default-and-keyed-pools",
        "different-keyed-pools",
    ],
)
async def test_get_pool_lock_uses_pool_identity(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    first_gid: str | None,
    second_gid: str | None,
    same_lock: bool,
):
    first_settings = deepcopy(sum_model_settings)
    assert first_settings.parameters is not None
    first_settings.parameters.inference_pool_gid = first_gid
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.parameters.inference_pool_gid = second_gid

    first_lock = await inference_pool_registry._get_pool_lock(
        SumModel(first_settings), loading=True
    )
    second_lock = await inference_pool_registry._get_pool_lock(
        SumModel(second_settings), loading=True
    )

    assert (first_lock is second_lock) is same_lock


async def test_get_pool_lock_matches_loading_and_unloading_environment_identity(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    mocker,
):
    async def hash_environment_path(_path: str) -> str:
        return "environment-hash"

    mocker.patch(
        "mlserver.parallel.registry.compute_hash_of_string",
        new=hash_environment_path,
    )
    settings = deepcopy(sum_model_settings)
    assert settings.parameters is not None
    settings.parameters.environment_path = "/tmp/model-environment"
    settings.parameters.inference_pool_gid = "environment-gid"

    loading_model = SumModel(settings)
    unloading_model = SumModel(deepcopy(settings))
    _set_environment_hash(unloading_model, "environment-hash-environment-gid")

    loading_lock = await inference_pool_registry._get_pool_lock(
        loading_model, loading=True
    )
    unloading_lock = await inference_pool_registry._get_pool_lock(unloading_model)

    assert loading_lock is unloading_lock


@pytest.mark.parametrize(
    "environment_parameter",
    ["environment_path", "environment_tarball"],
    ids=["environment-path", "environment-tarball"],
)
async def test_get_pool_lock_distinguishes_empty_gid_in_custom_environment(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    mocker,
    environment_parameter: str,
):
    async def hash_environment(_value: str) -> str:
        return "environment-hash"

    mocker.patch(
        "mlserver.parallel.registry.compute_hash_of_string", new=hash_environment
    )
    mocker.patch(
        "mlserver.parallel.registry.compute_hash_of_file", new=hash_environment
    )

    none_settings = deepcopy(sum_model_settings)
    empty_settings = deepcopy(sum_model_settings)
    assert none_settings.parameters is not None
    assert empty_settings.parameters is not None
    setattr(none_settings.parameters, environment_parameter, "/tmp/model-environment")
    setattr(empty_settings.parameters, environment_parameter, "/tmp/model-environment")
    none_settings.parameters.inference_pool_gid = None
    empty_settings.parameters.inference_pool_gid = ""

    none_lock = await inference_pool_registry._get_pool_lock(
        SumModel(none_settings), loading=True
    )
    empty_lock = await inference_pool_registry._get_pool_lock(
        SumModel(empty_settings), loading=True
    )

    assert none_lock is not empty_lock


@pytest.mark.parametrize(
    ("first_gid", "second_gid", "same_lock"),
    [
        (None, None, True),
        ("shared-gid", "shared-gid", True),
        (None, "different-gid", False),
    ],
    ids=["same-tarball-pool", "same-tarball-gid", "different-tarball-gids"],
)
async def test_get_pool_lock_uses_environment_tarball_identity(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    first_gid: str | None,
    second_gid: str | None,
    same_lock: bool,
    mocker,
):
    async def hash_environment_tarball(_path: str) -> str:
        return "environment-hash"

    mocker.patch(
        "mlserver.parallel.registry.compute_hash_of_file",
        new=hash_environment_tarball,
    )
    first_settings = deepcopy(sum_model_settings)
    assert first_settings.parameters is not None
    first_settings.parameters.environment_tarball = "environment.tar.gz"
    first_settings.parameters.inference_pool_gid = first_gid
    second_settings = deepcopy(sum_model_settings)
    assert second_settings.parameters is not None
    second_settings.parameters.environment_tarball = "environment.tar.gz"
    second_settings.parameters.inference_pool_gid = second_gid

    first_lock = await inference_pool_registry._get_pool_lock(
        SumModel(first_settings), loading=True
    )
    second_lock = await inference_pool_registry._get_pool_lock(
        SumModel(second_settings), loading=True
    )

    assert (first_lock is second_lock) is same_lock


async def test_get_pool_lock_returns_independent_lock_for_unresolved_unload(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    model = SumModel(deepcopy(sum_model_settings))

    first_lock = await inference_pool_registry._get_pool_lock(model)
    second_lock = await inference_pool_registry._get_pool_lock(model)

    assert first_lock is not second_lock
    assert not inference_pool_registry._operation_locks


async def test_get_pool_lock_returns_independent_lock_for_non_pool_model(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    settings = deepcopy(sum_model_settings)
    settings.parallel_workers = 0
    model = SumModel(settings)

    first_lock = await inference_pool_registry._get_pool_lock(model, loading=True)
    second_lock = await inference_pool_registry._get_pool_lock(model, loading=True)

    assert first_lock is not second_lock


async def test_close_drains_operation_locks_and_is_terminal(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    settings = deepcopy(sum_model_settings)
    assert settings.parameters is not None
    settings.parameters.inference_pool_gid = "close-gid"
    model = SumModel(settings)
    lock = await inference_pool_registry._get_pool_lock(model, loading=True)
    await lock.acquire()

    close_task = asyncio.create_task(inference_pool_registry.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    lock.release()
    await close_task

    assert inference_pool_registry._default_pool is None
    assert not inference_pool_registry._pools

    # Test idempotency
    await inference_pool_registry.close()

    model = SumModel(deepcopy(sum_model_settings))
    with pytest.raises(InferencePoolUnavailable):
        await inference_pool_registry.load_model(model)
    with pytest.raises(InferencePoolUnavailable):
        await inference_pool_registry.unload_model(model)


@pytest.mark.parametrize(
    ("pool_id", "expected_name"),
    [("failing-gid", "failing pool"), (None, "failing default pool")],
    ids=["keyed-pool", "default-pool"],
)
async def test_close_pool_removes_pool_after_cleanup_failure(
    inference_pool_registry: InferencePoolRegistry,
    pool_id: str | None,
    expected_name: str,
):
    class FailingPool:
        name = expected_name
        _env = object()

        async def close(self):
            raise RuntimeError("pool cleanup failed")

    failing_pool = cast(InferencePool, FailingPool())
    original_default_pool = inference_pool_registry._default_pool
    if pool_id is None:
        inference_pool_registry._default_pool = failing_pool
    else:
        inference_pool_registry._pools[pool_id] = failing_pool

    try:
        with pytest.raises(RuntimeError, match="pool cleanup failed"):
            await inference_pool_registry._close_pool(pool_id)

        if pool_id is None:
            assert inference_pool_registry._default_pool is None
        else:
            assert pool_id not in inference_pool_registry._pools
    finally:
        if pool_id is None and original_default_pool is not None:
            await original_default_pool.close()


@pytest.mark.parametrize(
    ("inference_pool_gid", "uses_default_pool"),
    [("dummy_id", False), (None, True), ("", True)],
)
async def test_load_model(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
    inference_pool_gid: str | None,
    uses_default_pool: bool,
):
    sum_model_settings = deepcopy(sum_model_settings)
    sum_model_settings.name = "foo"
    assert sum_model_settings.parameters is not None
    sum_model_settings.parameters.inference_pool_gid = inference_pool_gid
    sum_model = SumModel(sum_model_settings)

    model = await inference_pool_registry.load_model(sum_model)
    inference_response = await model.predict(inference_request)

    assert inference_response.id == inference_request.id
    assert inference_response.model_name == sum_model.settings.name
    assert len(inference_response.outputs) == 1

    if uses_default_pool:
        assert inference_pool_registry._default_pool is not None
        assert inference_pool_registry._default_pool.has_model(sum_model.settings)
        assert inference_pool_gid not in inference_pool_registry._pools
    else:
        assert inference_pool_gid is not None
        assert inference_pool_gid in inference_pool_registry._pools
        assert inference_pool_registry._pools[inference_pool_gid].has_model(
            sum_model.settings
        )

    await inference_pool_registry.unload_model(sum_model)


@pytest.mark.parametrize(
    "pool_gid",
    [None, "shared-gid"],
    ids=["default-pool", "keyed-pool"],
)
async def test_load_model_serializes_models_sharing_pool(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    pool_gid: str | None,
    mocker,
):
    entered = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    class FakePool:
        env_hash = None

        async def load_model(self, model: MLModel) -> MLModel:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            entered.set()
            await release.wait()
            active -= 1
            return model

    pool = FakePool()

    async def get_pool(_model: MLModel) -> FakePool:
        return pool

    mocker.patch.object(inference_pool_registry, "_get_or_create", new=get_pool)
    first_settings = deepcopy(sum_model_settings)
    first_settings.name = "shared-pool-first"
    assert first_settings.parameters is not None
    first_settings.parameters.inference_pool_gid = pool_gid
    second_settings = deepcopy(sum_model_settings)
    second_settings.name = "shared-pool-second"
    assert second_settings.parameters is not None
    second_settings.parameters.inference_pool_gid = pool_gid

    first_load = asyncio.create_task(
        inference_pool_registry.load_model(SumModel(first_settings))
    )
    await entered.wait()
    second_load = asyncio.create_task(
        inference_pool_registry.load_model(SumModel(second_settings))
    )
    await asyncio.sleep(0)

    assert not second_load.done()
    assert maximum_active == 1
    release.set()
    await asyncio.gather(first_load, second_load)


@pytest.mark.parametrize(
    "pool_gids",
    [
        (None, "keyed-gid"),
        ("gid-a", "gid-b"),
    ],
    ids=["default-and-keyed", "different-keyed-pools"],
)
async def test_load_model_allows_different_pools_concurrently(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    pool_gids: tuple[str | None, str | None],
    mocker,
):
    entered = 0
    maximum_active = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()
    active = 0

    class FakePool:
        env_hash = None

        async def load_model(self, model: MLModel) -> MLModel:
            nonlocal active, entered, maximum_active
            entered += 1
            active += 1
            maximum_active = max(maximum_active, active)
            if entered == 2:
                both_entered.set()
            await release.wait()
            active -= 1
            return model

    pools = {pool_gids[0]: FakePool(), pool_gids[1]: FakePool()}

    async def get_pool(model: MLModel) -> FakePool:
        assert model.settings.parameters is not None
        gid = model.settings.parameters.inference_pool_gid
        return pools[gid]

    mocker.patch.object(inference_pool_registry, "_get_or_create", new=get_pool)
    default_settings = deepcopy(sum_model_settings)
    default_settings.name = "first-pool-model"
    assert default_settings.parameters is not None
    default_settings.parameters.inference_pool_gid = pool_gids[0]
    keyed_settings = deepcopy(sum_model_settings)
    keyed_settings.name = "second-pool-model"
    assert keyed_settings.parameters is not None
    keyed_settings.parameters.inference_pool_gid = pool_gids[1]

    default_load = asyncio.create_task(
        inference_pool_registry.load_model(SumModel(default_settings))
    )
    keyed_load = asyncio.create_task(
        inference_pool_registry.load_model(SumModel(keyed_settings))
    )
    await both_entered.wait()

    assert maximum_active == 2
    release.set()
    await asyncio.gather(default_load, keyed_load)


@pytest.mark.parametrize(
    "pool_gid",
    [None, "shared-gid"],
    ids=["default-pool", "keyed-pool"],
)
async def test_load_model_serializes_unload_for_same_pool(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    pool_gid: str | None,
    mocker,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    unload_entered = asyncio.Event()

    class FakePool:
        env_hash = None

        async def load_model(self, model: MLModel) -> MLModel:
            load_started.set()
            await release_load.wait()
            return model

        async def unload_model(self, model: MLModel) -> MLModel:
            unload_entered.set()
            return model

    pool = FakePool()

    async def get_pool(_model: MLModel) -> FakePool:
        return pool

    mocker.patch.object(inference_pool_registry, "_get_or_create", new=get_pool)
    mocker.patch.object(inference_pool_registry, "_find", new=get_pool)
    mocker.patch.object(inference_pool_registry, "_close_pool_if_empty")

    load_settings = deepcopy(sum_model_settings)
    load_settings.name = "load-model"
    assert load_settings.parameters is not None
    load_settings.parameters.inference_pool_gid = pool_gid
    unload_settings = deepcopy(sum_model_settings)
    unload_settings.name = "unload-model"
    assert unload_settings.parameters is not None
    unload_settings.parameters.inference_pool_gid = pool_gid

    load_task = asyncio.create_task(
        inference_pool_registry.load_model(SumModel(load_settings))
    )
    await load_started.wait()

    unload_model = SumModel(unload_settings)
    _set_environment_hash(unload_model, None)
    unload_task = asyncio.create_task(
        inference_pool_registry.unload_model(unload_model)
    )
    await asyncio.sleep(0)

    assert not unload_task.done()
    assert not unload_entered.is_set()

    release_load.set()
    await asyncio.gather(load_task, unload_task)
    assert unload_entered.is_set()


@pytest.mark.parametrize(
    "pool_gids",
    [
        (None, "keyed-gid"),
        ("gid-a", "gid-b"),
    ],
    ids=["default-and-keyed", "different-keyed-pools"],
)
async def test_load_model_allows_unload_for_different_pools(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    pool_gids: tuple[str | None, str | None],
    mocker,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    unload_entered = asyncio.Event()

    class FakePool:
        env_hash = None

        async def load_model(self, model: MLModel) -> MLModel:
            load_started.set()
            await release_load.wait()
            return model

        async def unload_model(self, model: MLModel) -> MLModel:
            unload_entered.set()
            return model

    pools = {pool_gids[0]: FakePool(), pool_gids[1]: FakePool()}

    async def get_pool(model: MLModel) -> FakePool:
        assert model.settings.parameters is not None
        gid = model.settings.parameters.inference_pool_gid
        return pools[gid]

    mocker.patch.object(inference_pool_registry, "_get_or_create", new=get_pool)
    mocker.patch.object(inference_pool_registry, "_find", new=get_pool)
    mocker.patch.object(inference_pool_registry, "_close_pool_if_empty")

    load_settings = deepcopy(sum_model_settings)
    load_settings.name = "load-model"
    assert load_settings.parameters is not None
    load_settings.parameters.inference_pool_gid = pool_gids[0]
    unload_settings = deepcopy(sum_model_settings)
    unload_settings.name = "unload-model"
    assert unload_settings.parameters is not None
    unload_settings.parameters.inference_pool_gid = pool_gids[1]

    load_task = asyncio.create_task(
        inference_pool_registry.load_model(SumModel(load_settings))
    )
    await load_started.wait()

    unload_model = SumModel(unload_settings)
    _set_environment_hash(unload_model, None)
    unload_task = asyncio.create_task(
        inference_pool_registry.unload_model(unload_model)
    )
    await unload_entered.wait()

    assert not load_task.done()
    release_load.set()
    await asyncio.gather(load_task, unload_task)


async def test_load_model_cancellation_settles_and_preserves_pool_state(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    mocker,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    loaded_models: list[MLModel] = []

    class FakePool:
        env_hash = None

        async def load_model(self, model: MLModel) -> MLModel:
            load_started.set()
            await release_load.wait()
            loaded_models.append(model)
            return model

    pool = FakePool()

    async def get_pool(_model: MLModel) -> FakePool:
        return pool

    mocker.patch.object(inference_pool_registry, "_get_or_create", new=get_pool)

    model = SumModel(deepcopy(sum_model_settings))
    load_task = asyncio.create_task(inference_pool_registry.load_model(model))
    await load_started.wait()

    load_task.cancel()
    await asyncio.sleep(0)
    assert not load_task.done()

    release_load.set()
    await load_task

    assert loaded_models == [model]


async def test_unload_model_cancellation_settles_and_clears_pool_state(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    mocker,
):
    unload_started = asyncio.Event()
    release_unload = asyncio.Event()
    loaded_models: list[MLModel] = []

    class FakePool:
        env_hash = None

        async def unload_model(self, model: MLModel) -> MLModel:
            unload_started.set()
            await release_unload.wait()
            loaded_models.remove(model)
            return model

    pool = FakePool()

    async def find_pool(_model: MLModel) -> FakePool:
        return pool

    mocker.patch.object(inference_pool_registry, "_find", new=find_pool)
    mocker.patch.object(inference_pool_registry, "_close_pool_if_empty")

    model = SumModel(deepcopy(sum_model_settings))
    _set_environment_hash(model, None)
    loaded_models.append(model)

    unload_task = asyncio.create_task(inference_pool_registry.unload_model(model))
    await unload_started.wait()

    unload_task.cancel()
    await asyncio.sleep(0)
    assert not unload_task.done()

    release_unload.set()
    await unload_task

    assert not loaded_models


async def test_load_model_with_hooks(
    settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry,
):
    """
    Verify hooks execute when loading models in parallel workers.

    Uses a hook that raises an error to prove it executed in the worker process.
    If the hook didn't execute, the error wouldn't be raised and the test would fail.
    """
    # Create registry with a hook that raises
    registry = InferencePoolRegistry(settings, on_worker_load=[_test_load_hook_raises])

    try:
        sum_model = SumModel(sum_model_settings)

        # Load model - hook should execute in worker and raise
        with pytest.raises(Exception, match="test_hook_executed_in_worker"):
            await registry.load_model(sum_model)
    finally:
        await registry.close()


def check_sklearn_version(response):
    # Note: These versions come from the `environment.yml` found in
    # `./tests/testdata/environment.yaml`
    assert len(response.outputs) == 1
    assert response.outputs[0].name == "sklearn_version"
    [sklearn_version] = StringCodec.decode_output(response.outputs[0])
    assert sklearn_version == "1.6.1"


async def test_load_model_with_env(
    inference_pool_registry: InferencePoolRegistry,
    env_model: MLModel,
    inference_request: InferenceRequest,
):
    response = await env_model.predict(inference_request)
    check_sklearn_version(response)


async def test_load_model_with_existing_env(
    inference_pool_registry: InferencePoolRegistry,
    existing_env_model: MLModel,
    inference_request: InferenceRequest,
):
    response = await existing_env_model.predict(inference_request)
    check_sklearn_version(response)


async def test_load_creates_pool(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
):
    assert len(inference_pool_registry._pools) == 0
    env_model = EnvModel(env_model_settings)
    await inference_pool_registry.load_model(env_model)

    assert len(inference_pool_registry._pools) == 1


async def test_load_reuses_pool(
    inference_pool_registry: InferencePoolRegistry,
    env_model: MLModel,
    env_model_settings: ModelSettings,
):
    env_model_settings.name = "foo"
    new_model = EnvModel(env_model_settings)

    assert len(inference_pool_registry._pools) == 1
    await inference_pool_registry.load_model(new_model)

    assert len(inference_pool_registry._pools) == 1


async def test_load_reuses_env_folder(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
):
    env_model_settings.name = "foo"
    new_model = EnvModel(env_model_settings)

    # Make sure there's already existing env
    env_hash = await compute_hash_of_file(env_tarball)
    env_path = inference_pool_registry._get_env_path(env_hash)
    await Environment.from_tarball(env_tarball, env_path, env_hash)

    await inference_pool_registry.load_model(new_model)


async def test_reload_model_with_env(
    inference_pool_registry: InferencePoolRegistry,
    env_model: MLModel,
    env_model_settings: ModelSettings,
):
    assert env_model_settings.parameters is not None
    env_model_settings.parameters.version = "v2.0"
    new_model = EnvModel(env_model_settings)

    assert len(inference_pool_registry._pools) == 1
    await inference_pool_registry.load_model(new_model)
    await inference_pool_registry.unload_model(env_model)

    assert len(inference_pool_registry._pools) == 1


async def test_unload_model_removes_pool_if_empty(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
):
    env_model = EnvModel(env_model_settings)
    assert len(inference_pool_registry._pools) == 0

    model = await inference_pool_registry.load_model(env_model)
    assert len(inference_pool_registry._pools) == 1

    await inference_pool_registry.unload_model(model)

    env_hash = _get_environment_hash(model)
    assert env_hash is not None
    env_path = inference_pool_registry._get_env_path(env_hash)
    assert len(inference_pool_registry._pools) == 0
    assert not os.path.isdir(env_path)


async def test_invalid_env_hash(
    inference_pool_registry: InferencePoolRegistry, sum_model: MLModel
):
    _set_environment_hash(sum_model, "foo")
    with pytest.raises(EnvironmentNotFound):
        await inference_pool_registry._find(sum_model)


async def test_worker_stop(
    settings: Settings,
    inference_pool_registry: InferencePoolRegistry,
    sum_model: MLModel,
    inference_request: InferenceRequest,
    caplog,
):
    # Pick random worker and kill it
    default_pool = inference_pool_registry._default_pool
    assert default_pool is not None
    workers = list(default_pool._workers.values())
    stopped_worker = workers[0]
    stopped_worker.kill()

    async def wait_for_replacement() -> None:
        while (
            len(default_pool._dispatcher._ready_workers) != settings.parallel_workers
            or stopped_worker.pid in default_pool._dispatcher._ready_workers
        ):
            await asyncio.sleep(0.1)

    await asyncio.wait_for(wait_for_replacement(), timeout=5)

    # Ensure SIGCHD signal was handled
    assert f"with PID {stopped_worker.pid}" in caplog.text

    # Cycle through every worker
    assert len(default_pool._workers) == settings.parallel_workers
    for _ in range(settings.parallel_workers + 2):
        inference_response = await sum_model.predict(inference_request)
        assert len(inference_response.outputs) > 0


@pytest.mark.parametrize(
    "env_hash, inference_pool_gid, expected_env_hash",
    [
        ("dummy_hash", "dummy_gid", "dummy_hash-dummy_gid"),
    ],
)
async def test__get_environment_hash_gid(
    env_hash: str, inference_pool_gid: str | None, expected_env_hash: str
):
    _env_hash = _append_gid_environment_hash(env_hash, inference_pool_gid)
    assert _env_hash == expected_env_hash


async def test_default_and_default_gid(
    inference_pool_registry: InferencePoolRegistry,
    simple_model_settings: ModelSettings,
):
    simple_model_settings_gid = deepcopy(simple_model_settings)
    params = simple_model_settings_gid.parameters
    assert params is not None
    params.inference_pool_gid = "dummy_id"

    simple_model = SumModel(simple_model_settings)
    simple_model_gid = SumModel(simple_model_settings_gid)

    model = await inference_pool_registry.load_model(simple_model)
    model_gid = await inference_pool_registry.load_model(simple_model_gid)

    assert len(inference_pool_registry._pools) == 1
    await inference_pool_registry.unload_model(model)
    await inference_pool_registry.unload_model(model_gid)


async def test_env_and_env_gid(
    inference_request: InferenceRequest,
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
):
    env_model_settings = deepcopy(env_model_settings)
    assert env_model_settings.parameters is not None
    env_model_settings.parameters.environment_tarball = env_tarball

    env_model_settings_gid = deepcopy(env_model_settings)
    assert env_model_settings_gid.parameters is not None
    env_model_settings_gid.parameters.inference_pool_gid = "dummy_id"

    env_model = EnvModel(env_model_settings)
    env_model_gid = EnvModel(env_model_settings_gid)

    model = await inference_pool_registry.load_model(env_model)
    model_gid = await inference_pool_registry.load_model(env_model_gid)
    assert len(inference_pool_registry._pools) == 2

    response = await model.predict(inference_request)
    response_gid = await model_gid.predict(inference_request)
    check_sklearn_version(response)
    check_sklearn_version(response_gid)

    await inference_pool_registry.unload_model(model)
    await inference_pool_registry.unload_model(model_gid)


@pytest.mark.parametrize(
    "inference_pool_gid, autogenerate_inference_pool_gid",
    [
        ("dummy_gid", False),
        ("dummy_gid", True),
        (None, True),
        (None, False),
        ("", True),
        ("", False),
    ],
)
def test_autogenerate_inference_pool_gid(
    inference_pool_gid: str | None, autogenerate_inference_pool_gid: bool
):
    patch_uuid = "patch-uuid"
    with patch("uuid.uuid4", return_value=patch_uuid) as uuid4:
        model_settings = ModelSettings(
            name="dummy-model",
            implementation=SumModel,
            parameters=ModelParameters(
                inference_pool_gid=inference_pool_gid,
                autogenerate_inference_pool_gid=autogenerate_inference_pool_gid,
            ),
        )
    assert uuid4.call_count == int(
        autogenerate_inference_pool_gid and inference_pool_gid is None
    )

    expected_gid = (
        patch_uuid
        if autogenerate_inference_pool_gid and inference_pool_gid is None
        else inference_pool_gid
    )
    assert model_settings.parameters is not None
    assert model_settings.parameters.inference_pool_gid == expected_gid


async def test_same_gid_reuses_pool(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    """Two models with the same inference_pool_gid must share one pool instance."""
    shared_gid = "shared-gid"

    settings_a = deepcopy(sum_model_settings)
    settings_a.name = "model-a"
    assert settings_a.parameters is not None
    settings_a.parameters.inference_pool_gid = shared_gid

    settings_b = deepcopy(sum_model_settings)
    settings_b.name = "model-b"
    assert settings_b.parameters is not None
    settings_b.parameters.inference_pool_gid = shared_gid

    loaded_a = await inference_pool_registry.load_model(SumModel(settings_a))
    pool_after_first = inference_pool_registry._pools[shared_gid]

    loaded_b = await inference_pool_registry.load_model(SumModel(settings_b))
    pool_after_second = inference_pool_registry._pools[shared_gid]

    assert pool_after_first is pool_after_second
    assert len(inference_pool_registry._pools) == 1

    response_a = await loaded_a.predict(inference_request)
    response_b = await loaded_b.predict(inference_request)
    assert len(response_a.outputs) == 1
    assert len(response_b.outputs) == 1

    await inference_pool_registry.unload_model(loaded_a)
    await inference_pool_registry.unload_model(loaded_b)


async def test_same_gid_no_redundant_pool_spawn(
    inference_pool_registry: InferencePoolRegistry,
    simple_model_settings: ModelSettings,
):
    """Second load with same GID must not construct a new InferencePool.

    Regression test for the setdefault() bug where InferencePool() was
    eagerly evaluated (spawning workers) even when the gid was already
    registered.
    """
    shared_gid = "shared-gid"

    settings_a = deepcopy(simple_model_settings)
    settings_a.name = "model-a"
    assert settings_a.parameters is not None
    settings_a.parameters.inference_pool_gid = shared_gid

    settings_b = deepcopy(simple_model_settings)
    settings_b.name = "model-b"
    assert settings_b.parameters is not None
    settings_b.parameters.inference_pool_gid = shared_gid

    loaded_a = await inference_pool_registry.load_model(SumModel(settings_a))
    existing_pool = inference_pool_registry._pools[shared_gid]

    with patch("mlserver.parallel.registry.InferencePool") as MockPool:
        loaded_b = await inference_pool_registry.load_model(SumModel(settings_b))
        MockPool.assert_not_called()

    assert inference_pool_registry._pools[shared_gid] is existing_pool

    await inference_pool_registry.unload_model(loaded_a)
    await inference_pool_registry.unload_model(loaded_b)


async def test_same_gid_pool_cleanup(
    inference_pool_registry: InferencePoolRegistry,
    simple_model_settings: ModelSettings,
):
    """GID-only pool must be properly cleaned up when empty.

    Tests that when the last model using a GID-only pool is unloaded,
    the pool is correctly removed from the registry and its resources are freed.
    Validates the fix for the cleanup bug where GID-only pools were leaked.
    """
    shared_gid = "cleanup-test-gid"

    settings = deepcopy(simple_model_settings)
    assert settings.parameters is not None
    settings.parameters.inference_pool_gid = shared_gid

    # Load and verify pool exists
    loaded = await inference_pool_registry.load_model(SumModel(settings))
    assert shared_gid in inference_pool_registry._pools
    assert len(inference_pool_registry._pools) == 1

    # Unload and verify pool is cleaned up
    await inference_pool_registry.unload_model(loaded)
    assert shared_gid not in inference_pool_registry._pools
    assert len(inference_pool_registry._pools) == 0


async def test_same_gid_pool_cleanup_multi_model(
    inference_pool_registry: InferencePoolRegistry,
    simple_model_settings: ModelSettings,
):
    """Shared pool persists until all models unloaded."""
    shared_gid = "multi-model-gid"

    settings_a = deepcopy(simple_model_settings)
    settings_a.name = "model-a"
    assert settings_a.parameters is not None
    settings_a.parameters.inference_pool_gid = shared_gid

    settings_b = deepcopy(simple_model_settings)
    settings_b.name = "model-b"
    assert settings_b.parameters is not None
    settings_b.parameters.inference_pool_gid = shared_gid

    loaded_a = await inference_pool_registry.load_model(SumModel(settings_a))
    loaded_b = await inference_pool_registry.load_model(SumModel(settings_b))

    assert shared_gid in inference_pool_registry._pools
    assert len(inference_pool_registry._pools) == 1

    await inference_pool_registry.unload_model(loaded_a)
    assert shared_gid in inference_pool_registry._pools
    assert len(inference_pool_registry._pools) == 1

    await inference_pool_registry.unload_model(loaded_b)
    assert shared_gid not in inference_pool_registry._pools
    assert len(inference_pool_registry._pools) == 0


async def test_reload_model_same_gid(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    """Reloading a model with the same GID reuses the same inference pool."""
    settings = deepcopy(sum_model_settings)
    assert settings.parameters is not None
    settings.parameters.inference_pool_gid = "reload-gid"

    old_model = await inference_pool_registry.load_model(SumModel(settings))
    assert len(inference_pool_registry._pools) == 1
    assert "reload-gid" in inference_pool_registry._pools

    new_model = await inference_pool_registry.load_model(SumModel(settings))
    assert len(inference_pool_registry._pools) == 1
    assert "reload-gid" in inference_pool_registry._pools

    assert old_model != new_model

    await inference_pool_registry.unload_model(old_model)
    assert len(inference_pool_registry._pools) == 1
    assert "reload-gid" in inference_pool_registry._pools

    await inference_pool_registry.unload_model(new_model)
    assert len(inference_pool_registry._pools) == 0


async def test_reload_model_different_gid(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    """Reloading a model with a different GID loads it into the new pool
    and removes it from the old pool."""
    settings_gid1 = deepcopy(sum_model_settings)
    assert settings_gid1.parameters is not None
    settings_gid1.parameters.inference_pool_gid = "gid-1"

    settings_gid2 = deepcopy(sum_model_settings)
    assert settings_gid2.parameters is not None
    settings_gid2.parameters.inference_pool_gid = "gid-2"

    old_model = await inference_pool_registry.load_model(SumModel(settings_gid1))
    assert len(inference_pool_registry._pools) == 1
    assert "gid-1" in inference_pool_registry._pools

    new_model = await inference_pool_registry.load_model(SumModel(settings_gid2))
    assert len(inference_pool_registry._pools) == 2
    assert "gid-2" in inference_pool_registry._pools

    assert old_model != new_model

    await inference_pool_registry.unload_model(old_model)
    assert len(inference_pool_registry._pools) == 1
    assert "gid-2" in inference_pool_registry._pools

    await inference_pool_registry.unload_model(new_model)
    assert len(inference_pool_registry._pools) == 0


async def test_reload_model_default_to_gid(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    """Reloading from the default pool to a keyed pool moves pool ownership."""
    default_settings = deepcopy(sum_model_settings)
    assert default_settings.parameters is not None
    default_settings.parameters.inference_pool_gid = None

    gid_settings = deepcopy(default_settings)
    assert gid_settings.parameters is not None
    gid_settings.parameters.inference_pool_gid = "migration-gid"

    old_model = await inference_pool_registry.load_model(SumModel(default_settings))
    assert len(inference_pool_registry._pools) == 0
    assert inference_pool_registry._default_pool is not None
    assert inference_pool_registry._default_pool.has_model(old_model.settings)

    new_model = await inference_pool_registry.load_model(SumModel(gid_settings))
    assert old_model != new_model
    assert len(inference_pool_registry._pools) == 1
    assert "migration-gid" in inference_pool_registry._pools
    assert inference_pool_registry._pools["migration-gid"].has_model(new_model.settings)

    await inference_pool_registry.unload_model(old_model)
    assert not inference_pool_registry._default_pool.has_model(old_model.settings)
    assert "migration-gid" in inference_pool_registry._pools

    await inference_pool_registry.unload_model(new_model)
    assert len(inference_pool_registry._pools) == 0


async def test_reload_model_gid_to_default(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
):
    """Reloading from a keyed pool to the default pool moves pool ownership."""
    gid_settings = deepcopy(sum_model_settings)
    assert gid_settings.parameters is not None
    gid_settings.parameters.inference_pool_gid = "migration-gid"

    default_settings = deepcopy(gid_settings)
    assert default_settings.parameters is not None
    default_settings.parameters.inference_pool_gid = None

    old_model = await inference_pool_registry.load_model(SumModel(gid_settings))
    assert len(inference_pool_registry._pools) == 1
    assert "migration-gid" in inference_pool_registry._pools
    assert inference_pool_registry._pools["migration-gid"].has_model(old_model.settings)
    assert inference_pool_registry._default_pool is not None
    assert not inference_pool_registry._default_pool.has_model(old_model.settings)

    new_model = await inference_pool_registry.load_model(SumModel(default_settings))
    assert len(inference_pool_registry._pools) == 1
    assert "migration-gid" in inference_pool_registry._pools
    assert old_model != new_model
    assert inference_pool_registry._default_pool.has_model(new_model.settings)

    await inference_pool_registry.unload_model(old_model)
    assert len(inference_pool_registry._pools) == 0

    await inference_pool_registry.unload_model(new_model)
    assert not inference_pool_registry._default_pool.has_model(new_model.settings)
