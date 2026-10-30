"""Lifecycle regressions use existing trusted runtimes and controlled barriers."""

import asyncio

import pytest

from mlserver.errors import ModelLoadError, ModelNotFound
from mlserver.registry import MultiModelRegistry
from tests.fixtures import SumModel


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("stage", ["load", "hook", "wrapped-hook"])
async def test_abandoned_load_never_publishes(
    sum_model_settings, monkeypatch, replacement, stage
):
    entered = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    models = []
    registry = MultiModelRegistry(model_operation_timeout=0.05)
    old = await registry.load(sum_model_settings) if replacement else None

    def initialise(settings):
        model = SumModel(settings)
        models.append(model)

        async def load():
            entered.set()
            await release.wait()
            return True

        async def unload():
            cleaned.set()
            return True

        monkeypatch.setattr(model, "unload", unload)
        if stage == "load":
            monkeypatch.setattr(model, "load", load)
        return model

    async def hook(model):
        if stage == "wrapped-hook":
            model = initialise(model.settings)
        return model

    async def paused_hook(model):
        entered.set()
        await release.wait()
        return model

    registry._model_initialiser = initialise
    registry._on_model_load = [hook, paused_hook] if stage != "load" else []
    if old is not None:
        single = registry._models[old.name]
        single._model_initialiser = initialise
        single._on_model_load = registry._on_model_load

    attempt = asyncio.create_task(registry.load(sum_model_settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        if old is not None:
            assert await registry.get_model(old.name, old.version) is old
            assert old.ready
        else:
            with pytest.raises(ModelNotFound):
                await registry.get_model(sum_model_settings.name)
        # Native preparation may still be running. Its retained owner must
        # finish it before cleanup, and prevent a conflicting retry.
        release.set()
        await asyncio.wait_for(cleaned.wait(), 5)
        if old is not None:
            await registry.unload_version(old.name, old.version)
        assert await registry.get_models() == []
        registry._on_model_load = []
        if sum_model_settings.name in registry._models:
            registry._models[sum_model_settings.name]._on_model_load = []
        retry = await registry.load(sum_model_settings)
        assert retry.ready
        assert await registry.get_model(retry.name, retry.version) is retry
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        if sum_model_settings.name in registry._models:
            await registry.unload(sum_model_settings.name)


async def test_failed_load_cleanup_remains_owned(sum_model_settings, monkeypatch):
    registry = MultiModelRegistry()
    model = SumModel(sum_model_settings)
    primary = ValueError("preparation failed")
    cleanup = RuntimeError("resource still live")

    async def load():
        raise primary

    async def unload():
        raise cleanup

    monkeypatch.setattr(model, "load", load)
    monkeypatch.setattr(model, "unload", unload)
    registry._model_initialiser = lambda _: model
    with pytest.raises(ModelLoadError) as failure:
        await registry.load(sum_model_settings)
    assert failure.value.__cause__ is primary
    assert failure.value.cleanup_error is cleanup
    assert await registry.get_models() == []
    assert sum_model_settings.name in registry._models
    with pytest.raises(RuntimeError, match="resource still live"):
        await registry.load(sum_model_settings)
    monkeypatch.setattr(model, "unload", SumModel.unload.__get__(model))
    await registry.unload(sum_model_settings.name)


async def test_unload_timeout_retains_cleanup(sum_model_settings, monkeypatch):
    registry = MultiModelRegistry(model_operation_timeout=0.05)
    model = await registry.load(sum_model_settings)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def unload():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return True

    monkeypatch.setattr(model, "unload", unload)
    operation = asyncio.create_task(registry.unload(model.name))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert await registry.get_models() == []
        assert model.name in registry._models
        repeated = asyncio.create_task(registry.unload(model.name))
        release.set()
        await asyncio.wait_for(repeated, 5)
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


async def test_retry_waits_for_abandoned_preparation(sum_model_settings, monkeypatch):
    registry = MultiModelRegistry(model_operation_timeout=0.05)
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    abandoned = SumModel(sum_model_settings)

    async def load():
        entered.set()
        await release.wait()
        return True

    async def unload():
        cleaned.set()
        return True

    monkeypatch.setattr(abandoned, "load", load)
    monkeypatch.setattr(abandoned, "unload", unload)
    registry._model_initialiser = lambda _: abandoned
    attempt = asyncio.create_task(registry.load(sum_model_settings))
    retry = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        registry._model_operation_timeout = 5
        registry._models[abandoned.name]._model_initialiser = SumModel
        retry = asyncio.create_task(registry.load(sum_model_settings))
        await asyncio.sleep(0)
        assert not retry.done()
        release.set()
        current = await asyncio.wait_for(retry, 5)
        assert cleaned.is_set()
        assert current is not abandoned
        assert current.ready
        assert await registry.get_model(current.name, current.version) is current
    finally:
        release.set()
        await asyncio.gather(
            attempt, *([retry] if retry else []), return_exceptions=True
        )
        await registry.unload(sum_model_settings.name)


async def test_postcommit_cleanup_failure_keeps_replacement(
    sum_model_settings, monkeypatch
):
    registry = MultiModelRegistry()
    old = await registry.load(sum_model_settings)

    async def fail():
        raise RuntimeError("old resources retained")

    monkeypatch.setattr(old, "unload", fail)
    with pytest.raises(RuntimeError, match="old resources retained"):
        await registry.load(sum_model_settings)
    current = await registry.get_model(old.name, old.version)
    assert current is not old
    assert current.ready
    monkeypatch.setattr(old, "unload", SumModel.unload.__get__(old))
    await registry.unload(old.name)


async def test_failed_replacement_cannot_mutate_committed_settings(
    sum_model_settings,
):
    registry = MultiModelRegistry()
    old = await registry.load(sum_model_settings)
    original = old.settings.model_dump()

    async def failing_hook(model):
        model.settings.platform = "uncommitted-platform"
        raise ValueError("hook failed")

    registry._models[old.name]._on_model_load = [failing_hook]
    with pytest.raises(ValueError, match="hook failed"):
        await registry.load(old.settings)
    assert old.settings.model_dump() == original
    assert await registry.get_model(old.name, old.version) is old
    await registry.unload(old.name)


async def test_failure_preserves_unrelated_names_versions_and_default(
    sum_model_settings, monkeypatch
):
    registry = MultiModelRegistry(model_operation_timeout=0.05)
    versions = []
    for version in ["1", "2", None]:
        settings = sum_model_settings.model_copy(deep=True)
        settings.parameters.version = version
        versions.append(await registry.load(settings))
    other = sum_model_settings.model_copy(deep=True)
    other.name = "other-name"
    other_model = await registry.load(other)
    failed = SumModel(versions[0].settings.model_copy(deep=True))
    entered, release = asyncio.Event(), asyncio.Event()

    async def load():
        entered.set()
        await release.wait()
        return True

    monkeypatch.setattr(failed, "load", load)
    registry._models[failed.name]._model_initialiser = lambda _: failed
    operation = asyncio.create_task(registry.load(failed.settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        for model in versions[:2]:
            assert await registry.get_model(model.name, model.version) is model
            assert model.ready
        assert await registry.get_model(failed.name) is versions[-1]
        assert await registry.get_model(other.name) is other_model
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)
        await registry.unload(failed.name)
        await registry.unload(other.name)


async def test_interrupted_cleanup_does_not_cancel_native_preparation(
    sum_model_settings, monkeypatch
):
    import threading

    registry = MultiModelRegistry(model_operation_timeout=0.05)
    entered, release = asyncio.Event(), threading.Event()
    finished = threading.Event()
    unloaded = asyncio.Event()
    loop = asyncio.get_running_loop()
    model = SumModel(sum_model_settings)

    def native_prepare():
        loop.call_soon_threadsafe(entered.set)
        release.wait()
        finished.set()

    async def load():
        await asyncio.to_thread(native_prepare)
        return True

    async def unload():
        assert finished.is_set()
        unloaded.set()
        return True

    monkeypatch.setattr(model, "load", load)
    monkeypatch.setattr(model, "unload", unload)
    registry._model_initialiser = lambda _: model
    attempt = asyncio.create_task(registry.load(sum_model_settings))
    retry = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        cleanup = registry._models[model.name]._cleanup_tasks[model]
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert not unloaded.is_set()
        registry._model_operation_timeout = 5
        retry = asyncio.create_task(registry.unload(model.name))
        await asyncio.sleep(0)
        assert not retry.done()
        release.set()
        await asyncio.wait_for(retry, 5)
        assert unloaded.is_set()
    finally:
        release.set()
        await asyncio.gather(
            attempt, *([retry] if retry else []), return_exceptions=True
        )
        await asyncio.wait_for(asyncio.to_thread(finished.wait), 5)


async def test_version_unload_waits_for_abandoned_preparation(
    sum_model_settings, monkeypatch
):
    registry = MultiModelRegistry(model_operation_timeout=0.05)
    healthy_settings = sum_model_settings.model_copy(deep=True)
    healthy_settings.parameters.version = "2"
    healthy = await registry.load(healthy_settings)
    entered, release = asyncio.Event(), asyncio.Event()
    failed_settings = sum_model_settings.model_copy(deep=True)
    failed_settings.parameters.version = "1"
    partial = SumModel(failed_settings)

    async def load():
        entered.set()
        await release.wait()
        return True

    monkeypatch.setattr(partial, "load", load)
    registry._models[healthy.name]._model_initialiser = lambda _: partial
    attempt = asyncio.create_task(registry.load(failed_settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        with pytest.raises(asyncio.TimeoutError):
            await registry.unload_version(healthy.name, "2")
        assert await registry.get_model(healthy.name, "2") is healthy
        assert healthy.ready
        release.set()
        registry._model_operation_timeout = 5
        await registry.unload_version(healthy.name, "2")
        assert await registry.get_models() == []
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        if healthy.name in registry._models:
            await registry.unload(healthy.name)


@pytest.mark.parametrize("target_exists", [False, True], ids=["unused", "committed"])
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("outcome", ["error", "timeout", "success"])
async def test_hook_version_change_keeps_instance_ownership(
    sum_model_settings, monkeypatch, replacement, wrap, outcome, target_exists
):
    registry = MultiModelRegistry(model_operation_timeout=5)
    healthy_settings = sum_model_settings.model_copy(deep=True)
    healthy_settings.parameters.version = "3"
    healthy = await registry.load(healthy_settings)
    target = None
    target_unloaded = asyncio.Event()
    if target_exists:
        target_settings = sum_model_settings.model_copy(deep=True)
        target_settings.parameters.version = "2"
        target = await registry.load(target_settings)

        async def unload_target():
            target_unloaded.set()
            return True

        monkeypatch.setattr(target, "unload", unload_target)
    incoming = sum_model_settings.model_copy(deep=True)
    incoming.parameters.version = "4"
    old = await registry.load(incoming) if replacement else None
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def hook(model):
        if wrap:
            model = SumModel(model.settings.model_copy(deep=True))
        model.settings.parameters.version = "2"

        async def load():
            entered.set()
            await release.wait()
            return True

        async def unload():
            cleaned.set()
            return True

        monkeypatch.setattr(model, "load", load)
        monkeypatch.setattr(model, "unload", unload)
        return model

    async def reject(model):
        entered.set()
        raise ValueError("hook rejected preparation")

    registry._models[incoming.name]._on_model_load = (
        [hook, reject] if outcome == "error" else [hook]
    )
    if outcome == "timeout":
        registry._model_operation_timeout = 0.05
    attempt = asyncio.create_task(registry.load(incoming.model_copy(deep=True)))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if outcome == "success":
            release.set()
            current = await asyncio.wait_for(attempt, 5)
            assert await registry.get_model(incoming.name, "2") is current
            assert current.ready
            with pytest.raises(ModelNotFound):
                await registry.get_model(incoming.name, "4")
            assert await registry.get_model(incoming.name) is healthy
            if target is not None:
                assert not target.ready
                assert target_unloaded.is_set()
        else:
            expected = ValueError if outcome == "error" else asyncio.TimeoutError
            with pytest.raises(expected):
                await asyncio.wait_for(asyncio.shield(attempt), 5)
            release.set()
            await asyncio.wait_for(cleaned.wait(), 5)
            if target is None:
                with pytest.raises(ModelNotFound):
                    await registry.get_model(incoming.name, "2")
            else:
                assert await registry.get_model(incoming.name, "2") is target
                assert target.ready
                assert not target_unloaded.is_set()
            if old is None:
                with pytest.raises(ModelNotFound):
                    await registry.get_model(incoming.name, "4")
            else:
                assert await registry.get_model(incoming.name, "4") is old
                assert old.ready
                assert old.version == "4"
        assert await registry.get_model(incoming.name, "3") is healthy
        assert healthy.ready
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        registry._model_operation_timeout = 5
        await registry.unload(incoming.name)


@pytest.mark.parametrize("outcome", ["error", "timeout"])
async def test_redirected_commit_owns_both_displaced_instances(
    sum_model_settings, monkeypatch, outcome
):
    registry = MultiModelRegistry(model_operation_timeout=5)
    loaded = {}
    for version in ("2", "3", "4"):
        settings = sum_model_settings.model_copy(deep=True)
        settings.parameters.version = version
        loaded[version] = await registry.load(settings)
    entered, release, original_cleaned = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def cleanup_target():
        entered.set()
        if outcome == "error":
            raise ValueError("target cleanup failed")
        await release.wait()
        return True

    async def cleanup_original():
        original_cleaned.set()
        return True

    async def redirect(model):
        model.settings.parameters.version = "2"
        return model

    monkeypatch.setattr(loaded["2"], "unload", cleanup_target)
    monkeypatch.setattr(loaded["4"], "unload", cleanup_original)
    registry._models[sum_model_settings.name]._on_model_load = [redirect]
    settings = sum_model_settings.model_copy(deep=True)
    settings.parameters.version = "4"
    registry._model_operation_timeout = 0.05 if outcome == "timeout" else 5
    attempt = asyncio.create_task(registry.load(settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        expected = ValueError if outcome == "error" else asyncio.TimeoutError
        with pytest.raises(expected):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        current = await registry.get_model(settings.name, "2")
        assert current is not loaded["2"]
        assert current.ready
        assert await registry.get_model(settings.name) is loaded["3"]
        assert not loaded["2"].ready
        assert not loaded["4"].ready
        await asyncio.wait_for(original_cleaned.wait(), 5)
        with pytest.raises(ModelNotFound):
            await registry.get_model(settings.name, "4")
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        registry._model_operation_timeout = 5
        monkeypatch.setattr(loaded["2"], "unload", cleanup_original)
        await registry.unload(settings.name)


@pytest.mark.parametrize("retained_count", [1, 3])
async def test_unload_all_attempts_every_settled_cleanup_owner(
    sum_model_settings, monkeypatch, retained_count
):
    registry = MultiModelRegistry()
    models = []
    attempts = {}
    for version in range(retained_count + 1):
        settings = sum_model_settings.model_copy(deep=True)
        settings.parameters.version = str(version)
        models.append(await registry.load(settings))
    failed, healthy = models[:-1], models[-1]
    healthy_cleaned = asyncio.Event()

    async def clean():
        healthy_cleaned.set()
        return True

    monkeypatch.setattr(healthy, "unload", clean)
    for model in failed:

        async def fail(model=model):
            attempts[model.version] = attempts.get(model.version, 0) + 1
            raise RuntimeError(f"retained cleanup {model.version}")

        monkeypatch.setattr(model, "unload", fail)
    try:
        for model in failed:
            with pytest.raises(RuntimeError, match="retained cleanup"):
                await registry.unload_version(model.name, model.version)
        with pytest.raises(RuntimeError, match="retained cleanup"):
            await registry.unload(healthy.name)
        assert healthy_cleaned.is_set()
        assert not healthy.ready
        with pytest.raises(ModelNotFound):
            await registry.get_models(healthy.name)
        owners = registry._models[healthy.name]._cleanup_tasks
        assert set(owners) == set(failed)
        assert all(attempts[model.version] == 2 for model in failed)
        assert all(task.done() and task.exception() for task in owners.values())
    finally:
        for model in failed:
            monkeypatch.setattr(model, "unload", SumModel.unload.__get__(model))
        if healthy.name in registry._models:
            await registry.unload(healthy.name)
    assert await registry.get_models() == []


async def test_unload_all_waits_for_preparation_before_independent_cleanup(
    sum_model_settings, monkeypatch
):
    registry = MultiModelRegistry(model_operation_timeout=5)
    settings = sum_model_settings.model_copy(deep=True)
    settings.parameters.version = "2"
    healthy = await registry.load(settings)
    settings = settings.model_copy(deep=True)
    settings.parameters.version = "1"
    partial = SumModel(settings)
    entered, release, partial_cleaned, healthy_cleaned = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def load():
        entered.set()
        await release.wait()
        return True

    async def fail():
        partial_cleaned.set()
        raise RuntimeError("retained settled cleanup")

    async def clean():
        healthy_cleaned.set()
        return True

    monkeypatch.setattr(partial, "load", load)
    monkeypatch.setattr(partial, "unload", fail)
    monkeypatch.setattr(healthy, "unload", clean)
    registry._models[healthy.name]._model_initialiser = lambda _: partial
    registry._model_operation_timeout = 0.05
    attempt = asyncio.create_task(registry.load(settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attempt), 5)
        with pytest.raises(asyncio.TimeoutError):
            await registry.unload(healthy.name)
        assert not healthy_cleaned.is_set()
        assert not partial_cleaned.is_set()
        assert await registry.get_model(healthy.name, healthy.version) is healthy
        assert healthy.ready
        release.set()
        await asyncio.wait_for(partial_cleaned.wait(), 5)
        registry._model_operation_timeout = 5
        with pytest.raises(RuntimeError, match="retained settled cleanup"):
            await registry.unload(healthy.name)
        assert healthy_cleaned.is_set()
    finally:
        release.set()
        await asyncio.gather(attempt, return_exceptions=True)
        monkeypatch.setattr(partial, "unload", SumModel.unload.__get__(partial))
        registry._model_operation_timeout = 5
        if healthy.name in registry._models:
            await registry.unload(healthy.name)
