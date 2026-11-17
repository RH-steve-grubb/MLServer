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
    # Purpose: Exercise fresh and replacement deadlines during runtime load, hooks, and
    # wrapper substitution.
    # Expect: Abandoned candidates must not become serving entries; an old committed
    # instance survives and a clean retry succeeds.
    # Method: Hold the selected preparation stage with events, await timeout, then
    # release it and observe exact-instance cleanup and registry lookups.
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
    # Purpose: Combine preparation failure with failed resource cleanup.
    # Expect: The primary cause and cleanup error must both remain observable, and retry
    # cannot bypass the retained failure.
    # Method: Inject separate load/unload errors on one trusted model, inspect lookup
    # and retry outcomes, then restore unload to release the owner.
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
    # Purpose: Exercise a caller deadline while model unload is still running.
    # Expect: Serving eligibility disappears immediately, but a later unload joins the
    # one retained cleanup instead of running it twice.
    # Method: Pause unload with events, let its waiter time out, start a second waiter
    # and release the original cleanup; count unload calls.
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
    # Purpose: Exercise retry while an abandoned candidate is still preparing.
    # Expect: The retry must wait for settlement and cleanup, then publish a distinct
    # ready instance.
    # Method: Hold the first load beyond its deadline, start retry with a larger budget,
    # release the barrier and compare model identities and cleanup events.
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
    # Purpose: Exercise old-instance cleanup failure after replacement publication.
    # Expect: The replacement remains the selected ready model even though load reports
    # cleanup failure.
    # Method: Make the old model's unload raise, reload the same public identity and
    # inspect the committed lookup before restoring cleanup.
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
    # Purpose: Check settings ownership during a replacement hook failure.
    # Expect: Mutating the candidate must not change the committed instance or its
    # configuration.
    # Method: Pass the old settings into reload, mutate the candidate in a rejecting
    # hook and compare the original settings snapshot and lookup identity.
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
    # Purpose: Check lookup isolation while one version's replacement times out.
    # Expect: Healthy versions, the unversioned default and another name retain their
    # exact committed instances and readiness.
    # Method: Load those controls, block replacement preparation with events and query
    # each lookup after its deadline.
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
    # Purpose: Exercise cancellation of a cleanup owner while preparation runs in a real
    # thread.
    # Expect: A later cleanup must join native completion before unloading the model.
    # Method: Block to_thread work with a threading event, cancel its cleanup task,
    # start explicit unload and verify unload follows the thread's finished marker.
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
    # Purpose: Exercise another version's unload while preparation under the same name
    # remains uncertain.
    # Expect: The healthy version stays ready until the abandoned preparation settles;
    # unload then succeeds.
    # Method: Hold version 1 past its deadline, time out unload of version 2, release
    # preparation and repeat unload with a generous bound.
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
    # Purpose: Exercise hook redirection to another version, with optional wrapping,
    # existing targets and precommit outcomes.
    # Expect: Failure preserves both committed identities; success publishes only the
    # redirected candidate and retires any displaced target.
    # Method: Use event-held preparation and explicit error/timeout/success branches,
    # then inspect exact version/default lookups and target cleanup events.
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
    # Purpose: Exercise a successful redirected replacement followed by old-target
    # cleanup failure or timeout.
    # Expect: Both displaced instances lose eligibility and receive cleanup owners while
    # the redirected replacement remains committed.
    # Method: Redirect version 4 to version 2, fault or pause version 2 cleanup, and
    # independently observe version 4 cleanup plus all resulting lookups.
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
    # Purpose: Combine earlier failed unloads with a later unload-all request.
    # Expect: Every settled failed owner is retried and the healthy version is unloaded
    # even though the overall operation still reports failure.
    # Method: Create one or several retained failures, count retries and healthy
    # cleanup, and inspect retained task errors before restoring release.
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
    # Purpose: Separate unfinished preparation from settled failed cleanup during
    # unload-all.
    # Expect: Uncertain preparation blocks independent removal; after settlement, a
    # retained cleanup error must not skip healthy cleanup.
    # Method: Hold a failed candidate's load with events, time out unload-all, then
    # release it and verify both cleanup attempts on the next call.
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


async def test_interrupted_unload_all_owns_every_selected_instance(
    sum_model_settings, monkeypatch
):
    # Purpose: Exercise interruption after bulk removal selects instances but before
    # child unload coroutines start.
    # Expect: Every selected version must still be retired and cleaned by an independent
    # owner.
    # Method: Cancel the accepted task at the selection boundary, then await per-
    # instance cleanup markers and verify no models remain visible.
    registry = MultiModelRegistry()
    models = []
    cleaned = []
    for version in ("1", "2"):
        settings = sum_model_settings.model_copy(deep=True)
        settings.parameters.version = version
        model = await registry.load(settings)
        models.append(model)
        event = asyncio.Event()
        cleaned.append(event)

        async def unload(event=event):
            event.set()
            return True

        monkeypatch.setattr(model, "unload", unload)
    single = registry._models[sum_model_settings.name]
    get_models = single.get_models

    async def interrupt_after_selection():
        selected = await get_models()
        # Model-operation deadlines cancel this accepted task. Reach the same
        # interruption point deterministically, before any unload child starts.
        asyncio.current_task().cancel()
        return selected

    monkeypatch.setattr(single, "get_models", interrupt_after_selection)
    try:
        with pytest.raises(asyncio.CancelledError):
            await registry.unload(sum_model_settings.name)
        monkeypatch.setattr(single, "get_models", get_models)
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in cleaned)), 5)
        assert all(not model.ready for model in models)
        assert await registry.get_models() == []
    finally:
        monkeypatch.setattr(single, "get_models", get_models)
        if sum_model_settings.name in registry._models:
            await registry.unload(sum_model_settings.name)
        # Also release test-owned instances on the affected implementation.
        for model in models:
            if model.ready:
                await model.unload()
