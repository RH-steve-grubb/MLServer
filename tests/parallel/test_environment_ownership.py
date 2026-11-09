"""Extracted directories outlive workers, but never retain stale deletion owners."""

import asyncio
import gc
import io
from pathlib import Path
import sys
import tarfile

import pytest

from mlserver.parallel.errors import InferencePoolUnavailable
from mlserver.parallel.registry import InferencePoolRegistry
from mlserver.registry import MultiModelRegistry
from tests.fixtures import SumModel


@pytest.fixture
def local_environment_tarball(tmp_path):
    # A complete local interpreter environment, with no dependency installation.
    tarball = tmp_path / "local-environment.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        executable = tarfile.TarInfo("bin/python")
        executable.type = tarfile.SYMTYPE
        executable.linkname = sys.executable
        archive.addfile(executable)
        configuration = (Path(sys.prefix) / "pyvenv.cfg").read_bytes()
        member = tarfile.TarInfo("pyvenv.cfg")
        member.size = len(configuration)
        archive.addfile(member, io.BytesIO(configuration))
        packages = tarfile.TarInfo(
            f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        packages.type = tarfile.SYMTYPE
        packages.linkname = str(Path(sys.prefix) / packages.name)
        archive.addfile(packages)
        marker = tarfile.TarInfo("environment-marker")
        marker.size = 4
        archive.addfile(marker, io.BytesIO(b"live"))
    return tarball


async def test_tarball_reuse_survives_old_frontend_collection_and_restart(
    settings,
    sum_model_settings,
    local_environment_tarball,
    development_mode,
    inference_request,
):
    # Purpose: Exercise repeated extraction at the same environment path while old
    # frontend references are collected.
    # Expect: Stale deletion authority cannot remove the current interpreter; real
    # replacement workers still replay and serve correctly.
    # Method: Reuse a local tarball for three load/unload cycles, collect retained
    # references, spawn replacements and check file markers, metadata, prediction and
    # final deletion.
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    settings.worker_start_timeout = 5
    pools = InferencePoolRegistry(settings)
    await pools.start()
    registry = MultiModelRegistry(
        on_model_load=[pools.load_model],
        on_model_unload=[pools.unload_model],
        model_initialiser=pools.model_initialiser,
        model_operation_timeout=10,
    )
    incoming = sum_model_settings.model_copy(deep=True)
    incoming.parameters.environment_tarball = str(local_environment_tarball)
    retained = []
    current = pool = path = None
    try:
        for cycle in range(3):
            current = await registry.load(incoming.model_copy(deep=True))
            pool = await pools._find(current)
            path = Path(pool._env._env_path)
            retained.append(current)
            assert (await current._send("metadata")).name == current.name
            if cycle:
                retained[:-1] = [None] * (len(retained) - 1)
                gc.collect()
            assert (path / "bin/python").exists()
            assert (path / "environment-marker").read_bytes() == b"live"
            replacement = await asyncio.wait_for(pool._start_worker(), 10)
            assert replacement.pid in pool._dispatcher._ready_workers
            for _ in range(2):
                assert (await current._send("metadata")).name == current.name
            response = await current.predict(inference_request)
            assert response.model_name == current.name
            await registry.unload(current.name)
            replacement = None
        assert not path.exists()
    finally:
        # Repair only the affected implementation's test-owned missing path so
        # its remaining destructor can finish after verified worker shutdown.
        if path is not None and not path.exists() and pools._pools:
            path.mkdir(parents=True, exist_ok=True)
        await pools.close()
        registry._models.clear()  # Workers have exited; drop test frontend refs.
        retained.clear()
        current = pool = None
        gc.collect()


async def test_failed_pool_close_retains_files_until_successful_release(
    settings,
    sum_model_settings,
    local_environment_tarball,
    development_mode,
    monkeypatch,
):
    # Purpose: Exercise environment-pool close whose child exit is initially
    # unconfirmed.
    # Expect: Files and ownership survive failed close, reuse is blocked, and successful
    # cleanup revokes stale deletion authority before re-extraction.
    # Method: Inject close failure with real workers alive, restore close, reload the
    # same tarball and verify an old owner cannot delete the new serving environment.
    settings.parallel_workers = 1
    settings.model_operation_timeout = 10
    pools = InferencePoolRegistry(settings)
    await pools.start()
    incoming = sum_model_settings.model_copy(deep=True)
    incoming.parameters.environment_tarball = str(local_environment_tarball)
    try:
        model = await pools.load_model(SumModel(incoming))
        pool = await pools._find(model)
        path = Path(pool._env._env_path)
        owner = pool._env
        workers = list(pool._workers.values())
        close = pool.close

        async def unconfirmed_close():
            raise RuntimeError("worker exit not confirmed")

        monkeypatch.setattr(pool, "close", unconfirmed_close)
        with pytest.raises(RuntimeError, match="not confirmed"):
            await pools._close_pool(pool.env_hash)
        assert pool._env is owner
        assert all(worker.is_alive() for worker in workers)
        assert (path / "bin/python").exists()
        assert (await model._send("metadata")).name == model.name
        with pytest.raises(InferencePoolUnavailable):
            await pools.load_model(SumModel(incoming.model_copy(deep=True)))
        monkeypatch.setattr(pool, "close", close)
        await pools._close_pool(pool.env_hash)
        assert all(not worker.is_alive() for worker in workers)
        assert not path.exists()
        # Even an explicitly retained old owner cannot delete a new extraction.
        current = await pools.load_model(SumModel(incoming.model_copy(deep=True)))
        owner.close()
        assert (path / "bin/python").exists()
        assert (await current._send("metadata")).name == current.name
        await pools.unload_model(current)
        assert not path.exists()
    finally:
        if "close" in locals():
            monkeypatch.setattr(pool, "close", close)
        await pools.close()
