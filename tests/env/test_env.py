import pytest
import sys
import os
import shutil
import gc

from mlserver.env import Environment, compute_hash_of_file


@pytest.fixture
def expected_python_folder(env_python_version: tuple[int, int]) -> str:
    major, minor = env_python_version
    return f"python{major}.{minor}"


async def test_compute_hash(env_tarball: str):
    env_hash = await compute_hash_of_file(env_tarball)
    assert len(env_hash) == 64


async def test_from_tarball(env_tarball: str, tmp_path: str):
    env = await Environment.from_tarball(env_tarball, env_path=tmp_path)

    executable = os.path.join(tmp_path, "bin", "python")
    assert os.path.isfile(executable)
    assert os.path.isdir(env._lib_path)
    assert len(env.env_hash) == 64


def test_sys_path(env: Environment, expected_python_folder: str):
    expected = [
        f"{env._env_path}/lib/{expected_python_folder}.zip",
        f"{env._env_path}/lib/{expected_python_folder}",
        f"{env._env_path}/lib/{expected_python_folder}/lib-dynload",
        f"{env._env_path}/lib/{expected_python_folder}/site-packages",
    ]
    assert env._sys_path == expected


def test_sys_path_empty(env: Environment):
    # Force lib_path to be empty
    env.__dict__["_lib_path"] = ""
    expected: list[str] = []
    assert env._sys_path == expected


@pytest.mark.parametrize(
    "folder_name",
    ["python3.11", "python3.12", ""],
)
def test_lib_path(env: Environment, folder_name: str, expected_python_folder: str):
    default_lib_path = env._lib_path
    assert default_lib_path == f"{env._env_path}/lib/{expected_python_folder}"

    if folder_name:
        base_path = os.path.dirname(default_lib_path)
        expected_path = os.path.join(base_path, folder_name)
        os.rename(default_lib_path, expected_path)
    else:
        shutil.rmtree(env._lib_path)
        expected_path = ""

    # Force to look again for lib_path
    del env.__dict__["_lib_path"]
    assert env._lib_path == expected_path


def test_bin_path(env: Environment):
    assert env._bin_path == f"{env._env_path}/bin"


def test_exec_path(env: Environment):
    assert env._exec_path == f"{env._env_path}/bin/python"


def test_activate_env(env: Environment):
    assert env._env_path not in ",".join(sys.path)
    assert env._env_path not in os.environ["PATH"]

    with env:
        sys_path = env._sys_path
        assert sys_path

        for i, p in enumerate(sys_path):
            assert sys.path[i] == p

        bin_paths = os.environ["PATH"].split(os.pathsep)
        assert bin_paths[0] == env._bin_path

        exec_path = os.path.join(bin_paths[0], "python")
        assert exec_path == env._exec_path

    assert env._env_path not in ",".join(sys.path)
    assert env._env_path not in os.environ["PATH"]


@pytest.mark.parametrize("delete_env", [True, False])
def test_close_revokes_stale_directory_deletion(tmp_path, delete_env):
    # Purpose: Exercise owned and borrowed environment release across path reuse.
    # Expect: An old environment object must never delete a later owner's files.
    # Method: Close twice, recreate a marker at the same path, then collect the old
    # object and verify the marker survives.
    path = tmp_path / "environment"
    path.mkdir()
    env = Environment(str(path), "owned", delete_env=delete_env)
    env.close()
    assert path.exists() is not delete_env
    env.close()
    path.mkdir(exist_ok=True)
    (path / "new-owner").touch()
    del env
    gc.collect()
    assert (path / "new-owner").exists()


def test_failed_close_retains_directory_owner(tmp_path, monkeypatch):
    # Purpose: Exercise an environment directory removal that fails once.
    # Expect: The directory and deletion authority must survive for a successful retry.
    # Method: Inject rmtree failure, restore it, then close again and check the real
    # directory disappears.
    path = tmp_path / "environment"
    path.mkdir()
    env = Environment(str(path), "owned")

    def fail(*args, **kwargs):
        raise OSError("cannot remove environment yet")

    with monkeypatch.context() as patch:
        patch.setattr("mlserver.env.shutil.rmtree", fail)
        with pytest.raises(OSError, match="cannot remove"):
            env.close()
    assert path.exists()
    env.close()
    assert not path.exists()


def test_worker_copy_cannot_release_parent_environment(tmp_path):
    # Purpose: Check that copying an environment into a worker does not transfer
    # deletion authority.
    # Expect: A different process must leave the parent's directory intact.
    # Method: Change the recorded owner PID, attempt close, then restore the real owner
    # and verify release.
    path = tmp_path / "environment"
    path.mkdir()
    env = Environment(str(path), "owned")
    env._owner_pid = os.getpid() + 1
    env.close()
    assert path.exists()
    env._owner_pid = os.getpid()
    env.close()
    assert not path.exists()


def test_missing_child_during_release_keeps_directory_owned(tmp_path, monkeypatch):
    # Purpose: Distinguish a missing nested entry from an already released environment
    # root.
    # Expect: An incomplete removal must report failure and retain authority to retry.
    # Method: Make rmtree raise FileNotFoundError while the root exists, then restore it
    # and verify a second close removes the root.
    path = tmp_path / "environment"
    path.mkdir()
    env = Environment(str(path), "owned")

    def fail(*args, **kwargs):
        raise FileNotFoundError("a nested entry disappeared")

    with monkeypatch.context() as patch:
        patch.setattr("mlserver.env.shutil.rmtree", fail)
        with pytest.raises(FileNotFoundError, match="nested entry"):
            env.close()
    assert path.exists()
    env.close()
    assert not path.exists()
