"""Blocking-sync setup tests mock libcudart and never call the CUDA driver."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlserver.errors import ModelLoadError
from mlserver_onnx import cuda_sync


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    path = tmp_path / "libcudart.so.12"
    path.write_bytes(b"mock-library")
    events = []
    library = SimpleNamespace(
        cudaSetDevice=Mock(
            side_effect=lambda device: events.append(("device", device)) or 0
        ),
        cudaSetDeviceFlags=Mock(
            side_effect=lambda flags: events.append(("flags", flags)) or 0
        ),
        cudaGetErrorString=Mock(return_value=b"invalid device"),
    )

    def read(pointer):
        events.append(("read", None))
        pointer._obj.value = 12
        return 0

    library.cudaGetDeviceFlags = Mock(side_effect=read)
    distribution = SimpleNamespace(locate_file=Mock(return_value=path))
    monkeypatch.setattr(
        cuda_sync.metadata, "distribution", Mock(return_value=distribution)
    )
    monkeypatch.setattr(cuda_sync.ctypes, "CDLL", Mock(return_value=library))
    monkeypatch.setattr(cuda_sync, "find_library", Mock())
    return library, path, events


def test_selected_device_flag_order_and_post_session_readback(runtime):
    library, path, events = runtime
    receipt = cuda_sync.configure_blocking_sync(device_id=1)
    events.append(("session", None))
    actual = cuda_sync.read_device_flags(receipt["library"], device_id=1)
    assert receipt == {
        "library": str(path.resolve()),
        "device_id": 1,
        "requested_flags": 4,
        "before_session_flags": 12,
    }
    assert actual == 12
    assert events == [
        ("device", 1),
        ("flags", 4),
        ("read", None),
        ("session", None),
        ("device", 1),
        ("read", None),
    ]
    library.cudaSetDeviceFlags.assert_called_once_with(4)
    cuda_sync.find_library.assert_not_called()


@pytest.mark.parametrize("actual", [4, 12])
def test_readback_accepts_blocking_with_or_without_implicit_maphost(runtime, actual):
    library, path, _ = runtime

    def read(pointer):
        pointer._obj.value = actual
        return 0

    library.cudaGetDeviceFlags.side_effect = read
    assert cuda_sync.read_device_flags(str(path)) == actual
    library.cudaSetDeviceFlags.assert_not_called()


@pytest.mark.parametrize(
    "operation", ["cudaSetDevice", "cudaSetDeviceFlags", "cudaGetDeviceFlags"]
)
def test_cuda_errors_are_model_load_errors(runtime, operation):
    library, _, _ = runtime
    function = getattr(library, operation)
    function.side_effect = None
    function.return_value = 1
    with pytest.raises(ModelLoadError, match=operation):
        cuda_sync.configure_blocking_sync()
    if operation == "cudaSetDevice":
        library.cudaSetDeviceFlags.assert_not_called()


def test_missing_scheduling_bit_is_rejected(runtime):
    library, path, _ = runtime

    def read(pointer):
        pointer._obj.value = 8
        return 0

    library.cudaGetDeviceFlags.side_effect = read
    with pytest.raises(ModelLoadError, match="flags=8"):
        cuda_sync.read_device_flags(str(path))


@pytest.mark.parametrize("device", [-1, True, "0"])
def test_invalid_device_fails_before_library_loading(monkeypatch, device):
    load = Mock()
    monkeypatch.setattr(cuda_sync, "_library", load)
    monkeypatch.setattr(cuda_sync.ctypes, "CDLL", Mock())
    with pytest.raises(ModelLoadError, match="device_id"):
        cuda_sync.configure_blocking_sync(device)
    with pytest.raises(ModelLoadError, match="device_id"):
        cuda_sync.read_device_flags("unused", device)
    load.assert_not_called()
    cuda_sync.ctypes.CDLL.assert_not_called()


def test_system_library_is_used_when_optional_wheel_is_absent(runtime, monkeypatch):
    library, _, _ = runtime
    monkeypatch.setattr(
        cuda_sync.metadata,
        "distribution",
        Mock(
            side_effect=cuda_sync.metadata.PackageNotFoundError(
                "nvidia-cuda-runtime-cu12"
            )
        ),
    )
    monkeypatch.setattr(cuda_sync, "find_library", Mock(return_value="libcudart.so.12"))
    receipt = cuda_sync.configure_blocking_sync()
    assert receipt["library"] == "libcudart.so.12"
    cuda_sync.ctypes.CDLL.assert_called_once_with("libcudart.so.12")
    library.cudaSetDevice.assert_called_once_with(0)


def test_missing_runtime_fails_only_when_setup_is_requested(monkeypatch):
    monkeypatch.setattr(
        cuda_sync.metadata,
        "distribution",
        Mock(
            side_effect=cuda_sync.metadata.PackageNotFoundError(
                "nvidia-cuda-runtime-cu12"
            )
        ),
    )
    monkeypatch.setattr(cuda_sync, "find_library", Mock(return_value=None))
    load = Mock()
    monkeypatch.setattr(cuda_sync.ctypes, "CDLL", load)
    with pytest.raises(ModelLoadError, match="requires libcudart"):
        cuda_sync.configure_blocking_sync()
    load.assert_not_called()


def test_unloadable_runtime_error_is_actionable(runtime, monkeypatch):
    monkeypatch.setattr(
        cuda_sync.ctypes, "CDLL", Mock(side_effect=OSError("missing dependency"))
    )
    with pytest.raises(ModelLoadError, match="missing dependency"):
        cuda_sync.configure_blocking_sync()


def test_import_does_not_discover_or_load_cuda(monkeypatch):
    discover = Mock()
    load = Mock()
    monkeypatch.setattr(cuda_sync.metadata, "distribution", discover)
    monkeypatch.setattr(cuda_sync.ctypes, "CDLL", load)
    importlib.reload(cuda_sync)
    discover.assert_not_called()
    load.assert_not_called()
