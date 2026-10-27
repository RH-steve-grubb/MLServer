"""Opt-in CUDA blocking waits; importing this module does not load CUDA."""

import ctypes
from ctypes.util import find_library
from importlib import metadata
from pathlib import Path
from typing import Any

from mlserver.errors import ModelLoadError


def _runtime(library: str) -> Any:
    try:
        runtime = ctypes.CDLL(library)
        runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        runtime.cudaSetDevice.restype = ctypes.c_int
        runtime.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
        runtime.cudaSetDeviceFlags.restype = ctypes.c_int
        runtime.cudaGetDeviceFlags.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        runtime.cudaGetDeviceFlags.restype = ctypes.c_int
        runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        runtime.cudaGetErrorString.restype = ctypes.c_char_p
    except (OSError, AttributeError) as error:
        raise ModelLoadError(
            "Cannot load the CUDA runtime for blocking synchronization "
            f"({library}): {error}"
        ) from error
    return runtime


def _library() -> str:
    try:
        distribution = metadata.distribution("nvidia-cuda-runtime-cu12")
    except metadata.PackageNotFoundError:
        pass
    else:
        path = Path(
            str(distribution.locate_file("nvidia/cuda_runtime/lib/libcudart.so.12"))
        )
        if path.is_file():
            return str(path.resolve())
    library = find_library("cudart")
    if not library:
        raise ModelLoadError(
            "CUDA blocking synchronization requires libcudart from the NVIDIA "
            "CUDA runtime package or the system library path"
        )
    return library


def _check(runtime: Any, status: int, operation: str) -> None:
    if status:
        message = runtime.cudaGetErrorString(status)
        detail = message.decode(errors="replace") if message else "unknown error"
        raise ModelLoadError(f"{operation} failed (CUDA {status}: {detail})")


def _select(runtime: Any, device_id: int) -> None:
    if type(device_id) is not int or device_id < 0:
        raise ModelLoadError("CUDA device_id must be a non-negative integer")
    _check(runtime, runtime.cudaSetDevice(device_id), f"cudaSetDevice({device_id})")


def _flags(runtime: Any) -> int:
    flags = ctypes.c_uint()
    _check(
        runtime,
        runtime.cudaGetDeviceFlags(ctypes.byref(flags)),
        "cudaGetDeviceFlags",
    )
    # The runtime can also return the implicit MapHost bit. Only the three
    # scheduling bits determine whether synchronization blocks the host thread.
    if flags.value & 7 != 4:
        raise ModelLoadError(
            "CUDA blocking synchronization was not applied "
            f"(device flags={flags.value})"
        )
    return flags.value


def configure_blocking_sync(device_id: int = 0) -> dict[str, Any]:
    """Apply blocking waits before ORT constructs a session on this device.

    This policy belongs to the device context in this process, so other models
    sharing that context also observe it. Do not reset it during model unload.
    """
    if type(device_id) is not int or device_id < 0:
        raise ModelLoadError("CUDA device_id must be a non-negative integer")
    library = _library()
    runtime = _runtime(library)
    # Use the visible CUDA ordinal configured on the provider, rather than
    # silently modifying device zero. CUDA 12 permits updating the flags of an
    # already initialized primary context after selecting the device.
    _select(runtime, device_id)
    _check(runtime, runtime.cudaSetDeviceFlags(4), "cudaSetDeviceFlags(4)")
    return {
        "library": library,
        "device_id": device_id,
        "requested_flags": 4,
        "before_session_flags": _flags(runtime),
    }


def read_device_flags(library: str, device_id: int = 0) -> int:
    """Verify the selected device retained blocking waits after session load."""
    if type(device_id) is not int or device_id < 0:
        raise ModelLoadError("CUDA device_id must be a non-negative integer")
    runtime = _runtime(library)
    _select(runtime, device_id)
    return _flags(runtime)
