"""CUDA Graph buffer tests with fake device values; no CUDA driver is called."""

from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import weakref

import numpy as np
import pytest

from mlserver.errors import ModelLoadError
from mlserver_onnx import cuda


class DeviceValue:
    def __init__(self, array: np.ndarray, events: list[str]) -> None:
        self.array = array.copy()
        self.events = events

    def update_inplace(self, tensor: np.ndarray) -> None:
        self.events.append("upload")
        self.array[:] = tensor


class Binding:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.inputs: dict[str, DeviceValue] = {}
        self.outputs: dict[str, DeviceValue] = {}
        self.before_download: Any = None

    def bind_ortvalue_input(self, name: str, value: DeviceValue) -> None:
        self.inputs[name] = value

    def bind_ortvalue_output(self, name: str, value: DeviceValue) -> None:
        self.outputs[name] = value

    def synchronize_inputs(self) -> None:
        self.events.append("sync-input")

    def synchronize_outputs(self) -> None:
        self.events.append("sync-output")

    def copy_outputs_to_cpu(self) -> list[np.ndarray]:
        self.events.append("download")
        if self.before_download:
            self.before_download()
        return [value.array.copy() for value in self.outputs.values()]


class Session:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.binding = Binding(events)
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.options = {"device_id": "1", "enable_cuda_graph": "1", "use_tf32": "0"}
        self.inputs = [
            SimpleNamespace(name="image", shape=[2, 3], type="tensor(float)")
        ]
        self.outputs = [
            SimpleNamespace(name=name, shape=[2, 3], type="tensor(float)")
            for name in ("boxes", "scores")
        ]
        self.disable_fallback = Mock()
        self.before_run: Any = None

    def get_providers(self) -> list[str]:
        return self.providers

    def get_provider_options(self) -> dict:
        return {"CUDAExecutionProvider": self.options}

    def get_inputs(self) -> list:
        return self.inputs

    def get_outputs(self) -> list:
        return self.outputs

    def io_binding(self) -> Binding:
        return self.binding

    def run_with_iobinding(self, binding: Binding) -> None:
        self.events.append("run")
        if self.before_run:
            self.before_run()
        for multiplier, value in enumerate(binding.outputs.values(), 1):
            value.array[:] = binding.inputs["image"].array * multiplier


@pytest.fixture
def device_values(monkeypatch):
    events: list[str] = []
    allocated: list[DeviceValue] = []
    devices = []

    def from_numpy(array, device, ordinal):
        devices.append((device, ordinal))
        value = DeviceValue(array, events)
        allocated.append(value)
        return value

    def from_shape(shape, dtype, device, ordinal):
        return from_numpy(np.zeros(shape, dtype=dtype), device, ordinal)

    monkeypatch.setattr(
        cuda.ort,
        "OrtValue",
        SimpleNamespace(
            ortvalue_from_numpy=from_numpy, ortvalue_from_shape_and_type=from_shape
        ),
    )
    return events, allocated, devices


def test_each_call_refreshes_input_and_downloads_independent_cpu_outputs(device_values):
    events, allocated, devices = device_values
    session = Session(events)
    proxy = cuda.CudaGraphSession(session, device_id=1)
    tensor = np.arange(6, dtype=np.float32).reshape(2, 3)
    first = proxy.run(None, {"image": tensor})
    tensor[:] += 10
    second = proxy.run(["scores", "boxes"], {"image": tensor})
    third = proxy.run([], {"image": np.ascontiguousarray(tensor[:, ::-1])})
    assert len(allocated) == 3 and devices == [("cuda", 1)] * 3
    assert events == ["upload", "sync-input", "run", "sync-output", "download"] * 3
    np.testing.assert_array_equal(first[0], np.arange(6).reshape(2, 3))
    np.testing.assert_array_equal(second[0], tensor * 2)
    np.testing.assert_array_equal(second[1], tensor)
    np.testing.assert_array_equal(third[0], tensor[:, ::-1])
    assert proxy.get_inputs() is session.inputs
    assert session.options["use_tf32"] == "0"
    session.disable_fallback.assert_called_once_with()


@pytest.mark.parametrize(
    "failure",
    ["provider", "device", "graph", "inputs", "shape", "rank", "dtype", "output"],
)
def test_invalid_session_metadata_fails_before_device_allocation(
    device_values, failure
):
    events, allocated, _ = device_values
    session = Session(events)
    if failure == "provider":
        session.providers.reverse()
    elif failure == "device":
        session.options["device_id"] = "0"
    elif failure == "graph":
        session.options["enable_cuda_graph"] = "0"
    elif failure == "inputs":
        session.inputs = []
    elif failure == "shape":
        session.inputs[0].shape = [None, 3]
    elif failure == "rank":
        session.inputs[0].shape = None
    elif failure == "dtype":
        session.outputs[0].type = "tensor(double)"
    else:
        session.outputs = []
    with pytest.raises(ModelLoadError):
        cuda.CudaGraphSession(session, device_id=1)
    assert allocated == []


@pytest.mark.parametrize("device_id", [-1, True, "0"])
def test_invalid_device_id_does_not_touch_session(device_id):
    session = Mock()
    with pytest.raises(ModelLoadError, match="device_id"):
        cuda.CudaGraphSession(session, device_id)
    assert session.mock_calls == []


def test_invalid_request_options_feed_and_outputs_do_not_upload(device_values):
    events, _, _ = device_values
    proxy = cuda.CudaGraphSession(Session(events), 1)
    tensor = np.zeros((2, 3), dtype=np.float32)
    for feed in (
        {},
        {"other": tensor},
        {"image": tensor.astype(np.float64)},
        {"image": tensor.reshape(3, 2)},
        {"image": tensor[:, ::-1]},
    ):
        with pytest.raises(ValueError):
            proxy.run(None, feed)
    with pytest.raises(ValueError, match="run_options"):
        proxy.run(None, {"image": tensor}, run_options=object())
    with pytest.raises(ValueError, match="Unknown"):
        proxy.run(["missing"], {"image": tensor})
    assert events == []


def test_run_failure_releases_lock_and_next_call_refreshes_input(device_values):
    events, _, _ = device_values
    session = Session(events)
    proxy = cuda.CudaGraphSession(session, 1)
    session.before_run = Mock(side_effect=RuntimeError("device error"))
    tensor = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="device error"):
        proxy.run(None, {"image": tensor})
    session.before_run = None
    tensor[:] = 3
    actual = proxy.run(["scores"], {"image": tensor})
    np.testing.assert_array_equal(actual[0], tensor * 2)
    assert events.count("upload") == 2 and events.count("download") == 1


@pytest.mark.parametrize("blocked_stage", ["run", "download"])
def test_concurrent_calls_protect_upload_through_download(
    device_values, blocked_stage, monkeypatch
):
    events, _, _ = device_values
    session = Session(events)
    proxy = cuda.CudaGraphSession(session, 1)
    entered, release, second_attempt = Event(), Event(), Event()

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()
            self.calls = 0
            self.second_was_blocked = False

        def __enter__(self):
            self.calls += 1
            if self.calls == 2:
                self.second_was_blocked = not self.lock.acquire(blocking=False)
                second_attempt.set()
                if self.second_was_blocked:
                    self.lock.acquire()
            else:
                self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    observed_lock = ObservedLock()
    monkeypatch.setattr(proxy, "_lock", observed_lock)

    def before_run():
        if not entered.is_set():
            entered.set()
            assert release.wait(3)

    if blocked_stage == "run":
        session.before_run = before_run
    else:
        session.binding.before_download = before_run
    first = np.ones((2, 3), dtype=np.float32)
    second = first * 7
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(proxy.run, ["boxes"], {"image": first})
        try:
            assert entered.wait(3)
            b = executor.submit(proxy.run, ["boxes"], {"image": second})
            assert second_attempt.wait(3)
            assert observed_lock.second_was_blocked
            assert events.count("upload") == 1
        finally:
            release.set()
        np.testing.assert_array_equal(a.result(timeout=3)[0], first)
        np.testing.assert_array_equal(b.result(timeout=3)[0], second)


def test_session_and_buffers_are_released_with_proxy(device_values):
    events, allocated, _ = device_values
    session = Session(events)
    proxy = cuda.CudaGraphSession(session, 1)
    references = [weakref.ref(value) for value in [proxy, session, *allocated]]
    allocated.clear()
    del proxy, session
    gc.collect()
    assert all(reference() is None for reference in references)
