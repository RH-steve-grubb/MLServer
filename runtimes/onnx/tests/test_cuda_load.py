"""Normal graph-setting load wiring, with every ORT session/device call mocked."""

import gc
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import weakref

import numpy as np
import pytest

from mlserver.errors import InferenceError, ModelLoadError
from mlserver.settings import ModelParameters, ModelSettings
from mlserver.types import InferenceRequest, MetadataTensor, RequestInput, RequestOutput
from mlserver_onnx import OnnxModel
from mlserver_onnx import cuda
from mlserver_onnx import onnx as runtime
from mlserver_onnx.settings import OnnxSettings, get_onnx_settings


def settings(extra: dict) -> ModelSettings:
    return ModelSettings(
        name="custom-fixed-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri="/unused/model.onnx", extra=extra),
    )


def request(outputs: list[str] | None = None) -> InferenceRequest:
    return InferenceRequest(
        inputs=[
            RequestInput(
                name="image", shape=[2, 3], datatype="FP32", data=list(range(6))
            )
        ],
        outputs=[RequestOutput(name=name) for name in outputs] if outputs else None,
    )


@pytest.fixture
def loading(monkeypatch):
    events = []
    session = Mock()
    session.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session.get_provider_options.return_value = {
        "CUDAExecutionProvider": {"device_id": "2", "enable_cuda_graph": "1"}
    }
    session.get_inputs.return_value = [
        SimpleNamespace(name="image", shape=[2, 3], type="tensor(float)")
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name=name, shape=[2, 3], type="tensor(float)")
        for name in ("boxes", "scores")
    ]

    def run(names, feed, run_options=None):
        return [feed["image"] * (2 if name == "scores" else 1) for name in names]

    session.run.side_effect = run

    def construct(*args, **kwargs):
        events.append("constructor")
        return session

    constructor = Mock(side_effect=construct)
    monkeypatch.setattr(runtime.ort, "InferenceSession", constructor)
    monkeypatch.setattr(
        runtime, "get_model_uri", AsyncMock(return_value="/unused/model.onnx")
    )
    monkeypatch.setattr(
        runtime.onnx,
        "load",
        Mock(return_value=SimpleNamespace(SerializeToString=lambda: b"mock-model")),
    )
    metadata = {
        "inputs": [MetadataTensor(name="image", datatype="FP32", shape=[2, 3])],
        "outputs": [
            MetadataTensor(name=name, datatype="FP32", shape=[2, 3])
            for name in ("boxes", "scores")
        ],
    }

    def extract(uri):
        events.append("metadata")
        return metadata

    monkeypatch.setattr(runtime, "_extract_metadata", extract)

    class Proxy:
        def __init__(self, wrapped, device):
            events.append("wrap")
            self.wrapped = wrapped
            self.device = device

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    wrap = Mock(side_effect=Proxy)
    monkeypatch.setattr(runtime, "CudaGraphSession", wrap)
    return SimpleNamespace(
        session=session,
        constructor=constructor,
        wrap=wrap,
        events=events,
        metadata=metadata,
    )


def test_graph_setting_defaults_off(monkeypatch):
    monkeypatch.delenv("MLSERVER_MODEL_ONNX_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC", raising=False)
    assert OnnxSettings().cuda_graph is False
    assert OnnxSettings().cuda_blocking_sync is False
    assert get_onnx_settings(settings({})).cuda_graph is False
    assert get_onnx_settings(settings({})).cuda_blocking_sync is False


@pytest.mark.parametrize("file_value", [None, False, True])
def test_graph_setting_env_and_file_precedence(monkeypatch, file_value):
    monkeypatch.setenv("MLSERVER_MODEL_ONNX_CUDA_GRAPH", "true")
    extra = {} if file_value is None else {"cuda_graph": file_value}
    assert get_onnx_settings(settings(extra)).cuda_graph is (
        True if file_value is None else file_value
    )


async def test_graph_options_are_set_before_constructor_and_wrap_before_metadata(
    loading,
):
    original = {"device_id": "2", "use_tf32": "0"}
    model = OnnxModel(
        settings(
            {
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [original, {}],
                "cuda_graph": True,
            }
        )
    )
    assert await model.load()
    assert loading.events == ["constructor", "wrap", "metadata"]
    options = loading.constructor.call_args.kwargs["provider_options"]
    assert options == [
        {"device_id": "2", "use_tf32": "0", "enable_cuda_graph": "1"},
        {},
    ]
    assert original == {"device_id": "2", "use_tf32": "0"}
    assert model._onnx_settings.provider_options == [original, {}]
    loading.wrap.assert_called_once_with(loading.session, 2)
    assert (
        model.inputs == loading.metadata["inputs"]
        and model.outputs == loading.metadata["outputs"]
    )
    assert model._input_names == ["image"] and model._output_names == [
        "boxes",
        "scores",
    ]


async def test_graph_supplies_default_provider_options_without_changing_tf32(loading):
    model = OnnxModel(
        settings({"providers": ["CUDAExecutionProvider"], "cuda_graph": True})
    )
    assert await model.load()
    assert loading.constructor.call_args.kwargs["provider_options"] == [
        {"enable_cuda_graph": "1"}
    ]
    loading.wrap.assert_called_once_with(loading.session, 0)


@pytest.mark.parametrize("options", [None, {}])
async def test_graph_accepts_only_default_run_options(loading, options):
    model = OnnxModel(
        settings(
            {
                "providers": ["CUDAExecutionProvider"],
                "cuda_graph": True,
                "run_options": options,
            }
        )
    )
    assert await model.load()
    assert model._run_options is None


@pytest.mark.parametrize(
    "extra",
    [
        {"providers": ["CPUExecutionProvider"]},
        {"providers": ["CPUExecutionProvider", "CUDAExecutionProvider"]},
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": {"enable_cuda_graph": "0"},
        },
        {
            "providers": ["CUDAExecutionProvider"],
            "run_options": {"log_severity_level": 3},
        },
        {"providers": ["CUDAExecutionProvider"], "provider_options": {"device_id": -1}},
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": {"device_id": True},
        },
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": {"device_id": "bad"},
        },
    ],
)
async def test_unsupported_graph_configuration_fails_before_session_creation(
    loading, extra
):
    model = OnnxModel(settings({**extra, "cuda_graph": True}))
    with pytest.raises(ModelLoadError):
        await model.load()
    loading.constructor.assert_not_called()


@pytest.mark.parametrize("failure", ["fallback", "dynamic", "constructor"])
async def test_unsupported_active_session_is_a_model_load_error(
    loading, monkeypatch, failure
):
    model = OnnxModel(
        settings(
            {
                "providers": ["CUDAExecutionProvider"],
                "provider_options": {"device_id": "2"},
                "cuda_graph": True,
            }
        )
    )
    monkeypatch.setattr(runtime, "CudaGraphSession", cuda.CudaGraphSession)
    if failure == "fallback":
        loading.session.get_providers.return_value = ["CPUExecutionProvider"]
    elif failure == "dynamic":
        loading.session.get_inputs.return_value[0].shape = [None, 3]
    else:
        loading.constructor.side_effect = RuntimeError("unsupported CUDA graph")
    with pytest.raises(ModelLoadError, match="Failed to load ONNX model"):
        await model.load()
    assert "metadata" not in loading.events


async def test_graph_keeps_metadata_prediction_output_order_and_validation(loading):
    model = OnnxModel(
        settings({"providers": ["CUDAExecutionProvider"], "cuda_graph": True})
    )
    assert await model.load()
    response = await model.predict(request(["scores", "boxes"]))
    assert [output.name for output in response.outputs] == ["scores", "boxes"]
    np.testing.assert_array_equal(
        np.array(response.outputs[0].data).reshape(2, 3), np.arange(6).reshape(2, 3) * 2
    )
    assert loading.session.run.call_args.args[0] == ["scores", "boxes"]
    assert loading.session.run.call_args.kwargs["run_options"] is None
    default = await model.predict(request())
    assert default.outputs[0].name == "predict"
    assert loading.session.run.call_args.args[0] == ["boxes"]
    calls = loading.session.run.call_count
    with pytest.raises(InferenceError):
        await model.predict(request(["unknown"]))
    assert loading.session.run.call_count == calls


async def test_graph_disabled_keeps_original_run_options_passthrough(loading):
    loading.session.get_providers.return_value = ["CPUExecutionProvider"]
    model = OnnxModel(
        settings(
            {
                "providers": ["CPUExecutionProvider"],
                "run_options": {"log_severity_level": 3},
            }
        )
    )
    assert await model.load()
    assert model._model is loading.session
    loading.wrap.assert_not_called()
    await model.predict(request())
    assert loading.session.run.call_args.kwargs["run_options"] is model._run_options
    assert model._run_options.log_severity_level == 3


async def test_unload_releases_the_graph_session_owner(loading):
    model = OnnxModel(
        settings({"providers": ["CUDAExecutionProvider"], "cuda_graph": True})
    )
    assert await model.load()
    owned = weakref.ref(model._model)
    assert await model.unload()
    gc.collect()
    assert owned() is None and model._model is None
    assert model.inputs == [] and model.outputs == []


@pytest.fixture
def wait_policy(loading, monkeypatch):
    def configure(device):
        loading.events.append("configure-wait")
        return {
            "library": "/mock/libcudart.so.12",
            "device_id": device,
            "requested_flags": 4,
            "before_session_flags": 12,
        }

    def read(library, device):
        assert loading.session.get_providers.call_count >= 1
        assert loading.session.get_provider_options.call_count >= 1
        loading.events.append("read-wait")
        return 12

    configure_mock, read_mock = Mock(side_effect=configure), Mock(side_effect=read)
    monkeypatch.setattr(runtime, "configure_blocking_sync", configure_mock)
    monkeypatch.setattr(runtime, "read_device_flags", read_mock)
    return SimpleNamespace(configure=configure_mock, read=read_mock)


@pytest.mark.parametrize("file_value", [None, False, True])
def test_blocking_setting_env_and_file_precedence(monkeypatch, file_value):
    monkeypatch.setenv("MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC", "true")
    extra = {} if file_value is None else {"cuda_blocking_sync": file_value}
    assert get_onnx_settings(settings(extra)).cuda_blocking_sync is (
        True if file_value is None else file_value
    )


async def test_blocking_wait_keeps_ordinary_run_options_and_does_not_enable_graph(
    loading, wait_policy
):
    original = {"device_id": "2", "use_tf32": "0", "enable_cuda_graph": "0"}
    model = OnnxModel(
        settings(
            {
                "providers": ["CUDAExecutionProvider"],
                "provider_options": original,
                "cuda_blocking_sync": True,
                "run_options": {"log_severity_level": 3},
            }
        )
    )
    loading.session.get_provider_options.return_value = {
        "CUDAExecutionProvider": original
    }
    assert await model.load()
    assert loading.events == ["configure-wait", "constructor", "read-wait", "metadata"]
    wait_policy.configure.assert_called_once_with(2)
    wait_policy.read.assert_called_once_with("/mock/libcudart.so.12", 2)
    passed = loading.constructor.call_args.kwargs["provider_options"]
    assert passed == [original] and passed[0] is not original
    assert model._onnx_settings.provider_options == original
    loading.wrap.assert_not_called()
    await model.predict(request())
    assert loading.session.run.call_args.kwargs["run_options"] is model._run_options
    assert model._run_options.log_severity_level == 3
    assert model._cuda_wait_policy["after_session_flags"] == 12
    assert await model.unload()
    assert model._cuda_wait_policy is None
    # Unload releases model ownership, not the shared process/device policy.
    wait_policy.configure.assert_called_once_with(2)
    wait_policy.read.assert_called_once_with("/mock/libcudart.so.12", 2)


async def test_combined_wait_is_verified_before_graph_wrapper(loading, wait_policy):
    model = OnnxModel(
        settings(
            {
                "providers": ["CUDAExecutionProvider"],
                "provider_options": {"device_id": 2},
                "cuda_graph": True,
                "cuda_blocking_sync": True,
            }
        )
    )
    assert await model.load()
    assert loading.events == [
        "configure-wait",
        "constructor",
        "read-wait",
        "wrap",
        "metadata",
    ]
    loading.wrap.assert_called_once_with(loading.session, 2)
    assert model._cuda_wait_policy == {
        "library": "/mock/libcudart.so.12",
        "device_id": 2,
        "requested_flags": 4,
        "before_session_flags": 12,
        "after_session_flags": 12,
    }


async def test_default_cpu_load_does_not_configure_or_read_cuda(loading, wait_policy):
    loading.session.get_providers.return_value = ["CPUExecutionProvider"]
    model = OnnxModel(settings({}))
    assert await model.load()
    wait_policy.configure.assert_not_called()
    wait_policy.read.assert_not_called()
    loading.wrap.assert_not_called()
    assert model._cuda_wait_policy is None


@pytest.mark.parametrize(
    "failure", ["configure", "readback", "fallback", "device", "constructor"]
)
async def test_blocking_wait_failure_is_a_load_error_without_verified_receipt(
    loading, wait_policy, failure
):
    model = OnnxModel(
        settings(
            {
                "providers": ["CUDAExecutionProvider"],
                "provider_options": {"device_id": 2},
                "cuda_blocking_sync": True,
            }
        )
    )
    if failure == "configure":
        wait_policy.configure.side_effect = ModelLoadError("missing CUDA runtime")
    elif failure == "readback":
        wait_policy.read.side_effect = ModelLoadError("flags were changed")
    elif failure == "fallback":
        loading.session.get_providers.return_value = ["CPUExecutionProvider"]
    elif failure == "device":
        loading.session.get_provider_options.return_value = {
            "CUDAExecutionProvider": {"device_id": "0"}
        }
    else:
        loading.constructor.side_effect = RuntimeError("provider initialization failed")
    with pytest.raises(ModelLoadError, match="Failed to load ONNX model"):
        await model.load()
    assert model._cuda_wait_policy is None and "metadata" not in loading.events
    loading.wrap.assert_not_called()
    if failure == "configure":
        loading.constructor.assert_not_called()
    if failure != "readback":
        wait_policy.read.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"providers": ["CPUExecutionProvider"]},
        {"providers": ["CUDAExecutionProvider"], "provider_options": {"device_id": -1}},
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": {"device_id": True},
        },
    ],
)
async def test_blocking_configuration_is_validated_before_cuda_calls(
    loading, wait_policy, extra
):
    model = OnnxModel(settings({**extra, "cuda_blocking_sync": True}))
    with pytest.raises(ModelLoadError):
        await model.load()
    wait_policy.configure.assert_not_called()
    loading.constructor.assert_not_called()
