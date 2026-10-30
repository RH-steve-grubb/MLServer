"""Tests that exercise CUDAExecutionProvider.

Marked ``@pytest.mark.cuda`` so CI without GPU hardware can skip the entire
module with ``pytest -m "not cuda"`` or let them auto-skip via the
``requires_cuda`` guard.
"""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
import pytest

import onnxruntime as ort

from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import Datatype, InferenceRequest, RequestInput

from mlserver_onnx import OnnxModel


def _has_cuda() -> bool:
    """Probe actual CUDA hardware, not just compile-time support."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        from onnx import helper, TensorProto

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "cuda_probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model_proto = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        model_proto.ir_version = 7
        model_bytes = model_proto.SerializeToString()
        session = ort.InferenceSession(model_bytes, providers=["CUDAExecutionProvider"])
        return "CUDAExecutionProvider" in session.get_providers()
    except Exception:
        return False


HAS_CUDA = _has_cuda()
requires_cuda = pytest.mark.skipif(
    not HAS_CUDA, reason="CUDAExecutionProvider not available"
)

pytestmark = [pytest.mark.cuda, requires_cuda]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_model_settings(model_uri: str) -> ModelSettings:
    """Model settings configured for CUDA + CPU fallback."""
    return ModelSettings(
        name="onnx-cuda-model",
        implementation=OnnxModel,
        parameters=ModelParameters(
            uri=model_uri,
            version="v1.0.0",
            extra={
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            },
        ),
    )


@pytest.fixture
async def cuda_model(
    cuda_model_settings: ModelSettings,
) -> AsyncGenerator[OnnxModel, None]:
    """Loaded model with CUDA provider."""
    model = OnnxModel(cuda_model_settings)
    model.ready = await model.load()
    yield model
    await model.unload()


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------


async def test_cuda_device_available():
    """ONNX Runtime reports a GPU device."""
    assert ort.get_device() == "GPU"


async def test_cuda_provider_active(cuda_model: OnnxModel):
    """Session actually uses CUDAExecutionProvider when loaded."""
    active = cuda_model._model.get_providers()
    assert "CUDAExecutionProvider" in active


# ---------------------------------------------------------------------------
# Inference on GPU
# ---------------------------------------------------------------------------


async def test_predict_on_cuda(cuda_model: OnnxModel):
    """Inference through CUDAExecutionProvider produces correct results."""
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input",
                datatype=Datatype.FP32,
                shape=[2, 4],
                data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
        ]
    )
    response = await cuda_model.predict(request)

    assert len(response.outputs) == 1
    assert response.outputs[0].shape == [2, 4]
    result = np.array(response.outputs[0].data).reshape(2, 4)
    expected = np.array([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
    np.testing.assert_allclose(result, expected)


async def test_parallel_predict_on_cuda(cuda_model: OnnxModel):
    """Multiple concurrent inferences through CUDA complete correctly."""

    def _make_request() -> InferenceRequest:
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name="input",
                    datatype=Datatype.FP32,
                    shape=[1, 4],
                    data=[10.0, 20.0, 30.0, 40.0],
                )
            ]
        )

    tasks = [cuda_model.predict(_make_request()) for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert len(resp.outputs) == 1
        result = np.array(resp.outputs[0].data).reshape(1, 4)
        expected = np.array([[11.0, 21.0, 31.0, 41.0]])
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Provider options
# ---------------------------------------------------------------------------


async def test_cuda_provider_with_device_id(model_uri: str):
    """CUDAExecutionProvider accepts device_id option."""
    settings = ModelSettings(
        name="onnx-cuda-device0",
        implementation=OnnxModel,
        parameters=ModelParameters(
            uri=model_uri,
            version="v1.0.0",
            extra={
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [{"device_id": "0"}, {}],
            },
        ),
    )
    model = OnnxModel(settings)
    model.ready = await model.load()
    try:
        assert model.ready
        assert "CUDAExecutionProvider" in model._model.get_providers()
    finally:
        await model.unload()


async def test_cuda_provider_with_arena_extend_strategy(model_uri: str):
    """CUDAExecutionProvider accepts arena_extend_strategy option."""
    settings = ModelSettings(
        name="onnx-cuda-arena",
        implementation=OnnxModel,
        parameters=ModelParameters(
            uri=model_uri,
            version="v1.0.0",
            extra={
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [
                    {"arena_extend_strategy": "kSameAsRequested"},
                    {},
                ],
            },
        ),
    )
    model = OnnxModel(settings)
    model.ready = await model.load()
    try:
        assert model.ready
    finally:
        await model.unload()


async def test_invalid_device_id_falls_back_to_cpu(model_uri: str):
    """Invalid device_id triggers ORT's silent fallback to CPUExecutionProvider.

    ORT >= 1.20 silently ignores an invalid device_id and falls back to CPU
    rather than raising.  If a future ORT version changes this behavior to
    raise, this test will fail with ModelLoadError and should be converted
    to ``pytest.raises(ModelLoadError)`` instead.
    """
    settings = ModelSettings(
        name="onnx-cuda-bad-device",
        implementation=OnnxModel,
        parameters=ModelParameters(
            uri=model_uri,
            version="v1.0.0",
            extra={
                "providers": ["CUDAExecutionProvider"],
                "provider_options": [{"device_id": "999"}],
            },
        ),
    )
    model = OnnxModel(settings)
    model.ready = await model.load()
    try:
        assert model.ready
        active = model._model.get_providers()
        assert "CPUExecutionProvider" in active
        assert "CUDAExecutionProvider" not in active
    finally:
        await model.unload()


# ---------------------------------------------------------------------------
# CUDA-only (no CPU fallback)
# ---------------------------------------------------------------------------


async def test_cuda_only_provider(model_uri: str):
    """Model loads with CUDAExecutionProvider when it is the sole requested provider.

    ORT always appends CPUExecutionProvider as a fallback, so we only
    verify that CUDAExecutionProvider is the *first* (primary) provider.
    """
    settings = ModelSettings(
        name="onnx-cuda-only",
        implementation=OnnxModel,
        parameters=ModelParameters(
            uri=model_uri,
            version="v1.0.0",
            extra={
                "providers": ["CUDAExecutionProvider"],
            },
        ),
    )
    model = OnnxModel(settings)
    model.ready = await model.load()
    try:
        assert model.ready
        active = model._model.get_providers()
        assert active[0] == "CUDAExecutionProvider"
    finally:
        await model.unload()


# ---------------------------------------------------------------------------
# Metadata consistency
# ---------------------------------------------------------------------------


async def test_metadata_matches_cpu(cuda_model: OnnxModel, model: OnnxModel):
    """CUDA-loaded model exposes the same metadata as CPU-loaded model."""
    cuda_meta = await cuda_model.metadata()
    cpu_meta = await model.metadata()

    assert cuda_meta.inputs == cpu_meta.inputs
    assert cuda_meta.outputs == cpu_meta.outputs
