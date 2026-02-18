"""Tests for ONNX prediction functionality."""

import asyncio

from mlserver_onnx import OnnxModel
from mlserver_onnx.onnx import PREDICT_OUTPUT
from mlserver.types import RequestOutput


async def test_predict_default_output(model: OnnxModel, inference_request):
    """Test prediction with default output (should use PREDICT_OUTPUT)."""
    response = await model.predict(inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == PREDICT_OUTPUT
    assert response.outputs[0].shape == inference_request.inputs[0].shape


async def test_predict_explicit_output(model: OnnxModel, inference_request):
    """Test prediction with explicitly named output."""
    inference_request.outputs = [RequestOutput(name=PREDICT_OUTPUT)]

    response = await model.predict(inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == PREDICT_OUTPUT


async def test_predict_model_output_name(model: OnnxModel, inference_request):
    """Test prediction using actual model output name."""
    inference_request.outputs = [RequestOutput(name="output")]

    response = await model.predict(inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == "output"


async def test_predict_values(model: OnnxModel, inference_request):
    """Test that predictions return correct values (input + 1)."""
    response = await model.predict(inference_request)

    assert len(response.outputs) == 1
    output_data = response.outputs[0].data
    expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    for i, (actual, exp) in enumerate(zip(output_data, expected)):
        assert abs(actual - exp) < 0.001, f"Output {i}: {actual} != {exp}"


async def test_parallel_predict(model: OnnxModel, inference_request):
    """Test that model can handle parallel predictions."""
    tasks = [model.predict(inference_request) for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    assert all(len(response.outputs) == 1 for response in responses)
    # Verify all predictions return same result
    first_output = responses[0].outputs[0].data
    for response in responses[1:]:
        assert response.outputs[0].data == first_output
