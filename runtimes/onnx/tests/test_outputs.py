"""Tests for ONNX model output handling."""

import pytest

from mlserver_onnx import OnnxModel
from mlserver_onnx.onnx import PREDICT_OUTPUT
from mlserver.errors import InferenceError
from mlserver.types import RequestOutput


async def test_invalid_output_error(model: OnnxModel, inference_request):
    """Test that requesting an invalid output raises an error."""
    inference_request.outputs = [RequestOutput(name="invalid_output")]

    with pytest.raises(InferenceError) as exc_info:
        await model.predict(inference_request)

    assert "OnnxModel only supports" in str(exc_info.value)


async def test_multi_output_predict_default(
    multi_output_model: OnnxModel, inference_request
):
    """Test multi-output model with default output (returns first output)."""
    response = await multi_output_model.predict(inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == PREDICT_OUTPUT


async def test_multi_output_predict_specific(
    multi_output_model: OnnxModel, inference_request
):
    """Test multi-output model requesting specific outputs."""
    inference_request.outputs = [
        RequestOutput(name="output1"),
        RequestOutput(name="output2"),
    ]

    response = await multi_output_model.predict(inference_request)

    assert len(response.outputs) == 2
    assert response.outputs[0].name == "output1"
    assert response.outputs[1].name == "output2"

    output1_data = response.outputs[0].data
    output2_data = response.outputs[1].data
    expected_output1 = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    expected_output2 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for actual, exp in zip(output1_data, expected_output1):
        assert abs(actual - exp) < 0.001

    for actual, exp in zip(output2_data, expected_output2):
        assert abs(actual - exp) < 0.001


async def test_multi_output_predict_mixed(
    multi_output_model: OnnxModel, inference_request
):
    """Test multi-output model with mixed output requests."""
    inference_request.outputs = [
        RequestOutput(name=PREDICT_OUTPUT),
        RequestOutput(name="output2"),
    ]

    response = await multi_output_model.predict(inference_request)

    assert len(response.outputs) == 2
    assert response.outputs[0].name == PREDICT_OUTPUT
    assert response.outputs[1].name == "output2"


async def test_multi_output_single_named_output(
    multi_output_model: OnnxModel, inference_request
):
    """Test requesting single named output from multi-output model."""
    inference_request.outputs = [RequestOutput(name="output2")]

    response = await multi_output_model.predict(inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == "output2"
