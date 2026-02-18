"""Tests for ONNX error handling and edge cases."""

import os
import onnx
import pytest
from onnx import TensorProto, helper

from mlserver_onnx import OnnxModel
from mlserver.errors import InvalidModelURI, ModelLoadError
from mlserver.settings import ModelSettings
from mlserver.types import RequestInput


async def test_extra_inputs_ignored(model: OnnxModel, inference_request):
    """Extra inputs beyond the model's required names are ignored (name-based)."""
    inference_request.inputs.append(
        RequestInput(
            name="input-1", shape=[2, 4], data=[0, 1, 2, 3, 4, 5, 6, 7], datatype="FP32"
        )
    )

    # Name-based: only required "input" used; extra "input-1" ignored
    response = await model.predict(inference_request)
    assert len(response.outputs) == 1
    assert response.outputs[0].data is not None


async def test_model_no_inputs_error(tmp_path, model_settings: ModelSettings):
    """Test that model without inputs raises error on load."""
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([], "no_inputs", [], [output_tensor])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=9
    )
    model_uri = os.path.join(tmp_path, "no-inputs.onnx")
    onnx.save(model, model_uri)

    model_settings.parameters.uri = model_uri  # type: ignore
    onnx_model = OnnxModel(model_settings)
    with pytest.raises(ModelLoadError) as excinfo:
        await onnx_model.load()

    # Either our "no inputs" check or ONNX Runtime's validation
    assert "no inputs" in str(excinfo.value) or "Failed to load ONNX model" in str(
        excinfo.value
    )


async def test_model_no_outputs_error(tmp_path, model_settings: ModelSettings):
    """Test that model without outputs raises error during prediction."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([], "no_outputs", [input_tensor], [])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=9
    )
    model_uri = os.path.join(tmp_path, "no-outputs.onnx")
    onnx.save(model, model_uri)

    model_settings.parameters.uri = model_uri  # type: ignore
    onnx_model = OnnxModel(model_settings)
    with pytest.raises(ModelLoadError) as excinfo:
        await onnx_model.load()

    # ONNX Runtime validates model before we can check, so accept its error message
    assert "Failed to load ONNX model" in str(excinfo.value)


async def test_corrupted_model_load_error(model_settings, invalid_model_uri):
    """Test that loading corrupted model raises appropriate error."""
    model_settings.parameters.uri = invalid_model_uri  # type: ignore
    model = OnnxModel(model_settings)

    with pytest.raises(ModelLoadError) as excinfo:
        await model.load()

    assert "Failed to load ONNX model" in str(excinfo.value)


async def test_missing_model_file_error(tmp_path, model_settings: ModelSettings):
    """Test that missing model file raises error."""
    missing_path = os.path.join(tmp_path, "does-not-exist.onnx")
    model_settings.parameters.uri = missing_path  # type: ignore
    model = OnnxModel(model_settings)

    # get_model_uri raises InvalidModelURI for missing files
    with pytest.raises(InvalidModelURI):
        await model.load()
