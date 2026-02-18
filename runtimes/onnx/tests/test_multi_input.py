"""Tests for ONNX multi-input model support."""

import os
import onnx
import pytest
from onnx import TensorProto, helper

from mlserver_onnx import OnnxModel
from mlserver.errors import InferenceError
from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import InferenceRequest, RequestInput, Datatype


@pytest.fixture
def multi_input_model_uri(tmp_path) -> str:
    """Create an ONNX model with multiple inputs (e.g., encoder-decoder style)."""
    input1_tensor = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [None, 4]
    )
    input2_tensor = helper.make_tensor_value_info(
        "input2", TensorProto.FLOAT, [None, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 4]
    )

    # Create a simple model that adds both inputs
    add_node = helper.make_node(
        "Add",
        inputs=["input1", "input2"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [add_node],
        "multi_input_model",
        [input1_tensor, input2_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=9,
    )

    model_uri = os.path.join(tmp_path, "multi-input-model.onnx")
    onnx.save(model, model_uri)
    return model_uri


@pytest.fixture
def multi_input_model_settings(multi_input_model_uri: str) -> ModelSettings:
    """Model settings for multi-input model."""
    return ModelSettings(
        name="multi-input-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=multi_input_model_uri, version="v1.0.0"),
    )


@pytest.fixture
async def multi_input_model(
    multi_input_model_settings: ModelSettings,
) -> OnnxModel:
    """Loaded multi-input ONNX model."""
    model = OnnxModel(multi_input_model_settings)
    model.ready = await model.load()
    return model


@pytest.fixture
def multi_input_request() -> InferenceRequest:
    """Sample inference request with multiple inputs."""
    return InferenceRequest(
        inputs=[
            RequestInput(
                name="input1",
                datatype=Datatype.FP32,
                shape=[2, 4],
                data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ),
            RequestInput(
                name="input2",
                datatype=Datatype.FP32,
                shape=[2, 4],
                data=[0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            ),
        ]
    )


async def test_multi_input_load(multi_input_model: OnnxModel):
    """Test that multi-input model loads correctly."""
    assert multi_input_model.ready
    assert multi_input_model._model is not None
    assert len(multi_input_model._input_names) == 2
    assert "input1" in multi_input_model._input_names
    assert "input2" in multi_input_model._input_names


async def test_multi_input_predict(
    multi_input_model: OnnxModel, multi_input_request: InferenceRequest
):
    """Test prediction with multiple inputs."""
    response = await multi_input_model.predict(multi_input_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == "predict"
    assert response.outputs[0].shape == [2, 4]

    # Verify output is sum of inputs
    output_data = response.outputs[0].data
    expected = [1.5, 2.5, 3.5, 4.5, 6.0, 7.0, 8.0, 9.0]

    for actual, exp in zip(output_data, expected):
        assert abs(actual - exp) < 0.001


async def test_multi_input_wrong_count(
    multi_input_model: OnnxModel, inference_request: InferenceRequest
):
    """Missing required inputs raises error (name-based)."""
    # request has 1 input "input"; model expects "input1" and "input2"
    with pytest.raises(InferenceError) as exc_info:
        await multi_input_model.predict(inference_request)

    assert "Missing:" in str(exc_info.value)
    assert "input1" in str(exc_info.value)
    assert "input2" in str(exc_info.value)


async def test_multi_input_metadata(multi_input_model: OnnxModel):
    """Test that metadata correctly reflects multiple inputs."""
    metadata = await multi_input_model.metadata()

    assert metadata.inputs is not None
    assert len(metadata.inputs) == 2
    assert metadata.inputs[0].name == "input1"
    assert metadata.inputs[1].name == "input2"
    # Datatype is serialized as string in metadata
    assert metadata.inputs[0].datatype == "FP32"
    assert metadata.inputs[1].datatype == "FP32"


async def test_multi_input_order_independent(
    multi_input_model: OnnxModel,
):
    """Request input order does not matter when names match (name-based)."""
    # Send inputs in order input1, input2
    request1 = InferenceRequest(
        inputs=[
            RequestInput(
                name="input1",
                datatype=Datatype.FP32,
                shape=[1, 4],
                data=[1.0, 2.0, 3.0, 4.0],
            ),
            RequestInput(
                name="input2",
                datatype=Datatype.FP32,
                shape=[1, 4],
                data=[1.0, 1.0, 1.0, 1.0],
            ),
        ]
    )
    response1 = await multi_input_model.predict(request1)
    output1 = response1.outputs[0].data

    # Send same inputs in reversed order (input2, input1)
    request2 = InferenceRequest(
        inputs=[
            RequestInput(
                name="input2",
                datatype=Datatype.FP32,
                shape=[1, 4],
                data=[1.0, 1.0, 1.0, 1.0],
            ),
            RequestInput(
                name="input1",
                datatype=Datatype.FP32,
                shape=[1, 4],
                data=[1.0, 2.0, 3.0, 4.0],
            ),
        ]
    )
    response2 = await multi_input_model.predict(request2)
    output2 = response2.outputs[0].data

    # Results should be identical (mapping is by name, not position)
    assert output1 == output2
    expected = [2.0, 3.0, 4.0, 5.0]
    for actual, exp in zip(output1, expected):
        assert abs(actual - exp) < 0.001


async def test_multi_input_missing_one_raises(
    multi_input_model: OnnxModel,
):
    """Test that providing only one of two required inputs raises InferenceError."""
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input1",
                datatype=Datatype.FP32,
                shape=[1, 4],
                data=[1.0, 2.0, 3.0, 4.0],
            ),
        ]
    )
    with pytest.raises(InferenceError) as exc_info:
        await multi_input_model.predict(request)

    assert "Missing:" in str(exc_info.value)
    assert "input2" in str(exc_info.value)


@pytest.fixture
def three_input_model_uri(tmp_path) -> str:
    """Create an ONNX model with three inputs for comprehensive testing."""
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [None, 2])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [None, 2])
    input3 = helper.make_tensor_value_info("input3", TensorProto.FLOAT, [None, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])

    # output = input1 + input2 + input3
    add1 = helper.make_node("Add", inputs=["input1", "input2"], outputs=["temp"])
    add2 = helper.make_node("Add", inputs=["temp", "input3"], outputs=["output"])

    graph = helper.make_graph(
        [add1, add2],
        "three_input_model",
        [input1, input2, input3],
        [output],
    )

    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=9,
    )

    model_uri = os.path.join(tmp_path, "three-input-model.onnx")
    onnx.save(model, model_uri)
    return model_uri


async def test_three_input_model(three_input_model_uri: str):
    """Test model with three inputs to verify scalability."""
    settings = ModelSettings(
        name="three-input-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=three_input_model_uri, version="v1.0.0"),
    )

    model = OnnxModel(settings)
    model.ready = await model.load()

    assert len(model._input_names) == 3

    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input1",
                datatype=Datatype.FP32,
                shape=[1, 2],
                data=[1.0, 2.0],
            ),
            RequestInput(
                name="input2",
                datatype=Datatype.FP32,
                shape=[1, 2],
                data=[3.0, 4.0],
            ),
            RequestInput(
                name="input3",
                datatype=Datatype.FP32,
                shape=[1, 2],
                data=[5.0, 6.0],
            ),
        ]
    )

    response = await model.predict(request)

    assert len(response.outputs) == 1
    # Should be 1+3+5=9, 2+4+6=12
    expected = [9.0, 12.0]
    for actual, exp in zip(response.outputs[0].data, expected):
        assert abs(actual - exp) < 0.001


async def test_multi_input_different_shapes(tmp_path):
    """Test multi-input model with different shaped inputs."""
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [None, 4])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [None, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])

    # Simple concatenation-like operation (just pass through input1)
    identity = helper.make_node("Identity", inputs=["input1"], outputs=["output"])

    graph = helper.make_graph(
        [identity],
        "different_shapes",
        [input1, input2],
        [output],
    )

    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=9,
    )

    model_uri = os.path.join(tmp_path, "different-shapes.onnx")
    onnx.save(model, model_uri)

    settings = ModelSettings(
        name="different-shapes-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=model_uri, version="v1.0.0"),
    )

    onnx_model = OnnxModel(settings)
    onnx_model.ready = await onnx_model.load()

    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input1",
                datatype=Datatype.FP32,
                shape=[1, 4],
                data=[1.0, 2.0, 3.0, 4.0],
            ),
            RequestInput(
                name="input2",
                datatype=Datatype.FP32,
                shape=[1, 2],
                data=[5.0, 6.0],
            ),
        ]
    )

    response = await onnx_model.predict(request)
    assert len(response.outputs) == 1
    assert response.outputs[0].shape == [1, 4]
