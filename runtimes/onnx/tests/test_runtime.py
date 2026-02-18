import asyncio
import pytest

from mlserver.codecs import CodecError
from mlserver.errors import InferenceError
from mlserver.types import Datatype, InferenceRequest, MetadataTensor, RequestInput
from mlserver_onnx import OnnxModel


async def test_load(model: OnnxModel):
    """Model loads and exposes a session."""
    assert model.ready
    assert model._model is not None


async def test_unload(model: OnnxModel):
    """Unload clears session and cached names."""
    assert await model.unload()
    assert model._model is None
    assert model._output_names == []
    assert model._input_names == []


async def test_predict(model: OnnxModel, inference_request: InferenceRequest):
    """Predict returns one output with expected shape."""
    response = await model.predict(inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].shape == inference_request.inputs[0].shape


async def test_metadata_from_model(model: OnnxModel):
    """Metadata matches model inputs and outputs."""
    metadata = await model.metadata()

    expected_inputs = [
        MetadataTensor(name="input", datatype=Datatype.FP32, shape=[-1, 4])
    ]
    expected_outputs = [
        MetadataTensor(name="output", datatype=Datatype.FP32, shape=[-1, 4])
    ]

    assert metadata.inputs == expected_inputs
    assert metadata.outputs == expected_outputs


async def test_metadata_multi_output(multi_output_model: OnnxModel):
    """Metadata returns all output tensors for multi-output models."""
    metadata = await multi_output_model.metadata()

    expected_outputs = [
        MetadataTensor(name="output1", datatype=Datatype.FP32, shape=[-1, 4]),
        MetadataTensor(name="output2", datatype=Datatype.FP32, shape=[-1, 4]),
    ]

    assert metadata.outputs == expected_outputs


async def test_inputs_outputs_populated(model: OnnxModel):
    """inputs/outputs are populated after load."""
    assert model.inputs == [
        MetadataTensor(name="input", datatype=Datatype.FP32, shape=[-1, 4])
    ]
    assert model.outputs == [
        MetadataTensor(name="output", datatype=Datatype.FP32, shape=[-1, 4])
    ]


async def test_invalid_request_shape(model: OnnxModel):
    """Invalid input shape raises codec error."""
    request = InferenceRequest(
        inputs=[
            RequestInput(name="input", datatype=Datatype.FP32, shape=[3], data=[1, 2])
        ]
    )

    with pytest.raises(CodecError):
        await model.predict(request)


async def test_invalid_request_length(model: OnnxModel):
    """Invalid input length raises codec error."""
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input",
                datatype=Datatype.FP32,
                shape=[2, 4],
                data=[1.0, 2.0, 3.0],
            )
        ]
    )

    with pytest.raises(CodecError):
        await model.predict(request)


async def test_invalid_request_datatype(model: OnnxModel):
    """Invalid input datatype raises codec error."""
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input",
                datatype=Datatype.FP32,
                shape=[1],
                data=["not-a-number"],
            )
        ]
    )

    with pytest.raises(CodecError):
        await model.predict(request)


async def test_missing_inputs(model: OnnxModel):
    """Test that missing inputs raises error (name-based: missing required names)."""
    request = InferenceRequest(inputs=[])

    with pytest.raises(InferenceError) as exc_info:
        await model.predict(request)

    assert "Missing:" in str(exc_info.value)
    assert "input" in str(exc_info.value)


async def test_parallel_predict(model: OnnxModel, inference_request):
    """Parallel predictions complete successfully."""
    tasks = [model.predict(inference_request) for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    assert all(len(response.outputs) == 1 for response in responses)
