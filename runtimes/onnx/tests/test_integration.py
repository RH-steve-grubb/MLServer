import pytest

from mlserver.grpc.converters import (
    ModelInferRequestConverter,
    ModelInferResponseConverter,
)
from mlserver.types import RequestOutput


@pytest.mark.asyncio
async def test_rest_infer_e2e(rest_client, inference_request, model_settings):
    """REST inference returns expected output shape."""
    response = await rest_client.infer(model_settings.name, inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].shape == inference_request.inputs[0].shape


@pytest.mark.asyncio
async def test_grpc_infer_raw_bytes(grpc_stub, inference_request, model_settings):
    """gRPC inference with raw bytes returns expected values."""
    expected = [value + 1.0 for value in inference_request.inputs[0].data]
    pb_request = ModelInferRequestConverter.from_types(
        inference_request,
        model_name=model_settings.name,
        model_version=model_settings.parameters.version,
        use_raw=True,
    )

    assert pb_request.raw_input_contents

    pb_response = await grpc_stub.ModelInfer(pb_request)
    response = ModelInferResponseConverter.to_types(pb_response)

    assert len(response.outputs) == 1
    assert response.outputs[0].shape == inference_request.inputs[0].shape

    output_data = getattr(response.outputs[0].data, "root", response.outputs[0].data)
    assert list(output_data) == expected


@pytest.mark.asyncio
async def test_rest_infer_named_output(rest_client, inference_request, model_settings):
    """REST inference can request a named output."""
    inference_request.outputs = [RequestOutput(name="output")]

    response = await rest_client.infer(model_settings.name, inference_request)

    assert len(response.outputs) == 1
    assert response.outputs[0].name == "output"


@pytest.mark.asyncio
async def test_multi_model_rest(
    rest_client_multi_model, model_settings, inference_request
):
    """Multi-model REST server serves multiple ONNX models."""
    onnx_response = await rest_client_multi_model.infer(
        model_settings.name, inference_request
    )
    multi_output_response = await rest_client_multi_model.infer(
        "multi-output-model", inference_request
    )

    assert len(onnx_response.outputs) == 1
    assert len(multi_output_response.outputs) == 1
