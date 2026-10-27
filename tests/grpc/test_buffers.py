import copy
from collections.abc import AsyncIterator
import gc
from multiprocessing.reduction import ForkingPickler
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc
import numpy as np
import pytest

from mlserver.batching.requests import BatchedRequests
from mlserver.codecs.numpy import NumpyCodec
from mlserver.grpc import dataplane_pb2 as pb
from mlserver.grpc.buffers import convert_request, restore_list_inputs
from mlserver.grpc.converters import ModelInferRequestConverter
from mlserver.grpc import server as grpc_server
from mlserver.grpc.servicers import InferenceServicer
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.model import MLModel
from mlserver.parallel.messages import ModelRequestMessage
from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest, InferenceResponse, Parameters, RequestInput


def request() -> pb.ModelInferRequest:
    return pb.ModelInferRequest(
        model_name="numpy-model",
        model_version="2",
        id="sample-request",
        inputs=[
            pb.ModelInferRequest.InferInputTensor(
                name="images", datatype="FP32", shape=[1, 2]
            )
        ],
        raw_input_contents=[struct.pack("<2f", 1.25, -2.5)],
    )


def test_raw_fp32_uses_readonly_view_and_standard_metadata_conversion() -> None:
    message = request()
    original = ModelInferRequestConverter.to_types
    with patch.object(
        ModelInferRequestConverter, "to_types", wraps=original
    ) as convert:
        converted = convert_request(message)
    data = converted.inputs[0].data.root
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.dtype("<f4")
    assert data.shape == (2,)
    assert not data.flags.writeable and not data.flags.owndata
    np.testing.assert_array_equal(data, [1.25, -2.5])
    convert.assert_called_once()
    delegated = convert.call_args.args[0]
    assert delegated is not message
    assert not delegated.raw_input_contents
    assert delegated.model_name == "numpy-model"
    # Opt-in conversion does not replace the shared converter globally.
    assert isinstance(original(message).inputs[0].data.root, list)


def test_model_and_tensor_metadata_outputs_and_caller_bytes_are_preserved() -> None:
    message = request()
    message.parameters["sequence"].int64_param = 42
    message.parameters["mode"].string_param = "detection"
    message.parameters["disabled"].bool_param = False
    message.inputs[0].parameters["content_type"].string_param = "np"
    message.outputs.add(name="dets").parameters["option"].bool_param = True
    message.outputs.add(name="labels").parameters["count"].int64_param = 0
    before = message.SerializeToString()
    expected = ModelInferRequestConverter.to_types(message)
    converted = convert_request(message)
    assert converted.id == expected.id
    assert converted.parameters == expected.parameters
    assert converted.outputs == expected.outputs
    actual, standard = converted.inputs[0], expected.inputs[0]
    assert actual.name == standard.name
    assert actual.datatype == standard.datatype
    assert actual.shape == standard.shape
    assert actual.parameters == standard.parameters
    assert message.SerializeToString() == before


def test_views_retain_bytes_after_original_protobuf_is_destroyed() -> None:
    message = request()
    converted = convert_request(message)
    del message
    gc.collect()
    data = converted.inputs[0].data.root
    assert isinstance(data, np.ndarray)
    assert isinstance(data.base, bytes)
    np.testing.assert_array_equal(data, [1.25, -2.5])


def test_fp32_bits_survive_pickle_and_worker_numpy_decode() -> None:
    bits = np.array(
        [0, 0x80000000, 0x7FC00001, 0x7F800001, 0x7F800000, 0xFF800000, 1],
        dtype="<u4",
    )
    message = request()
    message.inputs[0].shape[:] = [1, len(bits)]
    message.raw_input_contents[0] = bits.tobytes()
    converted = convert_request(message)
    data = converted.inputs[0].data.root
    assert isinstance(data, np.ndarray)
    np.testing.assert_array_equal(data.view("<u4"), bits)
    envelope = ModelRequestMessage(
        model_name=message.model_name,
        model_version=message.model_version,
        method_name="predict",
        method_args=[converted],
    )
    restored = ForkingPickler.loads(ForkingPickler.dumps(envelope))
    assert restored.model_name == "numpy-model" and restored.model_version == "2"
    array = NumpyCodec.decode_input(restored.method_args[0].inputs[0])
    assert array.shape == (1, len(bits)) and array.flags.writeable
    np.testing.assert_array_equal(array.reshape(-1).view("<u4"), bits)


@pytest.mark.parametrize("shape, values", [([0, 3], []), ([], [1.25])])
def test_zero_length_and_scalar_shapes(shape: list[int], values: list[float]) -> None:
    message = request()
    message.inputs[0].shape[:] = shape
    message.raw_input_contents[0] = struct.pack(f"<{len(values)}f", *values)
    converted = convert_request(message)
    np.testing.assert_array_equal(converted.inputs[0].data.root, values)
    assert NumpyCodec.decode_input(converted.inputs[0]).shape == tuple(shape)


def test_multiple_raw_inputs_preserve_individual_names_shapes_and_values() -> None:
    message = request()
    message.inputs.add(name="second", datatype="FP32", shape=[1])
    message.raw_input_contents.append(struct.pack("<f", 4.5))
    converted = convert_request(message)
    assert [tensor.name for tensor in converted.inputs] == ["images", "second"]
    np.testing.assert_array_equal(
        NumpyCodec.decode_input(converted.inputs[0]), [[1.25, -2.5]]
    )
    np.testing.assert_array_equal(NumpyCodec.decode_input(converted.inputs[1]), [4.5])


@pytest.mark.parametrize(
    "case",
    [
        "no-inputs",
        "no-raw",
        "extra-raw",
        "negative-shape",
        "short-raw",
        "long-raw",
        "typed-contents",
        "different-datatype",
        "mixed-datatypes",
    ],
)
def test_unsupported_inputs_delegate_original_request_and_behavior(case: str) -> None:
    message = request()
    if case == "no-inputs":
        message.ClearField("inputs")
    elif case == "no-raw":
        message.ClearField("raw_input_contents")
    elif case == "extra-raw":
        message.raw_input_contents.append(b"extra")
    elif case == "negative-shape":
        message.inputs[0].shape[0] = -1
    elif case == "short-raw":
        message.raw_input_contents[0] = b"short"
    elif case == "long-raw":
        message.raw_input_contents[0] += b"extra"
    elif case == "typed-contents":
        message.inputs[0].contents.fp32_contents.extend([1.25, -2.5])
    elif case == "different-datatype":
        message.inputs[0].datatype = "INT32"
    else:
        message.inputs.add(name="second", datatype="INT32", shape=[1])
        message.raw_input_contents.append(struct.pack("<i", 4))
    before = message.SerializeToString()
    original = ModelInferRequestConverter.to_types
    try:
        expected = original(copy.deepcopy(message))
    except Exception as error:
        expected_error = error
        with patch.object(
            ModelInferRequestConverter, "to_types", wraps=original
        ) as call:
            with pytest.raises(type(expected_error)) as raised:
                convert_request(message)
        assert str(raised.value) == str(expected_error)
    else:
        with patch.object(
            ModelInferRequestConverter, "to_types", wraps=original
        ) as call:
            converted = convert_request(message)
        assert converted.model_dump() == expected.model_dump()
    call.assert_called_once_with(message)
    assert call.call_args.args[0] is message
    assert message.SerializeToString() == before


def test_metadata_converter_exceptions_propagate_without_mutating_input() -> None:
    message = request()
    before = message.SerializeToString()
    original = Mock(side_effect=RuntimeError("metadata failed"))
    with patch.object(ModelInferRequestConverter, "to_types", original):
        with pytest.raises(RuntimeError, match="metadata failed"):
            convert_request(message)
    assert message.SerializeToString() == before


def context() -> Mock:
    result = Mock(spec=grpc.ServicerContext)
    result.invocation_metadata.return_value = (
        SimpleNamespace(key="x-request", value="forwarded"),
    )
    result.trailing_metadata.return_value = ()
    result.abort = AsyncMock(side_effect=AssertionError("unexpected gRPC abort"))
    return result


def response(payload: InferenceRequest) -> InferenceResponse:
    values = NumpyCodec.decode_input(payload.inputs[0])
    return InferenceResponse(
        id=payload.id,
        model_name="numpy-model",
        model_version="2",
        outputs=[NumpyCodec.encode_output("prediction", values)],
        parameters=Parameters(headers={"X-Response": "received"}),
    )


@pytest.mark.parametrize("enabled", [False, True])
async def test_unary_selection_preserves_payload_headers_and_raw_response(
    enabled: bool,
) -> None:
    data_plane = Mock(spec=DataPlane)
    data_plane.infer = AsyncMock(
        side_effect=lambda **kwargs: response(kwargs["payload"])
    )
    repository = Mock(spec=ModelRepositoryHandlers)
    # The disabled case exercises the public constructor's default.
    servicer = (
        InferenceServicer(data_plane, repository, raw_input_buffers=True)
        if enabled
        else InferenceServicer(data_plane, repository)
    )
    message, ctx = request(), context()
    original = message.SerializeToString()
    result = await servicer.ModelInfer(message, ctx)
    data_plane.infer.assert_awaited_once()
    call = data_plane.infer.call_args.kwargs
    assert (call["name"], call["version"]) == ("numpy-model", "2")
    payload = call["payload"]
    assert isinstance(payload.inputs[0].data.root, np.ndarray if enabled else list)
    assert payload.id == "sample-request"
    assert payload.parameters.headers == {"x-request": "forwarded"}
    assert result.id == "sample-request"
    assert (result.model_name, result.model_version) == ("numpy-model", "2")
    assert len(result.outputs) == 1
    assert result.outputs[0].name == "prediction"
    assert result.outputs[0].datatype == "FP32"
    assert list(result.outputs[0].shape) == [1, 2]
    assert not result.outputs[0].contents.ListFields()
    assert list(result.raw_output_contents) == [struct.pack("<2f", 1.25, -2.5)]
    ctx.set_trailing_metadata.assert_called_once_with((("x-response", "received"),))
    ctx.abort.assert_not_awaited()
    assert message.SerializeToString() == original


async def test_enabled_unary_keeps_typed_inputs_on_standard_conversion() -> None:
    message = request()
    message.ClearField("raw_input_contents")
    message.inputs[0].contents.fp32_contents.extend([1.25, -2.5])
    data_plane = Mock(spec=DataPlane)
    data_plane.infer = AsyncMock(
        side_effect=lambda **kwargs: response(kwargs["payload"])
    )
    servicer = InferenceServicer(
        data_plane, Mock(spec=ModelRepositoryHandlers), raw_input_buffers=True
    )
    result = await servicer.ModelInfer(message, context())
    payload = data_plane.infer.call_args.kwargs["payload"]
    assert isinstance(payload.inputs[0].data.root, list)
    assert list(result.outputs[0].contents.fp32_contents) == [1.25, -2.5]
    assert not result.raw_output_contents


async def test_enabled_buffers_do_not_change_streaming_payload_representation() -> None:
    messages = [request(), request()]
    messages[1].id = "second-request"
    messages[1].raw_input_contents[0] = struct.pack("<2f", 3.0, 4.0)
    payloads_received = []

    async def requests() -> AsyncIterator[pb.ModelInferRequest]:
        for message in messages:
            yield message

    async def infer_stream(
        payloads: AsyncIterator[InferenceRequest], name: str, version: str
    ) -> AsyncIterator[InferenceResponse]:
        assert (name, version) == ("numpy-model", "2")
        async for payload in payloads:
            payloads_received.append(payload)
            yield response(payload)

    data_plane = Mock(spec=DataPlane)
    data_plane.infer_stream.side_effect = infer_stream
    servicer = InferenceServicer(
        data_plane, Mock(spec=ModelRepositoryHandlers), raw_input_buffers=True
    )
    ctx = context()
    results = [item async for item in servicer.ModelStreamInfer(requests(), ctx)]
    assert [item.id for item in results] == ["sample-request", "second-request"]
    assert len(payloads_received) == 2
    for payload in payloads_received:
        assert isinstance(payload.inputs[0].data.root, list)
        assert payload.parameters is not None
        assert payload.parameters.headers == {"x-request": "forwarded"}
    assert list(results[0].raw_output_contents) == [struct.pack("<2f", 1.25, -2.5)]
    assert list(results[1].raw_output_contents) == [struct.pack("<2f", 3.0, 4.0)]
    ctx.abort.assert_not_awaited()


@pytest.mark.parametrize("enabled", [False, True])
async def test_server_setting_reaches_actual_unary_conversion(enabled: bool) -> None:
    settings = Settings(grpc_raw_input_buffers=enabled, metrics_endpoint=None)
    data_plane = Mock(spec=DataPlane)
    data_plane.infer = AsyncMock(
        side_effect=lambda **kwargs: response(kwargs["payload"])
    )
    server = grpc_server.GRPCServer(
        settings, data_plane, Mock(spec=ModelRepositoryHandlers)
    )
    # Construct the actual registered servicer without binding a network port.
    with patch.object(grpc_server.aio, "server", return_value=Mock()), patch.object(
        grpc_server, "ThreadPoolExecutor"
    ):
        server._create_server()
    await server._inference_servicer.ModelInfer(request(), context())
    payload = data_plane.infer.call_args.kwargs["payload"]
    assert isinstance(payload.inputs[0].data.root, np.ndarray if enabled else list)


def test_buffer_setting_defaults_off_and_accepts_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLSERVER_GRPC_RAW_INPUT_BUFFERS", raising=False)
    assert Settings().grpc_raw_input_buffers is False
    monkeypatch.setenv("MLSERVER_GRPC_RAW_INPUT_BUFFERS", "true")
    assert Settings().grpc_raw_input_buffers is True


def test_restored_inputs_merge_through_existing_batcher() -> None:
    messages = [request(), request()]
    messages[1].raw_input_contents[0] = struct.pack("<2f", 3.0, 4.0)
    payloads = [convert_request(message) for message in messages]
    for payload in payloads:
        restore_list_inputs(payload)
    merged = BatchedRequests(
        {"first": payloads[0], "second": payloads[1]}
    ).merged_request
    assert merged.inputs[0].shape == [2, 2]
    assert merged.inputs[0].data.root == [1.25, -2.5, 3.0, 4.0]
    np.testing.assert_array_equal(
        NumpyCodec.decode_input(merged.inputs[0]), [[1.25, -2.5], [3.0, 4.0]]
    )


def test_restore_lists_preserves_metadata_and_nonarray_identity() -> None:
    message = request()
    message.parameters["sequence"].int64_param = 42
    message.inputs[0].parameters["content_type"].string_param = "np"
    message.outputs.add(name="selected-output")
    payload = convert_request(message)
    ordinary = RequestInput(name="ordinary", datatype="INT32", shape=[1], data=[7])
    payload.inputs.append(ordinary)
    original_data = ordinary.data
    original_list = ordinary.data.root
    parameters, outputs = payload.parameters, payload.outputs
    input_parameters = payload.inputs[0].parameters
    restore_list_inputs(payload)
    assert payload.id == "sample-request"
    assert payload.parameters is parameters
    assert payload.outputs is outputs
    assert payload.inputs[0].parameters is input_parameters
    assert payload.inputs[0].name == "images"
    assert payload.inputs[0].datatype == "FP32"
    assert payload.inputs[0].shape == [1, 2]
    assert payload.inputs[0].data.root == [1.25, -2.5]
    assert ordinary.data is original_data and ordinary.data.root is original_list
    restored_data = payload.inputs[0].data
    restore_list_inputs(payload)
    assert payload.inputs[0].data is restored_data
    assert ordinary.data is original_data


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batching", [(0, 0.0), (2, 0.0), (0, 0.01)])
def test_dataplane_restores_only_enabled_batched_inputs_before_middleware(
    data_plane: DataPlane, enabled: bool, batching: tuple[int, float]
) -> None:
    data_plane._settings.grpc_raw_input_buffers = enabled
    model = Mock(spec=MLModel)
    model.settings = ModelSettings(
        name="numpy-model",
        implementation="tests.fixtures.SumModel",
        max_batch_size=batching[0],
        max_batch_time=batching[1],
    )
    payload = convert_request(request())
    compact_data = payload.inputs[0].data
    expected_type = list if enabled and any(batching) else np.ndarray

    def inspect_at_middleware(value: InferenceRequest, settings: ModelSettings) -> None:
        assert value is payload and settings is model.settings
        assert isinstance(value.inputs[0].data.root, expected_type)

    with patch.object(
        data_plane._inference_middleware,
        "request_middleware",
        side_effect=inspect_at_middleware,
    ) as middleware:
        assert data_plane._prepare_payload(payload, model) is payload
    middleware.assert_called_once_with(payload, model.settings)
    if expected_type is np.ndarray:
        assert payload.inputs[0].data is compact_data
    np.testing.assert_array_equal(
        NumpyCodec.decode_input(payload.inputs[0]), [[1.25, -2.5]]
    )
