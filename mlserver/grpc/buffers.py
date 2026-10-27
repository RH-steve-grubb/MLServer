"""Opt-in raw FP32 conversion for runtimes accepting NumPy-backed tensor data.

The wire protocol stays unchanged. NumPy-compatible workers can keep raw input
buffers compact through IPC instead of constructing Python float lists in the
frontend. Other runtimes must retain the standard request converter.
"""

import math

import numpy as np

from ..types import InferenceRequest, TensorData
from . import dataplane_pb2 as pb
from .converters import ModelInferRequestConverter


def restore_list_inputs(payload: InferenceRequest) -> None:
    """Restore the public list representation before list-based batching."""
    for tensor in payload.inputs:
        data = getattr(tensor.data, "root", tensor.data)
        if isinstance(data, np.ndarray):
            # TensorData is flat even when the logical tensor has dimensions.
            # Only the data representation changes; retain input metadata.
            tensor.data = TensorData(data.reshape(-1).tolist())


def convert_request(request: pb.ModelInferRequest) -> InferenceRequest:
    """Preserve complete raw FP32 inputs, delegating unsupported requests."""
    raw_contents = request.raw_input_contents
    if not request.inputs or len(raw_contents) != len(request.inputs):
        return ModelInferRequestConverter.to_types(request)
    for tensor, raw in zip(request.inputs, raw_contents):
        if (
            tensor.datatype != "FP32"
            or tensor.contents.ListFields()
            or any(dimension < 0 for dimension in tensor.shape)
            or math.prod(tensor.shape) * 4 != len(raw)
        ):
            return ModelInferRequestConverter.to_types(request)

    # Keep metadata conversion identical to the standard path. Work on a copy
    # so callers retain their original protobuf and its raw payload fields.
    metadata = pb.ModelInferRequest()
    metadata.CopyFrom(request)
    metadata.ClearField("raw_input_contents")
    converted = ModelInferRequestConverter.to_types(metadata)
    for converted_tensor, raw in zip(converted.inputs, raw_contents):
        # The read-only view retains its bytes owner. model_construct avoids
        # validating it back into a float list; the worker's NumPy codec makes
        # the writable, correctly shaped array expected by the runtime.
        converted_tensor.data = TensorData.model_construct(
            root=np.frombuffer(raw, dtype="<f4")
        )
    return converted
