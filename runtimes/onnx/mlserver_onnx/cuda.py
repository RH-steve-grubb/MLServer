"""Persistent CUDA Graph buffers for explicitly enabled fixed-shape models."""

from threading import Lock
from typing import Any

import numpy as np
import onnxruntime as ort

from mlserver.errors import ModelLoadError


def _shape(node: Any) -> tuple[int, ...]:
    if (
        node.type != "tensor(float)"
        or not isinstance(node.shape, (list, tuple))
        or any(type(dimension) is not int or dimension <= 0 for dimension in node.shape)
    ):
        raise ModelLoadError(
            "CUDA Graphs require fixed-shape FP32 inputs and outputs "
            f"(unsupported tensor: {node.name})"
        )
    return tuple(node.shape)


class CudaGraphSession:
    """Replay one graph with worker-owned addresses and fresh request contents.

    Calls retain ordinary CPU NumPy outputs. One lock protects input upload,
    graph execution and output downloads together: concurrent callers must not
    overwrite the persistent device input or read another request's outputs.
    """

    def __init__(self, session: Any, device_id: int = 0) -> None:
        if type(device_id) is not int or device_id < 0:
            raise ModelLoadError("CUDA Graph device_id must be a non-negative integer")
        providers = session.get_providers()
        options = session.get_provider_options().get("CUDAExecutionProvider", {})
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise ModelLoadError("CUDA Graphs require CUDAExecutionProvider first")
        if str(options.get("device_id")) != str(device_id):
            raise ModelLoadError("CUDA Graph session uses a different CUDA device")
        if str(options.get("enable_cuda_graph")) != "1":
            raise ModelLoadError("CUDA Graphs were not enabled on the CUDA provider")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1:
            raise ModelLoadError("CUDA Graphs require exactly one FP32 model input")
        if not outputs or len({node.name for node in outputs}) != len(outputs):
            raise ModelLoadError("CUDA Graphs require distinct, nonempty model outputs")
        self._shape = _shape(inputs[0])
        output_shapes = [_shape(node) for node in outputs]
        self._input_name = inputs[0].name
        self._output_names = [node.name for node in outputs]
        self._session = session
        self._lock = Lock()
        session.disable_fallback()
        try:
            self._binding = session.io_binding()
            self._device_input = ort.OrtValue.ortvalue_from_numpy(
                np.zeros(self._shape, dtype=np.float32), "cuda", device_id
            )
            self._binding.bind_ortvalue_input(self._input_name, self._device_input)
            # Retain every OrtValue, not just its pointer in the binding. Graph
            # replay requires these addresses to remain live across requests.
            self._device_outputs = [
                ort.OrtValue.ortvalue_from_shape_and_type(
                    shape, np.float32, "cuda", device_id
                )
                for shape in output_shapes
            ]
            for name, value in zip(self._output_names, self._device_outputs):
                self._binding.bind_ortvalue_output(name, value)
        except Exception as error:
            raise ModelLoadError(
                f"Unable to initialize CUDA Graph device buffers: {error}"
            ) from error

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
        run_options: Any = None,
    ) -> list[np.ndarray]:
        if run_options is not None:
            raise ValueError("CUDA Graphs require run_options=None")
        if set(input_feed) != {self._input_name}:
            raise ValueError("CUDA Graphs require the model's single input tensor")
        tensor = input_feed[self._input_name]
        if (
            not isinstance(tensor, np.ndarray)
            or tensor.shape != self._shape
            or tensor.dtype != np.float32
            or not tensor.flags.c_contiguous
        ):
            raise ValueError(
                "CUDA Graph input must be contiguous FP32 with the model's fixed shape"
            )
        # ORT treats both None and [] as requesting all model outputs.
        names = self._output_names if not output_names else output_names
        if any(name not in self._output_names for name in names):
            raise ValueError("Unknown CUDA Graph model output requested")
        # Keep all outputs bound so selecting a subset does not change the
        # captured graph or its device addresses between requests.
        with self._lock:
            # Reusing a host ndarray does not mean its contents are unchanged.
            self._device_input.update_inplace(tensor)
            self._binding.synchronize_inputs()
            self._session.run_with_iobinding(self._binding)
            self._binding.synchronize_outputs()
            values = self._binding.copy_outputs_to_cpu()
            if len(values) != len(self._output_names):
                raise RuntimeError("CUDA Graph did not return every bound model output")
            by_name = dict(zip(self._output_names, values))
            return [by_name[name] for name in names]
