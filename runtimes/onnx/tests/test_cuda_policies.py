"""Real-device validation of graph replay and CUDA blocking synchronization."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from mlserver.settings import ModelParameters, ModelSettings
from mlserver.types import InferenceRequest, RequestInput, RequestOutput
from mlserver_onnx import OnnxModel
from mlserver_onnx.cuda import CudaGraphSession
from mlserver_onnx.cuda_sync import read_device_flags

from .test_cuda import requires_cuda

pytestmark = [pytest.mark.cuda, requires_cuda]


@pytest.fixture
def fixed_cuda_model_uri(tmp_path) -> str:
    """A fixed-shape model whose MatMul/Add nodes both support CUDA graphs."""
    weights = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weights"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["shifted"]),
        ],
        "fixed_cuda_policies",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4])
            for name in ("linear", "shifted")
        ],
        initializer=[
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path = tmp_path / "fixed-cuda.onnx"
    onnx.save(model, path)
    return str(path)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("blocking", [False, True])
async def test_cuda_policies_replay_fresh_inputs_and_preserve_outputs(
    fixed_cuda_model_uri: str, graph: bool, blocking: bool
) -> None:
    model = OnnxModel(
        ModelSettings(
            name="cuda-policies",
            implementation=OnnxModel,
            parameters=ModelParameters(
                uri=fixed_cuda_model_uri,
                extra={
                    "providers": ["CUDAExecutionProvider"],
                    "provider_options": [{"device_id": "0", "use_tf32": "0"}],
                    "cuda_graph": graph,
                    "cuda_blocking_sync": blocking,
                },
            ),
        )
    )
    assert await model.load()
    try:
        assert model._model.get_providers()[0] == "CUDAExecutionProvider"
        assert isinstance(model._model, CudaGraphSession) == graph
        if blocking:
            receipt = model._cuda_wait_policy
            assert receipt is not None
            assert receipt["before_session_flags"] & 7 == 4
            assert receipt["after_session_flags"] & 7 == 4
            assert read_device_flags(receipt["library"], 0) & 7 == 4
        else:
            assert model._cuda_wait_policy is None

        weights = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        tensor = np.arange(4, dtype=np.float32).reshape(1, 4)
        saved = []
        for offset in (0, 7, -3):
            # Mutate the same host array: persistent graph addresses must not
            # cause replay to use the previous request's contents.
            tensor[:] = np.arange(4, dtype=np.float32) + offset
            outputs = model._model.run(None, {"input": tensor})
            expected = tensor @ weights
            np.testing.assert_allclose(outputs[0], expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(outputs[1], expected + 1, rtol=1e-5, atol=1e-5)
            saved.append((outputs, [value.copy() for value in outputs]))

            response = await model.predict(
                InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="input",
                            datatype="FP32",
                            shape=[1, 4],
                            data=tensor.ravel().tolist(),
                        )
                    ],
                    outputs=[
                        RequestOutput(name=name) for name in ("linear", "shifted")
                    ],
                )
            )
            assert [output.name for output in response.outputs] == ["linear", "shifted"]
            for output, value in zip(response.outputs, (expected, expected + 1)):
                np.testing.assert_allclose(
                    np.asarray(output.data.root).reshape(output.shape),
                    value,
                    rtol=1e-5,
                    atol=1e-5,
                )
        for outputs, copies in saved:
            for array, copy in zip(outputs, copies):
                np.testing.assert_array_equal(array, copy)
    finally:
        await model.unload()
