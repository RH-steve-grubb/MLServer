import os
import asyncio
import socket
import pytest
import numpy as np
import onnx
import aiohttp
from onnx import helper, TensorProto
from grpc import aio
from prometheus_client.registry import REGISTRY
from starlette_exporter import PrometheusMiddleware
from aiohttp.client_exceptions import (
    ClientConnectorError,
    ClientOSError,
    ServerDisconnectedError,
)
from aiohttp_retry import RetryClient, ExponentialRetry

from mlserver.settings import ModelSettings, ModelParameters, Settings
from mlserver.types import InferenceRequest, RepositoryIndexResponse, InferenceResponse
from mlserver.grpc.dataplane_pb2_grpc import GRPCInferenceServiceStub
from mlserver.server import MLServer
from mlserver.utils import install_uvloop_event_loop

from mlserver_onnx import OnnxModel

TESTS_PATH = os.path.dirname(__file__)
TESTDATA_PATH = os.path.join(TESTS_PATH, "testdata")


def unregister_metrics(registry):
    """Unregister all collectors from a Prometheus registry."""
    collectors = list(registry._collector_to_names.keys())
    for collector in collectors:
        registry.unregister(collector)


def get_available_ports(n: int = 1):
    """Return a list of n available TCP ports."""
    ports: set[int] = set()

    while len(ports) < n:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        ports.add(port)

    return list(ports)


class RESTClient:
    """Minimal REST client for MLServer integration tests."""

    def __init__(self, http_server: str):
        self._session = aiohttp.ClientSession(raise_for_status=True)
        self._http_server = http_server

    async def close(self) -> None:
        """Close underlying HTTP session."""
        await self._session.close()

    async def _retry_get(self, endpoint: str) -> None:
        retry_options = ExponentialRetry(
            attempts=10,
            start_timeout=0.5,
            statuses={400},
            exceptions={
                ClientConnectorError,
                ClientOSError,
                ServerDisconnectedError,
                ConnectionRefusedError,
            },
        )
        retry_client = RetryClient(raise_for_status=True, retry_options=retry_options)

        async with retry_client:
            await retry_client.get(endpoint)

    async def wait_until_live(self) -> None:
        """Wait until the server is live."""
        endpoint = f"http://{self._http_server}/v2/health/live"
        await self._retry_get(endpoint)

    async def list_models(self) -> RepositoryIndexResponse:
        """List ready models from the repository."""
        endpoint = f"http://{self._http_server}/v2/repository/index"
        response = await self._session.post(endpoint, json={"ready": True})

        raw_payload = await response.text()
        return RepositoryIndexResponse.model_validate_json(raw_payload)

    async def infer(
        self, model_name: str, inference_request: InferenceRequest
    ) -> InferenceResponse:
        """Send an inference request and parse the response."""
        endpoint = f"http://{self._http_server}/v2/models/{model_name}/infer"
        response = await self._session.post(
            endpoint, json=inference_request.model_dump()
        )

        raw_payload = await response.text()
        return InferenceResponse.model_validate_json(raw_payload)


@pytest.fixture
def event_loop():
    """Provide an event loop with uvloop installed."""
    install_uvloop_event_loop()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def prometheus_registry():
    """Provide a clean Prometheus registry and cleanup."""
    yield REGISTRY
    unregister_metrics(REGISTRY)
    PrometheusMiddleware._metrics.clear()


def _create_simple_onnx_model():
    """Create a simple ONNX model that adds 1 to input."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 4]
    )

    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["constant"],
        value=helper.make_tensor(
            "const_tensor",
            TensorProto.FLOAT,
            [1, 4],
            np.ones(4, dtype=np.float32).tolist(),
        ),
    )

    add_node = helper.make_node(
        "Add",
        inputs=["input", "constant"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [constant_node, add_node],
        "simple_model",
        [input_tensor],
        [output_tensor],
    )

    # Use IR version 9 and opset 11 for maximum compatibility
    # ONNX Runtime 1.23.x supports IR versions up to v11
    # This works with ONNX packages from 1.16 to 1.19
    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=9,
    )

    return model


@pytest.fixture
def model_uri(tmp_path) -> str:
    """Persist a simple ONNX model and return its path."""
    model = _create_simple_onnx_model()
    model_uri = os.path.join(tmp_path, "onnx-model.onnx")
    onnx.save(model, model_uri)
    return model_uri


@pytest.fixture
def model_settings(model_uri: str) -> ModelSettings:
    """Model settings for the simple ONNX model."""
    return ModelSettings(
        name="onnx-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=model_uri, version="v1.0.0"),
    )


@pytest.fixture
async def model(model_settings: ModelSettings) -> OnnxModel:
    """Loaded simple ONNX model instance."""
    model = OnnxModel(model_settings)
    model.ready = await model.load()
    return model


@pytest.fixture
def inference_request() -> InferenceRequest:
    """Inference request loaded from testdata JSON."""
    payload_path = os.path.join(TESTDATA_PATH, "inference-request.json")
    with open(payload_path) as f:
        return InferenceRequest.model_validate_json(f.read())


@pytest.fixture
def multi_output_model_uri(tmp_path) -> str:
    """Create an ONNX model with multiple outputs."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output1_tensor = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [None, 4]
    )
    output2_tensor = helper.make_tensor_value_info(
        "output2", TensorProto.FLOAT, [None, 4]
    )

    constant1_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["constant1"],
        value=helper.make_tensor(
            "const_tensor1",
            TensorProto.FLOAT,
            [1, 4],
            np.ones(4, dtype=np.float32).tolist(),
        ),
    )

    constant2_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["constant2"],
        value=helper.make_tensor(
            "const_tensor2",
            TensorProto.FLOAT,
            [1, 4],
            (np.ones(4, dtype=np.float32) * 2).tolist(),
        ),
    )

    add1_node = helper.make_node(
        "Add",
        inputs=["input", "constant1"],
        outputs=["output1"],
    )

    add2_node = helper.make_node(
        "Add",
        inputs=["input", "constant2"],
        outputs=["output2"],
    )

    graph = helper.make_graph(
        [constant1_node, constant2_node, add1_node, add2_node],
        "multi_output_model",
        [input_tensor],
        [output1_tensor, output2_tensor],
    )

    # Use IR version 9 and opset 11 for maximum compatibility
    model = helper.make_model(
        graph,
        producer_name="mlserver-onnx-test",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=9,
    )

    model_uri = os.path.join(tmp_path, "multi-output-model.onnx")
    onnx.save(model, model_uri)
    return model_uri


@pytest.fixture
def multi_output_model_settings(multi_output_model_uri: str) -> ModelSettings:
    """Model settings for the multi-output ONNX model."""
    return ModelSettings(
        name="multi-output-model",
        implementation=OnnxModel,
        parameters=ModelParameters(uri=multi_output_model_uri, version="v1.0.0"),
    )


@pytest.fixture
async def multi_output_model(
    multi_output_model_settings: ModelSettings,
) -> OnnxModel:
    """Loaded multi-output ONNX model instance."""
    model = OnnxModel(multi_output_model_settings)
    model.ready = await model.load()
    return model


@pytest.fixture
def invalid_model_uri(tmp_path) -> str:
    """Create a path to an invalid model file."""
    invalid_path = os.path.join(tmp_path, "corrupted-model.onnx")
    with open(invalid_path, "wb") as f:
        f.write(b"not-a-valid-onnx-file")
    return invalid_path


@pytest.fixture
def server_settings(tmp_path) -> Settings:
    """Server settings with free ports for tests."""
    http_port, grpc_port, metrics_port = get_available_ports(3)
    settings = Settings(
        host="127.0.0.1",
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        metrics_endpoint=None,
        model_repository_root=str(tmp_path),
        load_models_at_startup=False,
    )
    return settings


@pytest.fixture
async def onnx_mlserver(
    server_settings: Settings, model_settings: ModelSettings, prometheus_registry
):
    """Start MLServer with a single ONNX model."""
    server = MLServer(server_settings)
    server_task = asyncio.create_task(server.start())

    await server._model_registry.load(model_settings)

    yield server

    await server.stop()
    await server_task


@pytest.fixture
async def multi_model_mlserver(
    server_settings: Settings,
    model_settings: ModelSettings,
    multi_output_model_settings: ModelSettings,
    prometheus_registry,
):
    """Start MLServer with multiple ONNX models."""
    server = MLServer(server_settings)
    server_task = asyncio.create_task(server.start())

    await server._model_registry.load(model_settings)
    await server._model_registry.load(multi_output_model_settings)

    yield server

    await server.stop()
    await server_task


@pytest.fixture
async def rest_client(server_settings: Settings, onnx_mlserver: MLServer):
    """REST client for the ONNX MLServer."""
    http_server = f"{server_settings.host}:{server_settings.http_port}"
    client = RESTClient(http_server)
    await client.wait_until_live()

    yield client

    await client.close()


@pytest.fixture
async def rest_client_multi_model(
    server_settings: Settings, multi_model_mlserver: MLServer
):
    """REST client for the multi-model MLServer."""
    http_server = f"{server_settings.host}:{server_settings.http_port}"
    client = RESTClient(http_server)
    await client.wait_until_live()

    yield client

    await client.close()


@pytest.fixture
async def grpc_stub(server_settings: Settings, onnx_mlserver: MLServer):
    """gRPC stub for the ONNX MLServer."""
    async with aio.insecure_channel(
        f"{server_settings.host}:{server_settings.grpc_port}"
    ) as channel:
        yield GRPCInferenceServiceStub(channel)
