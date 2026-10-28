import argparse
import json
from pathlib import Path
import struct

import grpc
import pytest

from benchmarking import inference
from mlserver.grpc import dataplane_pb2 as pb


@pytest.mark.parametrize("fail", [False, True])
async def test_benchmark_sends_raw_grpc_and_checks_rpc_errors(
    tmp_path: Path, fail: bool
):
    received = []
    reply = pb.ModelInferResponse(model_name="sum-model").SerializeToString()

    async def infer(request: bytes, context: grpc.aio.ServicerContext) -> bytes:
        received.append(pb.ModelInferRequest.FromString(request))
        if fail:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid input")
        return reply

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        [
            grpc.method_handlers_generic_handler(
                "inference.GRPCInferenceService",
                {"ModelInfer": grpc.unary_unary_rpc_method_handler(infer)},
            )
        ]
    )
    port = server.add_insecure_port("127.0.0.1:0")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "name": "input",
                        "datatype": "FP32",
                        "shape": [1, 3],
                        "data": [1.0, 2.0, 3.0],
                    }
                ],
                "outputs": [{"name": "sum"}],
            }
        )
    )
    args = argparse.Namespace(
        protocol="grpc",
        request=request,
        model="sum-model",
        host="127.0.0.1",
        port=port,
        concurrency=2,
        requests=5,
        warmup=2,
        timeout=5,
    )
    await server.start()
    try:
        if fail:
            with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
                await inference.benchmark(args)
        else:
            result = await inference.benchmark(args)
            assert len(received) == 7
            assert result["requests"] == 5
            assert result["response_bytes"] == len(reply) * 5
    finally:
        await server.stop(grace=None)

    for message in received:
        assert message.model_name == "sum-model"
        assert message.inputs[0].shape == [1, 3]
        assert not message.inputs[0].HasField("contents")
        assert message.raw_input_contents == [struct.pack("<3f", 1.0, 2.0, 3.0)]
        assert [output.name for output in message.outputs] == ["sum"]
