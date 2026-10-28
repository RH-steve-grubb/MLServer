"""Accounting and transport checks for the portable study utilities."""

import asyncio
import json
from pathlib import Path
import sys
import time

from aiohttp import web
import grpc
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "study_tools"
sys.path.insert(0, str(TOOLS))
import analyze  # noqa: E402
import client  # noqa: E402
import payloads  # noqa: E402
import sweep  # noqa: E402


def specification() -> dict:
    return {
        "protocol": "rest",
        "url": "http://127.0.0.1:1/infer",
        "concurrency": 2,
        "seconds": 0.02,
        "warmup_seconds": 0.02,
        "warmup_requests": 3,
        "timeout_seconds": 2,
        "payloads": ["unused"],
    }


async def test_failure_stops_admission_and_drains_inflight() -> None:
    calls = 0

    async def send(index: int) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        current = calls
        await asyncio.sleep(0.01 if current == 1 else 0.04)
        return ("500" if current == 1 else "200"), 12

    result = await client.exercise(send, specification(), False)
    assert not result["valid"]
    assert result["attempted"] == 2
    assert len(result["samples"]) == 2
    assert result["successful"] == 1
    assert (result["drained_ns"] - result["started_ns"]) / 1e9 >= 0.04


async def test_warmup_requires_count_and_time() -> None:
    async def send(index: int) -> tuple[str, int]:
        await asyncio.sleep(0.005)
        return "200", 12

    result = await client.exercise(send, specification(), True)
    assert result["valid"]
    assert result["attempted"] >= 3
    assert result["drained_ns"] >= result["admission_deadline_ns"]


async def test_rest_consumes_complete_body_and_preserves_headers() -> None:
    received = []

    async def handler(request: web.Request) -> web.StreamResponse:
        received.append(
            (
                await request.read(),
                request.headers.get("Inference-Header-Content-Length"),
            )
        )
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await asyncio.sleep(0.03)
        await response.write(b"second")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/infer", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = runner.addresses[0][1]
        spec = {
            **specification(),
            "url": f"http://127.0.0.1:{port}/infer",
            "headers": {"Inference-Header-Content-Length": "3"},
        }
        async with client.transport(spec, [b"{} binary"]) as send:
            start = time.monotonic()
            assert await send(0) == ("200", 11)
            assert time.monotonic() - start >= 0.03
        assert received == [(b"{} binary", "3")]
    finally:
        await runner.cleanup()


async def test_grpc_sends_and_receives_serialized_bytes() -> None:
    received = []

    async def infer(request, context):
        received.append(request)
        return b"raw-response"

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "inference.GRPCInferenceService",
                {"ModelInfer": grpc.unary_unary_rpc_method_handler(infer)},
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        spec = {**specification(), "protocol": "grpc", "target": f"127.0.0.1:{port}"}
        async with client.transport(spec, [b"raw-request"]) as send:
            assert await send(0) == ("200", 12)
        assert received == [b"raw-request"]
    finally:
        await server.stop(None)


async def test_sweep_subprocess_barrier_and_raw_samples(tmp_path: Path) -> None:
    async def handler(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(body=b"output")

    app = web.Application()
    app.router.add_post("/infer", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    payload = tmp_path / "body"
    payload.write_bytes(b"input")
    plan = {
        "session": "fixture",
        "round_offset": 0,
        "rounds": 1,
        "concurrency": [2],
        "clients": 2,
        "seconds": 0.1,
        "warmup_seconds": 0.05,
        "warmup_requests": 4,
        "timeout_seconds": 2,
        "ready_timeout_seconds": 30,
        "prepare_timeout_seconds": 30,
        "max_start_lateness_ms": 100,
        "variants": [
            {
                "name": "fixture-only",
                "prepare_argv": [sys.executable, "-c", "pass"],
                "spec": {
                    "protocol": "rest",
                    "url": f"http://127.0.0.1:{runner.addresses[0][1]}/infer",
                    "payloads": [str(payload)],
                },
            }
        ],
    }
    try:
        result = await asyncio.to_thread(sweep.execute, plan, tmp_path / "run")
        row = result["results"][0]
        assert row["valid"]
        assert row["successful"] == row["attempted"] > 0
        assert row["duration_seconds"] >= 0.1
        assert len(row["clients"]) == 2
        samples = [
            json.loads(line)
            for path in (tmp_path / "run").rglob("samples.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(samples) == row["successful"]
        assert all(sample["response_bytes"] == 6 for sample in samples)
    finally:
        await runner.cleanup()


def test_payloads_preserve_float32_bits() -> None:
    tensor = np.arange(786432, dtype=np.float32).reshape(1, 3, 512, 512) / np.float32(
        100000
    )
    request = {
        "id": "fixture",
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": list(tensor.shape),
                "data": [],
            }
        ],
        "outputs": [{"name": "dets"}, {"name": "labels"}],
    }
    j0, j1, raw = payloads.encode(request, tensor)
    assert len(j1) < len(j0)
    parsed = payloads.pb.ModelInferRequest.FromString(raw)
    assert parsed.raw_input_contents[0] == tensor.tobytes()
    assert {out.name for out in parsed.outputs} == {"dets", "labels"}
    assert len({array.tobytes() for array in payloads.variants(tensor)}) == 8


def test_analysis_flags_invalid_pairs_and_duplicates() -> None:
    rows = []
    for round_number in range(7):
        for name, rps in [("before", 10), ("after", 12)]:
            rows.append(
                {
                    "session": "fixture",
                    "round": round_number,
                    "concurrency": 1,
                    "variant": name,
                    "valid": True,
                    "case": str(round_number),
                    "rps": rps,
                    "latency_ms": {"p95": 2, "p99": 3},
                }
            )
    result = analyze.analyze([{"results": rows}], ["before", "after"])
    assert result["metrics"]["1"]["rps"]["median"] == pytest.approx(1.2)
    rows[-1]["valid"] = False
    result = analyze.analyze([{"results": rows}], ["before", "after"])
    assert len(result["missing_or_invalid_pairs"]) == 1
    assert result["metrics"]["1"]["rps"]["pairs"] == 6
    with pytest.raises(ValueError, match="Duplicate"):
        analyze.analyze([{"results": rows}, {"results": rows}], ["before", "after"])


def test_validation_rejects_credentials_in_url(tmp_path: Path) -> None:
    path = tmp_path / "body"
    path.write_bytes(b"body")
    with pytest.raises(ValueError, match="credentials"):
        client.validate(
            {
                **specification(),
                "payloads": [str(path)],
                "url": "https://name:secret@example.com/infer",
            }
        )
