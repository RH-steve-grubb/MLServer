import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from benchmarking import inference


@pytest.fixture
def benchmark_args(tmp_path: Path) -> argparse.Namespace:
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
        + "\n"
    )
    return argparse.Namespace(
        request=request,
        protocol="rest",
        model="sum-model",
        host="127.0.0.1",
        port=0,
        requests=7,
        concurrency=3,
        warmup=2,
        timeout=5,
    )


async def test_rest_benchmark_excludes_warmup_and_reads_complete_responses(
    benchmark_args: argparse.Namespace,
):
    received = []
    body = b"already encoded response"

    async def infer(request: web.Request) -> web.Response:
        received.append(await request.read())
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.content_type == "application/json"
        return web.Response(body=body)

    app = web.Application()
    app.router.add_post("/v2/models/sum-model/infer", infer)
    async with TestServer(app) as server:
        benchmark_args.port = server.port
        result = await inference.benchmark(benchmark_args)

    payload = benchmark_args.request.read_bytes()
    assert received == [payload] * 9
    assert result["requests"] == 7
    assert result["response_bytes"] == 7 * len(body)
    assert result["request_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["rps"] == 7 / result["duration_seconds"]
    assert 0 <= result["latency_ms"]["p50"] <= result["latency_ms"]["p99"]


async def test_benchmark_rejects_http_error(benchmark_args: argparse.Namespace):
    async def infer(request: web.Request) -> web.Response:
        return web.Response(status=422, text="invalid input")

    app = web.Application()
    app.router.add_post("/v2/models/sum-model/infer", infer)
    async with TestServer(app) as server:
        benchmark_args.port = server.port
        with pytest.raises(RuntimeError, match="HTTP 422"):
            await inference.benchmark(benchmark_args)


async def test_request_error_drains_inflight_without_admitting_more():
    started = asyncio.Event()
    admitted = []
    completed = []

    async def send() -> int:
        index = len(admitted)
        admitted.append(index)
        if index == 0:
            await started.wait()
            raise RuntimeError("inference failed")
        started.set()
        await asyncio.sleep(0)
        completed.append(index)
        return 10

    with pytest.raises(RuntimeError, match="inference failed"):
        await inference.run_requests(send, requests=20, concurrency=2)
    assert admitted == [0, 1]
    assert completed == [1]


async def test_run_honors_concurrency_and_completes_uneven_request_count():
    active = 0
    peak = 0

    async def send() -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return 10

    result = await inference.run_requests(send, requests=7, concurrency=3)
    assert peak == 3
    assert active == 0
    assert result["requests"] == 7
    assert result["response_bytes"] == 70


async def test_empty_response_does_not_produce_successful_result():
    async def send() -> int:
        return 0

    with pytest.raises(RuntimeError, match="Empty response"):
        await inference.run_requests(send, requests=2, concurrency=1)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--requests", "0"],
        ["--concurrency", "0"],
        ["--warmup", "-1"],
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--timeout", "0"],
        ["--port", "0"],
        ["--port", "65536"],
    ],
)
def test_invalid_cli_arguments_are_rejected_before_connecting(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "sys.argv",
        ["inference.py", "--model", "model", "--request", "unused.json", *arguments],
    )
    with pytest.raises(SystemExit) as error:
        inference.main()
    assert error.value.code == 2
