"""Benchmark a running model using a prepared V2 JSON request, over REST or gRPC."""

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

import aiohttp
import grpc
import numpy as np

from mlserver.grpc.converters import ModelInferRequestConverter
from mlserver.types import InferenceRequest

Sender = Callable[[], Awaitable[int]]
RPC_METHOD = "/inference.GRPCInferenceService/ModelInfer"


def request_bytes(args: argparse.Namespace) -> bytes:
    payload = args.request.read_bytes()
    request = InferenceRequest.model_validate_json(payload)
    if args.protocol == "rest":
        return payload
    # Convert once, outside timing, and preserve the resulting binary payload
    # across requests. Client-side JSON/float conversion must not limit gRPC.
    return ModelInferRequestConverter.from_types(
        request, model_name=args.model, use_raw=True
    ).SerializeToString()


@asynccontextmanager
async def sender(args: argparse.Namespace, payload: bytes) -> AsyncIterator[Sender]:
    if args.protocol == "rest":
        connector = aiohttp.TCPConnector(limit=args.concurrency)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=args.timeout),
            auto_decompress=False,
        ) as session:
            url = (
                f"http://{args.host}:{args.port}/v2/models/"
                f"{quote(args.model, safe='')}/infer"
            )

            async def send_rest() -> int:
                async with session.post(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept-Encoding": "identity",
                    },
                    allow_redirects=False,
                ) as response:
                    body = await response.read()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:200]!r}")
                    return len(body)

            yield send_rest
        return
    message_limit = max(32 * 1024 * 1024, len(payload) + 1024)
    options = [
        ("grpc.max_send_message_length", message_limit),
        ("grpc.max_receive_message_length", message_limit),
    ]
    async with grpc.aio.insecure_channel(
        f"{args.host}:{args.port}", options=options
    ) as channel:
        # Receive serialized bytes, like REST. Response parsing is outside this
        # benchmark's scope; both protocols must read the complete reply.
        method = channel.unary_unary(RPC_METHOD)

        async def send_grpc() -> int:
            body = await method(payload, timeout=args.timeout)
            return len(body)

        yield send_grpc


async def run_requests(send: Sender, requests: int, concurrency: int) -> dict[str, Any]:
    pending = iter(range(requests))
    latencies: list[float] = []
    sizes: list[int] = []
    errors: list[Exception] = []

    async def worker() -> None:
        for _ in pending:
            if errors:
                return
            began = time.perf_counter()
            try:
                size = await send()
                if size <= 0:
                    raise RuntimeError("Empty response")
            except Exception as error:
                errors.append(error)
                return
            latencies.append((time.perf_counter() - began) * 1000)
            sizes.append(size)

    began = time.perf_counter()
    # On failure, stop new admissions but let every in-flight request drain.
    # Never report an incomplete or cancelled run as successful throughput.
    await asyncio.gather(*(worker() for _ in range(min(requests, concurrency))))
    duration = time.perf_counter() - began
    if errors:
        raise RuntimeError(f"Benchmark failed: {errors[0]}") from errors[0]
    if len(latencies) != requests:
        raise RuntimeError("Not all requests completed")
    return {
        "requests": requests,
        "duration_seconds": duration,
        "rps": requests / duration,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "response_bytes": sum(sizes),
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    payload = request_bytes(args)
    async with sender(args, payload) as send:
        if args.warmup:
            await run_requests(send, args.warmup, args.concurrency)
        result = await run_requests(send, args.requests, args.concurrency)
    return {
        "protocol": args.protocol,
        "model": args.model,
        "host": args.host,
        "port": args.port,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup,
        "request_bytes": len(payload),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--protocol", choices=("rest", "grpc"), default="rest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive; warmup must be >= 0")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.port is None:
        args.port = 8080 if args.protocol == "rest" else 8081
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new result filename")
    try:
        result = asyncio.run(benchmark(args))
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as output:
                output.write(text)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(text, end="")


if __name__ == "__main__":
    main()
