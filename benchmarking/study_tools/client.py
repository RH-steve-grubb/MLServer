"""Timed, full-response REST/raw-gRPC load; use sweep.py for multiple clients."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import time
from typing import Any, AsyncIterator, Awaitable, Callable

import aiohttp
import grpc
import numpy as np

Sender = Callable[[int], Awaitable[tuple[str, int]]]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(spec: dict[str, Any]) -> None:
    if spec["protocol"] not in ("rest", "grpc"):
        raise ValueError("protocol must be rest or grpc")
    for name in ("concurrency", "warmup_requests"):
        if type(spec[name]) is not int or spec[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seconds", "warmup_seconds", "timeout_seconds"):
        value = spec[name]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not spec["payloads"] or not all(Path(p).is_file() for p in spec["payloads"]):
        raise ValueError("payloads must name existing prebuilt wire files")
    if spec["protocol"] == "rest":
        from urllib.parse import urlsplit

        url = urlsplit(spec["url"])
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("REST requires a full http(s) inference URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Use bearer_token_env, not credentials/query in the URL")
    elif not spec.get("target"):
        raise ValueError("gRPC requires a host:port target")


@asynccontextmanager
async def transport(spec: dict[str, Any], bodies: list[bytes]) -> AsyncIterator[Sender]:
    headers = dict(spec.get("headers", {}))
    if spec.get("bearer_token_env"):
        headers["Authorization"] = "Bearer " + os.environ[spec["bearer_token_env"]]
    if spec["protocol"] == "rest":
        headers.setdefault("Content-Type", "application/json")
        headers["Accept-Encoding"] = "identity"
        tls = ssl.create_default_context(cafile=spec.get("ca_file"))
        connector = aiohttp.TCPConnector(limit=spec["concurrency"], ssl=tls)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=spec["timeout_seconds"]),
            auto_decompress=False,
        ) as session:

            async def rest(index: int) -> tuple[str, int]:
                async with session.post(
                    spec["url"],
                    data=bodies[index],
                    headers=headers,
                    allow_redirects=False,
                ) as response:
                    body = await response.read()
                    return str(response.status), len(body)

            yield rest
    else:
        options = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        if spec.get("tls", False):
            ca = Path(spec["ca_file"]).read_bytes() if spec.get("ca_file") else None
            channel = grpc.aio.secure_channel(
                spec["target"], grpc.ssl_channel_credentials(ca), options=options
            )
        else:
            channel = grpc.aio.insecure_channel(spec["target"], options=options)
        async with channel:
            method = channel.unary_unary("/inference.GRPCInferenceService/ModelInfer")
            metadata = [(key.lower(), value) for key, value in headers.items()]

            async def raw_grpc(index: int) -> tuple[str, int]:
                body = await method(
                    bodies[index], timeout=spec["timeout_seconds"], metadata=metadata
                )
                return "200", len(body)

            yield raw_grpc


async def exercise(send: Sender, spec: dict[str, Any], warm: bool) -> dict[str, Any]:
    began = time.monotonic_ns()
    cpu = time.process_time_ns()
    seconds = spec["warmup_seconds"] if warm else spec["seconds"]
    deadline = began + int(seconds * 1e9)
    minimum = spec["warmup_requests"] if warm else 0
    samples: list[dict[str, Any]] = []
    admitted = 0
    failed = False

    async def worker() -> None:
        nonlocal admitted, failed
        while not failed and (time.monotonic_ns() < deadline or admitted < minimum):
            sequence = admitted
            admitted += 1
            payload = (sequence + spec.get("payload_offset", 0)) % len(spec["payloads"])
            start = time.monotonic_ns()
            status, size = "ERROR", 0
            try:
                status, size = await send(payload)
            except Exception as error:
                # Do not persist response bodies, URLs from exceptions, or tokens.
                status = type(error).__name__
            ok = status == "200" and size >= spec.get("min_response_bytes", 1)
            failed |= not ok
            samples.append(
                {
                    "sequence": sequence,
                    "payload": payload,
                    "start_ns": start,
                    "end_ns": time.monotonic_ns(),
                    "status": status,
                    "response_bytes": size,
                    "ok": ok,
                }
            )

    await asyncio.gather(*(worker() for _ in range(spec["concurrency"])))
    drained = time.monotonic_ns()
    return {
        "started_ns": began,
        "admission_deadline_ns": deadline,
        "drained_ns": drained,
        "cpu_seconds": (time.process_time_ns() - cpu) / 1e9,
        "attempted": admitted,
        "successful": sum(row["ok"] for row in samples),
        "valid": not failed and len(samples) == admitted,
        "samples": samples,
    }


async def run(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(spec)
    bodies = [Path(path).read_bytes() for path in spec["payloads"]]
    if any(not body for body in bodies):
        raise ValueError("Empty wire payload")
    async with transport(spec, bodies) as send:
        warmup = await exercise(send, spec, True)
        write_json(output / "warmup.json", warmup)
        if not warmup["valid"]:
            raise RuntimeError("Warmup failed; inspect warmup.json")
        write_json(output / "ready.json", {"pid": os.getpid()})
        gate = Path(spec["start_file"])
        deadline = time.monotonic() + spec.get("gate_timeout_seconds", 180)
        while not gate.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Coordinator did not publish the start gate")
            await asyncio.sleep(0.01)
        requested = json.loads(gate.read_text())["start_ns"]
        await asyncio.sleep(max(0, (requested - time.monotonic_ns()) / 1e9))
        measured = await exercise(send, spec, False)
    rows = measured.pop("samples")
    with (output / "samples.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    latencies = [(row["end_ns"] - row["start_ns"]) / 1e6 for row in rows if row["ok"]]
    measured.update(
        requested_start_ns=requested,
        start_lateness_ms=(measured["started_ns"] - requested) / 1e6,
        latency_ms=(
            dict(
                zip(
                    ("p50", "p95", "p99"),
                    map(float, np.percentile(latencies, [50, 95, 99])),
                )
            )
            if latencies
            else {}
        ),
        payloads=[
            {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for body in bodies
        ],
        pid=os.getpid(),
        affinity=sorted(os.sched_getaffinity(0)),
    )
    write_json(output / "result.json", measured)
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    try:
        result = asyncio.run(run(json.loads(args.spec.read_text()), args.output))
    except Exception as error:
        write_json(args.output / "failure.json", {"error_type": type(error).__name__})
        raise
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
