"""Run a frozen JSON sweep plan; prepare_argv must restart and qualify each server."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from client import write_json


def snapshot(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    root = Path(path)
    values: dict[str, Any] = {"monotonic_ns": time.monotonic_ns(), "path": str(root)}
    for name in (
        "cpu.stat",
        "cpu.max",
        "cpuset.cpus.effective",
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.swap.max",
        "memory.events",
    ):
        values[name] = (root / name).read_text() if (root / name).exists() else None
    if values["cpu.stat"] is None:
        raise ValueError(f"Missing cgroup v2 CPU accounting in {root}")
    return values


def collect(processes: list[subprocess.Popen], limit: float) -> None:
    deadline = time.monotonic() + limit
    while any(process.poll() is None for process in processes):
        if time.monotonic() > deadline:
            raise TimeoutError("Load clients exceeded the bounded drain timeout")
        time.sleep(0.05)


def trial(config: dict[str, Any], location: Path) -> dict[str, Any]:
    total = config["concurrency"]
    clients = min(config.get("clients", 2), total)
    if total % clients:
        raise ValueError("Total concurrency must divide evenly across active clients")
    spec = config["spec"]
    processes: list[subprocess.Popen] = []
    logs = []
    result: dict[str, Any] = {"valid": False}
    try:
        for index in range(clients):
            child = {
                **spec,
                "concurrency": total // clients,
                "seconds": config["seconds"],
                "warmup_seconds": config["warmup_seconds"],
                "warmup_requests": max(
                    1, math_ceil(config["warmup_requests"], clients)
                ),
                "payload_offset": index,
                "timeout_seconds": config["timeout_seconds"],
                "gate_timeout_seconds": config["ready_timeout_seconds"],
                "start_file": str(location / "start.json"),
            }
            child_spec = location / f"client-{index}-spec.json"
            write_json(child_spec, child)
            log = (location / f"client-{index}.log").open("w")
            logs.append(log)
            command = [
                sys.executable,
                str(Path(__file__).with_name("client.py")),
                "--spec",
                str(child_spec),
                "--output",
                str(location / f"client-{index}"),
            ]
            processes.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + config["ready_timeout_seconds"]
        while not all(
            (location / f"client-{i}/ready.json").exists() for i in range(clients)
        ):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A load client failed during preparation/warmup")
            if time.monotonic() > deadline:
                raise TimeoutError("Clients did not finish warmup")
            time.sleep(0.05)
        # These optional cgroup counters enclose admission and the full drain.
        # Their envelope includes the one-second start gate and result collection.
        before = snapshot(config.get("cgroup"))
        requested = time.monotonic_ns() + 1_000_000_000
        write_json(
            location / "start.json",
            {
                "start_ns": requested,
                "utc_recorded": datetime.now(timezone.utc).isoformat(),
            },
        )
        collect(processes, config["seconds"] + config["timeout_seconds"] + 30)
        after = snapshot(config.get("cgroup"))
        rows = [
            json.loads((location / f"client-{i}/result.json").read_text())
            for i in range(clients)
        ]
        end = max(row["drained_ns"] for row in rows)
        latencies = []
        for index in range(clients):
            for line in (
                (location / f"client-{index}/samples.jsonl").read_text().splitlines()
            ):
                sample = json.loads(line)
                if sample["ok"]:
                    latencies.append((sample["end_ns"] - sample["start_ns"]) / 1e6)
        count = sum(row["successful"] for row in rows)
        result = {
            "valid": all(row["valid"] for row in rows)
            and all(process.returncode == 0 for process in processes)
            and max(row["start_lateness_ms"] for row in rows)
            <= config["max_start_lateness_ms"],
            "requested_start_ns": requested,
            "drained_ns": end,
            "duration_seconds": (end - requested) / 1e9,
            "successful": count,
            "attempted": sum(row["attempted"] for row in rows),
            "rps": count / ((end - requested) / 1e9),
            "latency_ms": (
                dict(
                    zip(
                        ("p50", "p95", "p99"),
                        map(float, np.percentile(latencies, [50, 95, 99])),
                    )
                )
                if latencies
                else {}
            ),
            "clients": rows,
            "cgroup_before": before,
            "cgroup_after": after,
            "cpu_ms_per_request_envelope": None,
        }
        if before and after and count:
            first = dict(line.split() for line in before["cpu.stat"].splitlines())
            last = dict(line.split() for line in after["cpu.stat"].splitlines())
            result["cpu_ms_per_request_envelope"] = (
                (int(last["usage_usec"]) - int(first["usage_usec"])) / 1000 / count
            )
    except Exception as error:
        result["error"] = str(error)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()
    write_json(location / "result.json", result)
    return result


def math_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def execute(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    if not plan["variants"] or len({v["name"] for v in plan["variants"]}) != len(
        plan["variants"]
    ):
        raise ValueError("Variant names must be unique and nonempty")
    results: list[dict[str, Any]] = []
    for round_number in range(
        plan["round_offset"], plan["round_offset"] + plan["rounds"]
    ):
        variants = (
            plan["variants"]
            if round_number % 2 == 0
            else list(reversed(plan["variants"]))
        )
        for concurrency in plan["concurrency"]:
            for variant in variants:
                case = output / f"case-{len(results):04d}"
                case.mkdir()
                environment = os.environ.copy()
                environment.update(
                    BENCHMARK_TRIAL_DIR=str(case), BENCHMARK_VARIANT=variant["name"]
                )
                row = {
                    "case": case.name,
                    "session": plan["session"],
                    "round": round_number,
                    "concurrency": concurrency,
                    "variant": variant["name"],
                }
                try:
                    # The hook is explicit, user-owned code. It must save readiness,
                    # source/config/resource receipts and restart the serving instance.
                    command = variant["prepare_argv"]
                    if not command or not all(isinstance(arg, str) for arg in command):
                        raise ValueError(
                            "Each variant requires an explicit prepare_argv"
                        )
                    with (case / "prepare.log").open("w") as log:
                        subprocess.run(
                            command,
                            check=True,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=plan["prepare_timeout_seconds"],
                        )
                    config = {**plan, **variant, "concurrency": concurrency}
                    if variant.get("cgroup_file"):
                        config["cgroup"] = (
                            Path(variant["cgroup_file"]).read_text().strip()
                        )
                    row.update(trial(config, case))
                except Exception as error:
                    row.update(valid=False, error=str(error))
                results.append(row)
                write_json(output / "SUMMARY.json", {"plan": plan, "results": results})
    return {"plan": plan, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(json.loads(args.plan.read_text()), args.output.resolve())
    if any(not row["valid"] for row in result["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
