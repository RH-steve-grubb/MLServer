"""Read-only client/server-host inventory; CUDA availability is not execution proof."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


def probe(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-2000:],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def inventory(bundle: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": {},
        "imports": {},
        "bundle": str(bundle),
    }
    for name in (
        "numpy",
        "orjson",
        "aiohttp",
        "grpcio",
        "onnxruntime-gpu",
        "onnx",
        "mlserver",
        "mlserver-onnx",
        "pydantic",
        "uvloop",
        "httptools",
        "pyzmq",
    ):
        try:
            result["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][name] = None
    for name in ("mlserver", "mlserver_onnx", "onnxruntime"):
        try:
            module = importlib.import_module(name)
            filename = module.__file__
            if filename is None:
                raise ValueError(f"No source filename for {name}")
            result["imports"][name] = str(Path(filename).resolve())
            if name == "onnxruntime":
                result["available_providers"] = module.get_available_providers()
        except Exception as error:
            result["imports"][name] = {"error": str(error)}
    for tool in (
        "git",
        "poetry",
        "perf",
        "nsys",
        "cc",
        "nm",
        "objdump",
        "taskset",
        "oc",
    ):
        result[tool] = (
            probe([tool, "--version"]) if shutil.which(tool) else {"missing": True}
        )
    result["nvidia_smi"] = probe(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv",
        ]
    )
    result["cpu_topology"] = probe(["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE"])
    result["cgroup"] = Path("/proc/self/cgroup").read_text()
    manifest = json.loads((bundle / "manifest.json").read_text())
    model = bundle / "model/model.onnx"
    with model.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result["model_sha256"] = digest
    result["model_matches_bundle"] = digest == manifest["onnx"]["sha256"]
    for relative in ("sample/request.json", "sample/input.npy", "sample/input.fp32"):
        path = bundle / relative
        result[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    result["scope"] = (
        "Inventory only; run model correctness and "
        "worker/resource qualification separately."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.bundle.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not result["model_matches_bundle"]:
        raise SystemExit("Model hash mismatch")


if __name__ == "__main__":
    main()
