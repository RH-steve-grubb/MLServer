"""Check direct ONNX and real MLServer worker/REST/gRPC paths, not throughput."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import grpc
import numpy as np
import onnxruntime as ort

from mlserver.grpc import dataplane_pb2 as pb
from mlserver.grpc import dataplane_pb2_grpc as rpc

from prepare import checksum, compare_detections, write_json
from serve import serving_environment


def decoded(outputs: dict[str, Any], image_size: list[int]) -> dict[str, Any]:
    boxes, logits = outputs["dets"][0], outputs["labels"][0]
    scores = 1 / (1 + np.exp(-np.clip(logits, -88, 88)))
    # Match upstream's global top-300 query/class selection. Official COCO
    # weights use sparse IDs: the final slot is class 90, not background.
    indices = np.argsort(scores.reshape(-1))[-300:][::-1]
    selected = scores.reshape(-1)[indices]
    indices, selected = indices[selected > 0.5], selected[selected > 0.5]
    queries, labels = np.divmod(indices, logits.shape[1])
    cx, cy, width, height = boxes[queries].T
    xyxy = np.stack(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=1
    )
    scale = np.array(image_size * 2, dtype=np.float32)
    return {
        "xyxy": np.clip(xyxy * scale, 0, scale).tolist(),
        "confidence": selected.tolist(),
        "class_id": labels.tolist(),
    }


def direct_inference(bundle: Path, device: str, output: Path) -> dict[str, Any]:
    if device == "cuda":
        ort.preload_dlls(directory="")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.enable_profiling = True
    options.profile_file_prefix = str(output / "onnx-profile")
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(
        str(bundle / "model" / "model.onnx"),
        sess_options=options,
        providers=providers,
    )
    if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA initialization failed; refusing silent CPU fallback")
    names = [value.name for value in session.get_outputs()]
    tensor = np.load(bundle / "sample" / "input.npy", allow_pickle=False)
    outputs = dict(zip(names, session.run(names, {"input": tensor})))
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text())
    assignments = Counter(
        event["args"]["provider"]
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    if device == "cuda" and not assignments["CUDAExecutionProvider"]:
        raise RuntimeError("Profiling did not record any CUDA node execution")
    np.savez(output / "direct.npz", **outputs)
    write_json(
        output / "direct.json",
        {
            "providers": session.get_providers(),
            "provider_options": session.get_provider_options(),
            "profile_node_execution_counts": dict(assignments),
            "profile": profile.name,
            "output_shapes": {
                name: list(value.shape) for name, value in outputs.items()
            },
        },
    )
    return outputs


def grpc_request(bundle: Path) -> Any:
    request = pb.ModelInferRequest(
        model_name="rfdetr-small",
        id="rfdetr-small-smoke",
        inputs=[
            pb.ModelInferRequest.InferInputTensor(
                name="input", shape=[1, 3, 512, 512], datatype="FP32"
            )
        ],
        outputs=[
            pb.ModelInferRequest.InferRequestedOutputTensor(name=name)
            for name in ("dets", "labels")
        ],
        raw_input_contents=[(bundle / "sample" / "input.fp32").read_bytes()],
    )
    (bundle / "sample" / "request.grpc.pb").write_bytes(request.SerializeToString())
    return request


def check_ports(ports: tuple[int, ...] = (18080, 18081, 18082)) -> None:
    for port in ports:
        with socket.socket() as probe:
            # Match the server's address reuse: TIME_WAIT after our previous
            # run is harmless, while an existing listener must still fail.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))


def wait_ready(process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MLServer exited during startup; inspect {log}")
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:18080/v2/models/rfdetr-small/ready", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    raise RuntimeError(f"Model did not become ready; inspect {log}")


def serving_inference(bundle: Path, device: str, output: Path) -> dict[str, Any]:
    check_ports()
    log = output / "server.log"
    command = [
        sys.executable,
        str(Path(__file__).with_name("serve.py")),
        "--bundle",
        str(bundle),
        "--device",
        device,
    ]
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=serving_environment(),
            start_new_session=True,
        )
        try:
            wait_ready(process, log)
            request = urllib.request.Request(
                "http://127.0.0.1:18080/v2/models/rfdetr-small/infer",
                data=(bundle / "sample" / "request.json").read_bytes(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                rest = json.load(response)
            rest_outputs = {
                value["name"]: np.asarray(value["data"], dtype=np.float32).reshape(
                    value["shape"]
                )
                for value in rest["outputs"]
            }
            options = [("grpc.max_receive_message_length", 32 * 1024 * 1024)]
            with grpc.insecure_channel("127.0.0.1:18081", options=options) as channel:
                response = rpc.GRPCInferenceServiceStub(channel).ModelInfer(
                    grpc_request(bundle), timeout=120
                )
            grpc_outputs = {
                value.name: np.frombuffer(raw, dtype="<f4").reshape(value.shape)
                for value, raw in zip(response.outputs, response.raw_output_contents)
            }
            if rest["id"] != "rfdetr-small-smoke" or response.id != rest["id"]:
                raise ValueError("Serving response did not preserve the request ID")
            return {"rest": rest_outputs, "grpc": grpc_outputs}
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Only terminate the process group created by this smoke test.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = bundle / "validation" / f"{args.device}-{stamp}"
    output.mkdir(parents=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if checksum(bundle / "model" / "model.onnx") != manifest["onnx"]["sha256"]:
        raise ValueError("Model does not match the prepared artifact's SHA-256")
    expected = json.loads(
        (bundle / "reference" / "pytorch-detections.json").read_text()
    )
    size = manifest["sample"]["image_size_wh"]
    direct = direct_inference(bundle, args.device, output)
    compare_detections(expected, decoded(direct, size))
    routed = serving_inference(bundle, args.device, output)
    for protocol, values in routed.items():
        if set(values) != {"dets", "labels"}:
            raise ValueError(f"{protocol} did not return both requested outputs")
        for name in direct:
            np.testing.assert_allclose(values[name], direct[name], rtol=1e-3, atol=1e-3)
        compare_detections(expected, decoded(values, size))
        np.savez(output / f"{protocol}.npz", **values)
    write_json(
        output / "SUMMARY.json",
        {
            "status": "passed",
            "device": args.device,
            "model_sha256": manifest["onnx"]["sha256"],
            "python": sys.version,
            "checks": [
                "direct ONNX",
                "MLServer REST + worker",
                "MLServer raw gRPC + worker",
            ],
            "detections_at_0_5": len(expected["class_id"]),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "mlserver",
                    "mlserver-onnx",
                    "onnxruntime-gpu",
                    "numpy",
                    "grpcio",
                )
            },
            "note": "Correctness/serving smoke test, not a throughput benchmark",
        },
    )
    print(f"Passed: {output / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
