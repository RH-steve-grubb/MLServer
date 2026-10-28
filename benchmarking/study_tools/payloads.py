"""Build J0/J1/raw-gRPC wire files and a finite, deterministic fresh-input corpus."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from mlserver.grpc import dataplane_pb2 as pb


def variants(tensor: np.ndarray) -> list[np.ndarray]:
    if tensor.shape != (1, 3, 512, 512) or tensor.dtype != np.float32:
        raise ValueError("Expected the prepared RF-DETR FP32 NCHW fixture")
    return [tensor.copy(), np.ascontiguousarray(tensor[:, :, :, ::-1])] + [
        np.ascontiguousarray(np.roll(tensor, shift, axis=3)) for shift in range(1, 7)
    ]


def encode(request: dict[str, Any], array: np.ndarray) -> tuple[bytes, bytes, bytes]:
    flat = array.reshape(-1)
    legacy = copy.deepcopy(request)
    legacy["inputs"][0]["data"] = flat.tolist()
    short = copy.deepcopy(request)
    short["inputs"][0]["data"] = flat
    j0 = orjson.dumps(legacy)
    j1 = orjson.dumps(short, option=orjson.OPT_SERIALIZE_NUMPY)
    raw = pb.ModelInferRequest(
        id=request.get("id", ""),
        model_name="rfdetr-small",
        inputs=[
            pb.ModelInferRequest.InferInputTensor(
                name="input", datatype="FP32", shape=list(array.shape)
            )
        ],
        outputs=[
            pb.ModelInferRequest.InferRequestedOutputTensor(name=name)
            for name in ("dets", "labels")
        ],
        raw_input_contents=[array.astype("<f4", copy=False).tobytes()],
    ).SerializeToString()
    for body in (j0, j1):
        restored = np.asarray(orjson.loads(body)["inputs"][0]["data"], dtype=np.float32)
        if restored.tobytes() != flat.tobytes():
            raise ValueError("Serialization changed FP32 input bits")
    return j0, j1, raw


def build(bundle: Path, output: Path) -> None:
    original = (bundle / "sample/request.json").read_bytes()
    request = orjson.loads(original)
    tensor = np.load(bundle / "sample/input.npy", allow_pickle=False)
    if len(request["inputs"]) != 1 or request["inputs"][0]["name"] != "input":
        raise ValueError("Expected one RF-DETR input")
    if {item["name"] for item in request["outputs"]} != {"dets", "labels"}:
        raise ValueError("Both raw model outputs must be requested")
    decoded = np.asarray(request["inputs"][0]["data"], dtype=np.float32).reshape(
        tensor.shape
    )
    if decoded.tobytes() != tensor.tobytes() or not np.isfinite(tensor).all():
        raise ValueError("Prepared JSON and array differ or contain nonfinite input")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, array in enumerate(variants(tensor)):
        item = copy.deepcopy(request)
        if index:
            item["id"] = f"corpus-{index:08d}"
        bodies = encode(item, array)
        np.save(output / f"input-{index}.npy", array, allow_pickle=False)
        for track, body in zip(("j0", "j1", "g"), bodies):
            if index == 0 and track == "j0":
                body = original  # Preserve the exact unchanged-client baseline bytes.
            name = f"{track}-{index}.bin"
            (output / name).write_bytes(body)
            records.append(
                {
                    "path": name,
                    "track": track,
                    "corpus_index": index,
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            )
    (output / "MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.bundle, args.output)


if __name__ == "__main__":
    main()
