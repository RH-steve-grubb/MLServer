"""Export the official RF-DETR Small checkpoint and build a serving fixture.

Run with the separate export-env Poetry project, not MLServer's serving Python.
All large/generated artifacts go into the explicitly selected durable directory.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import urllib.request

VERSION = "1.11.2"
WEIGHTS_URL = "https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth"
WEIGHTS_MD5 = "fb37061c1af7bace359c91b723a8d5c1"
# The original image hostname does not present a matching TLS certificate.
# Address the same public COCO bucket through S3's certificate-valid endpoint.
IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000039769.jpg"
LICENSE_URL = "https://raw.githubusercontent.com/roboflow/rf-detr/1.11.2/LICENSE"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def write_json(path: Path, value: Any, compact: bool = False) -> None:
    options: dict[str, Any] = {"allow_nan": False}
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    path.write_text(json.dumps(value, **options) + "\n")


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    print(f"Downloading {url}", flush=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as source:
        with partial.open("wb") as target:
            shutil.copyfileobj(source, target)
    partial.replace(destination)


def normalize(path: Path) -> Any:
    import numpy as np
    from PIL import Image
    from torchvision.transforms import functional as transforms

    with Image.open(path) as image:
        tensor = transforms.to_tensor(image.convert("RGB"))
    tensor = transforms.resize(tensor, [512, 512], antialias=False)
    tensor = transforms.normalize(tensor, MEAN, STD).unsqueeze(0)
    return np.ascontiguousarray(tensor.numpy(), dtype="<f4")


def decode(outputs: dict[str, Any], image_path: Path) -> dict[str, Any]:
    from PIL import Image
    from rfdetr.export._runtime.decode import decode_detections

    with Image.open(image_path) as image:
        size = image.size
    # Official COCO weights use sparse category IDs, including real class 90.
    # Do not drop the last logit as though it were a custom-model background slot.
    result = decode_detections(
        outputs["dets"][0],
        outputs["labels"][0],
        size,
        threshold=0.5,
        background_class_id=None,
    )
    return {
        "xyxy": result.xyxy.tolist(),
        "confidence": result.confidence.tolist(),
        "class_id": result.class_id.tolist(),
    }


def compare_detections(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    import numpy as np

    if not expected["class_id"] or len(expected["class_id"]) != len(actual["class_id"]):
        raise ValueError("Expected nonempty, equally sized detection sets")
    unused = set(range(len(actual["class_id"])))
    for box, score, label in zip(
        expected["xyxy"], expected["confidence"], expected["class_id"]
    ):
        candidates = [index for index in unused if actual["class_id"][index] == label]
        for index in candidates:
            other = actual["xyxy"][index]
            overlap = np.maximum(
                0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])
            )
            intersection = float(np.prod(overlap))
            area = float(np.prod(np.subtract(box[2:], box[:2])))
            other_area = float(np.prod(np.subtract(other[2:], other[:2])))
            union = area + other_area - intersection
            if union > 0 and intersection / union >= 0.95:
                if abs(score - actual["confidence"][index]) <= 0.02:
                    unused.remove(index)
                    break
        else:
            raise ValueError(f"No matching ONNX detection for COCO category {label}")


def prepare_repositories(root: Path) -> None:
    for device in ("cpu", "cuda"):
        repository = root / "repositories" / device
        model_dir = repository / "rfdetr-small"
        model_dir.mkdir(parents=True, exist_ok=True)
        extra: dict[str, Any] = {
            "providers": ["CPUExecutionProvider"],
            "session_options": {
                "intra_op_num_threads": 2,
                "inter_op_num_threads": 1,
            },
            "session_config_entries": {
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            },
        }
        if device == "cuda":
            extra["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            extra["provider_options"] = [{"device_id": 0}, {}]
        write_json(
            model_dir / "model-settings.json",
            {
                "name": "rfdetr-small",
                "implementation": "mlserver_onnx.OnnxModel",
                "max_batch_size": 0,
                "max_batch_time": 0,
                "parameters": {"uri": "../../../model/model.onnx", "extra": extra},
            },
        )
        write_json(
            repository / "settings.json",
            {
                "host": "127.0.0.1",
                "http_port": 18080,
                "grpc_port": 18081,
                "metrics_port": 18082,
                "metrics_endpoint": None,
                "parallel_workers": 1,
                "model_operation_timeout": 300,
                "grpc_max_message_length": 32 * 1024 * 1024,
                "access_log": False,
            },
        )


def export(root: Path, reuse_export: bool = False) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from rfdetr import RFDETRSmall

    if importlib.metadata.version("rfdetr") != VERSION:
        raise ValueError(f"Use the pinned RF-DETR {VERSION} export environment")
    torch.set_num_threads(32)
    torch.set_num_interop_threads(1)
    weights = root / "weights" / "rf-detr-small.pth"
    download(WEIGHTS_URL, weights)
    if checksum(weights, "md5") != WEIGHTS_MD5:
        raise ValueError("Official checkpoint checksum mismatch; refusing to load it")
    image = root / "sample" / "000000039769.jpg"
    download(IMAGE_URL, image)
    download(LICENSE_URL, root / "RF-DETR-LICENSE")
    tensor = normalize(image)
    np.save(root / "sample" / "input.npy", tensor, allow_pickle=False)
    tensor.tofile(root / "sample" / "input.fp32")
    print("Loading official weights and computing PyTorch reference", flush=True)
    model = RFDETRSmall(pretrain_weights=str(weights), device="cpu")
    network = model.model.model.eval()
    with torch.inference_mode():
        raw = network(torch.from_numpy(tensor))
    reference = {
        "dets": raw["pred_boxes"].cpu().numpy(),
        "labels": raw["pred_logits"].cpu().numpy(),
    }
    np.savez(root / "reference" / "pytorch.npz", **reference)
    reference_detections = decode(reference, image)
    write_json(root / "reference" / "pytorch-detections.json", reference_detections)
    from PIL import Image

    with Image.open(image) as sample_image:
        image_size = list(sample_image.size)
    artifact = root / "model" / "model.onnx"
    if not reuse_export or not artifact.is_file():
        print("Exporting fixed-shape FP32 ONNX, batch 1, opset 17", flush=True)
        artifact = model.export(
            output_dir=str(root / "model"),
            output_name="model",
            batch_size=1,
            dynamic_batch=False,
            opset_version=17,
            verbose=False,
        )
    graph = onnx.load(str(artifact), load_external_data=True)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 32
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
    )
    names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(names, session.run(names, {"input": tensor})))
    expected_shapes = {"dets": (1, 300, 4), "labels": (1, 300, 91)}
    for name, shape in expected_shapes.items():
        if outputs[name].shape != shape or not np.isfinite(outputs[name]).all():
            raise ValueError(f"Invalid output {name}: {outputs[name].shape}")
    if session.get_inputs()[0].shape != [1, 3, 512, 512]:
        raise ValueError("Unexpected exported input shape")
    np.savez(root / "reference" / "onnx-cpu.npz", **outputs)
    detections = decode(outputs, image)
    compare_detections(reference_detections, detections)
    write_json(root / "reference" / "onnx-cpu-detections.json", detections)
    write_json(
        root / "sample" / "request.json",
        {
            "id": "rfdetr-small-smoke",
            "inputs": [
                {
                    "name": "input",
                    "shape": list(tensor.shape),
                    "datatype": "FP32",
                    "data": tensor.reshape(-1).tolist(),
                }
            ],
            # MLServer's default selects only the first ONNX output.
            "outputs": [{"name": name} for name in names],
        },
        compact=True,
    )
    prepare_repositories(root)
    metadata = {
        "model": "RF-DETR Small, official COCO checkpoint",
        "rfdetr_version": VERSION,
        "resources_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "rfdetr",
                "torch",
                "torchvision",
                "onnx",
                "onnxruntime",
                "numpy",
            )
        },
        "checkpoint": {
            "url": WEIGHTS_URL,
            "md5": WEIGHTS_MD5,
            "sha256": checksum(weights),
        },
        "sample": {
            "url": IMAGE_URL,
            "sha256": checksum(image),
            "image_size_wh": image_size,
        },
        "onnx": {
            "path": "model/model.onnx",
            "sha256": checksum(artifact),
            "bytes": artifact.stat().st_size,
            "opset": 17,
            "inputs": [
                {"name": value.name, "shape": value.shape, "type": value.type}
                for value in session.get_inputs()
            ],
            "outputs": [
                {"name": value.name, "shape": value.shape, "type": value.type}
                for value in session.get_outputs()
            ],
            "initializer_elements": sum(
                int(np.prod(value.dims)) for value in graph.graph.initializer
            ),
        },
        "normalization": {"scale": "pixel / 255", "mean": MEAN, "std": STD},
        "input_bytes": tensor.nbytes,
        "rest_request_bytes": (root / "sample" / "request.json").stat().st_size,
        "output_bytes": sum(value.nbytes for value in outputs.values()),
        "cpu_validation": {
            "onnx_checker": "passed",
            "detection_parity": (
                "passed: same count/classes, IoU >= 0.95, score delta <= 0.02"
            ),
            "detections_at_0_5": len(detections["class_id"]),
            "raw_max_absolute_difference": {
                name: float(np.max(np.abs(outputs[name] - reference[name])))
                for name in names
            },
        },
    }
    project_dir = Path(__file__).resolve().parent
    # Export dependencies live beside this script; the serving environment
    # lives at the resources checkout root and uses a separate MLServer checkout.
    for name, directory in (
        ("export-env", project_dir / "export-env"),
        ("serving-env", project_dir.parent),
    ):
        for filename in ("pyproject.toml", "poetry.lock"):
            shutil.copy2(
                directory / filename,
                root / "provenance" / f"{name}-{filename}",
            )
    for filename in (
        "prepare.py",
        "serve.py",
        "validate.py",
        "test_prepare.py",
        "README.md",
    ):
        source = project_dir / filename
        if source.is_file():
            shutil.copy2(source, root / "provenance" / filename)
    write_json(root / "manifest.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-export", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if (root / "manifest.json").exists() and not args.reuse_export:
        raise SystemExit("Bundle already prepared; choose a new output directory")
    for name in ("weights", "sample", "model", "reference", "provenance", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for variable, path in {
        "RF_HOME": root / "weights",
        "HF_HOME": root / "cache" / "huggingface",
        "TORCH_HOME": root / "cache" / "torch",
        "MPLCONFIGDIR": root / "cache" / "matplotlib",
    }.items():
        os.environ[variable] = str(path)
    result = export(root, reuse_export=args.reuse_export)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
