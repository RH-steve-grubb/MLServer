# Prepare RF-DETR Small and the reference image

The study downloaded the official PyTorch checkpoint, then exported ONNX
locally. No training, quantization or TensorRT conversion was performed.

- [RF-DETR Small COCO checkpoint](https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth)
- [COCO validation image 000000039769.jpg](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000039769.jpg)
- [RF-DETR project](https://github.com/roboflow/rf-detr)

The 640x480 reference image shows two cats and two remote controls on a pink
couch. The script downloads both assets automatically. The COCO image retains
its original image license; the RF-DETR license is downloaded into the bundle.

## Export and prepare the input

From the resources checkout root, with Python 3.12 and Poetry installed:

```bash
resources_root="$PWD"
bundle_dir="$HOME/rfdetr-small-bundle"

poetry -C model-preparation/export-env env use python3.12
poetry -C model-preparation/export-env install --no-root
poetry -C model-preparation/export-env run python \
  "$resources_root/model-preparation/prepare.py" --output "$bundle_dir"
```

Choose a new output directory; a completed bundle is not overwritten by
default. Export runs on the CPU and uses 32 threads. The export environment
pins RF-DETR 1.11.2, PyTorch 2.9.0+cpu, torchvision 0.24.0+cpu, ONNX 1.21.0,
ONNX Runtime 1.23.2 and NumPy 2.2.6. The original Python was 3.12.13.

The script verifies the upstream checkpoint checksum, exports fixed-shape FP32
ONNX with batch 1 and opset 17, checks the graph, and compares ONNX detections
against PyTorch. It requires matching detection counts/classes, box IoU at
least 0.95 and confidence differences at most 0.02.

The generated bundle contains:

```text
model/model.onnx          Exported ONNX model
sample/000000039769.jpg   Original COCO image
sample/input.npy          Normalized NCHW tensor
sample/input.fp32         Same tensor as little-endian FP32 bytes
sample/request.json       Prepared V2 REST request, requesting both outputs
repositories/cpu/        CPU model and server settings
repositories/cuda/       CUDA model and server settings
reference/               PyTorch and ONNX reference outputs/detections
provenance/              Script snapshots and Poetry environment definitions
manifest.json            Actual versions, URLs, hashes, shapes and validation
```

The input is converted to RGB and scaled by 1/255, then resized to 512x512
using tensor bilinear interpolation with antialiasing disabled. Normalization
uses mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. The prepared tensor is already normalized; send it as
provided. JPEG decoding and preprocessing were outside the serving timings.

| Tensor | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input` | FP32 | `[1,3,512,512]` | Normalized RGB image, NCHW |
| `dets` | FP32 | `[1,300,4]` | Normalized center-x, center-y, width, height boxes |
| `labels` | FP32 | `[1,300,91]` | Class logits, not class IDs |

Request both `dets` and `labels`. The reference image produces six detections
above confidence 0.5. The 91 class slots use sparse COCO category IDs;
the final slot is real category 90, not a background slot to discard.

The original ONNX file is 114,429,235 bytes. Its SHA-256 is
`a018914e11d589b8b51154f00ed1103f148a320a3960f768bd44f57d55d0ab51`.
The original weights, image, model and version records are in
[`REFERENCE_MANIFEST.json`](REFERENCE_MANIFEST.json). A new export writes its
own manifest; exact ONNX bytes can differ across export environments, so the
script also checks numerical parity.

## Check serving correctness

Install the resources checkout's root environment as described in the
[overall guide](../README.md), then run:

```bash
poetry run python model-preparation/validate.py \
  --bundle "$HOME/rfdetr-small-bundle" --device cpu
poetry run python model-preparation/validate.py \
  --bundle "$HOME/rfdetr-small-bundle" --device cuda
```

Run these sequentially with no server already using ports 18080--18082.
Each check starts and stops its own MLServer instance, validates REST and raw
gRPC results against the reference, and records a timestamped summary. CUDA
validation checks real GPU execution. These are correctness checks, not
throughput benchmarks.

## Start the server

```bash
poetry run python model-preparation/serve.py \
  --bundle "$HOME/rfdetr-small-bundle" --device cuda
```

Use `--device cpu` for CPU inference. The server listens on localhost: REST
18080 and gRPC 18081. The launcher supplies the CUDA library paths to the
frontend and its spawned workers.

The generated settings use one inference worker, two ORT intra-op threads,
disabled idle spinning, and no adaptive batching. They leave all performance
options at their defaults. These are smoke-test settings; the study's selected
REST and gRPC configurations used eight and two workers respectively, both
with two ORT intra-op threads. Select the server and model options you want to
evaluate, restart the server, then use the [benchmark guide](../benchmarking/README.md).
