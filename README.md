# MLServer performance resources

Model preparation, a small reviewer benchmark, and the CPU/GPU performance
reports. This is the orphan `resources` branch: it contains no MLServer server
code and is independent of the implementation PR's history.

| Directory | Contents | Start here |
| --- | --- | --- |
| [`model-preparation/`](model-preparation/) | RF-DETR Small downloads, pinned ONNX export, normalized COCO input, serving and correctness checks | [Preparation guide](model-preparation/README.md) |
| [`benchmarking/`](benchmarking/) | One-command REST or raw-gRPC load test against an already-running model | [Benchmark guide](benchmarking/README.md) |
| [`reports/cpu/`](reports/cpu/) | CPU ONNX/REST study, including the single-instance OVMS comparison | [CPU report](reports/cpu/PERFORMANCE_REDESIGN.md) |
| [`reports/gpu/`](reports/gpu/) | RF-DETR Small REST and gRPC study on one RTX 4070 | [GPU report](reports/gpu/GPU_PERFORMANCE_STUDY.md) |

The reports preserve the measured studies and their diagram prompts. They
describe different workloads; the CPU OVMS measurements are not GPU results.
The quick benchmark is a serving check. The [study guide](benchmarking/STUDY_GUIDE.rst) adds timed sweeps, raw samples, tuning and profiling instructions.
Raw captures, generated requests, model weights and model binaries are not
checked in. The preparation script downloads the public assets and generates
the fixture locally.

## Checkouts and environments

Use two sibling checkouts. Keep the implementation branch in `MLServer` and
this branch in `MLServer-resources`; the directory named `MLServer` is required
by the relative Poetry dependencies below. For PR 356, the implementation is
on `winning-combination`:

```bash
git clone --branch winning-combination \
  https://github.com/RH-steve-grubb/MLServer.git MLServer
git clone --single-branch --branch resources \
  https://github.com/RH-steve-grubb/MLServer.git MLServer-resources
cd MLServer-resources

poetry env use python3.12
poetry install
```

Existing checkouts can be used with the same layout. The root Poetry
environment installs the sibling MLServer and ONNX runtime as editable
dependencies, with PyZMQ, ONNX Runtime GPU 1.23.2 and CUDA/cuDNN packages.
It does not install PyTorch. A compatible NVIDIA driver is needed to run CUDA
inference; the load client itself does not need a GPU.

The separate [`model-preparation/export-env/`](model-preparation/export-env/)
pins the CPU PyTorch/RF-DETR stack for exporting the model. Keep the export and
serving environments separate. Both use Python 3.12; the measured study used
3.12.13 with the GIL enabled.

Follow the preparation guide once, start the server, then use the benchmark
guide. Set performance options in the MLServer repository's `settings.json`
and the model's `model-settings.json`; see the implementation branch's
[performance options](https://github.com/RH-steve-grubb/MLServer/blob/winning-combination/docs/user-guide/performance.md).
These resources do not automatically enable the optimizations.

## Utility checks

```bash
make lint
make test
```

These check the resource scripts and their loopback REST/gRPC clients. They do
not download the model, run a performance study or launch GPU inference.

## Reproducible studies

- [Timed sweeps, payloads, accounting and worker/thread tuning](benchmarking/STUDY_GUIDE.rst)
- [Root, CPU/GIL, host-stage and CUDA/NVTX profiling](benchmarking/PROFILING.rst)
- [OpenShift predictor and route measurements](benchmarking/OPENSHIFT.rst)

These guides and tools are reusable measurement resources. They do not choose
implementation changes or define a project acceptance plan.
