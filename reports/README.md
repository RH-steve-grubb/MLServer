# Performance reports

- [`cpu/PERFORMANCE_REDESIGN.md`](cpu/PERFORMANCE_REDESIGN.md): REST/JSON and
  CPU ONNX inference, including Python/transport experiments and a
  resource-matched single-instance OVMS comparison.
- [`gpu/GPU_PERFORMANCE_STUDY.md`](gpu/GPU_PERFORMANCE_STUDY.md): RF-DETR Small
  over REST and raw gRPC on one RTX 4070, including input representation,
  worker JSON, ZeroMQ and CUDA execution changes.

The reports preserve the original study results and rendering prompts for
figures. The public benchmark client provides a simpler manual check; it is
not the root profiling harness used to produce those studies. Each report
states its workload and measured scope. No GPU OVMS comparison was performed.
