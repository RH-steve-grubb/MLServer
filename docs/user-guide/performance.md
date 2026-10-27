# Inference Performance Options

MLServer provides optional paths for reducing request conversion, frontend JSON
processing and inference IPC overhead. Each option can be enabled independently.
Choose settings using your model, request sizes and concurrency; compare
throughput, tail latency, CPU use and memory before adopting a configuration.

## Server Settings

These settings apply to the whole server and belong in `settings.json` at the
root of the model repository. They can also be set with environment variables.

| Setting | Default | Environment variable |
| --- | --- | --- |
| `grpc_raw_input_buffers` | `false` | `MLSERVER_GRPC_RAW_INPUT_BUFFERS` |
| `rest_worker_json` | `false` | `MLSERVER_REST_WORKER_JSON` |
| `parallel_inference_transport` | `"queue"` | `MLSERVER_PARALLEL_INFERENCE_TRANSPORT` |

For example, the following enables all three options with two worker processes:

```json
{
  "parallel_workers": 2,
  "grpc_raw_input_buffers": true,
  "rest_worker_json": true,
  "parallel_inference_transport": "zmq"
}
```

Install the optional `zmq` extra before selecting that transport. In a
Poetry-managed application, use:

```bash
poetry add "mlserver[zmq]"
```

From an MLServer source checkout, use `poetry install -E zmq` instead. Then start
the repository normally:

```bash
poetry run mlserver start .
```

Restart MLServer after changing server settings. Choose the worker count for
your CPU and model-memory budget; see [Parallel Inference](./parallel-inference.md).

### Raw FP32 gRPC Inputs

`grpc_raw_input_buffers=true` preserves unary gRPC input from
`raw_input_contents` as NumPy buffers through dispatch. It avoids constructing a
large list of Python floats in the frontend. The worker's codec still prepares
the final model input, and IPC still serializes and copies data.

Enable this only when the input codecs of models receiving raw FP32 requests
accept NumPy-backed tensor data, such as the ONNX runtime's default NumPy codec.
Custom runtimes that require lists should leave it disabled. Because the flag
is server-wide, check compatibility across all models served by the instance.

All inputs must have complete, correctly sized raw FP32 contents to use this
path. Other representations and streaming use standard conversion. Models
with batching settings recover list-based inputs before frontend batching.

### REST JSON in Workers

`rest_worker_json=true` forwards the original JSON body to the selected model's
worker. Parsing, validation and response JSON encoding run there, distributing
that CPU work across worker processes.

This applies to unary `/infer` routes, including versioned routes, for models
using parallel inference. Both `max_batch_size` and `max_batch_time` must be `0`
in the model settings for worker JSON processing. Batched models, models without
parallel workers and other media types retain frontend preparation. Generate
and streaming routes retain their existing behavior.

### ZeroMQ Inference Transport

`parallel_inference_transport="zmq"` uses local ZeroMQ IPC sockets for inference
requests and replies when parallel inference is enabled. It requires the optional
PyZMQ dependency provided by the `zmq` extra. Model load/unload messages continue
to use multiprocessing queues. Pickle serialization and round-robin worker
selection remain in use; this transport does not provide shared-memory or
zero-copy dispatch.

The JSON and gRPC conversion options work with either transport. Compare
`"queue"` and `"zmq"` at representative loads, including low concurrency. To
return to the default transport, set `parallel_inference_transport` to
`"queue"` and restart.

## ONNX CUDA Settings

The ONNX runtime also provides `cuda_graph` and `cuda_blocking_sync`, both
defaulting to `false`, under each model's `parameters.extra`. These settings
apply regardless of whether requests arrive over REST or gRPC.

See the [ONNX CUDA tuning guide](../runtimes/onnx.md#cuda-graph-replay-and-blocking-waits)
for a complete model configuration, environment variables, fixed-shape FP32
requirements and the scope of CUDA blocking waits within a process. CUDA Graphs
retain additional device buffers for each model session, so account for every
worker's GPU memory when choosing a worker count.
