# Run a manual serving benchmark

[`inference.py`](inference.py) sends a prepared V2 JSON request to an
already-running model. Use the root Poetry environment, or an existing MLServer
Python environment. No root privileges, profiler or GPU access are needed by
the client.

After [preparing the RF-DETR fixture](../model-preparation/README.md) and starting
its server, run this from the resources checkout root:

```bash
poetry run python benchmarking/inference.py \
  --model rfdetr-small --request "$HOME/rfdetr-small-bundle/sample/request.json" \
  --port 18080 --requests 1000 --concurrency 16 --output rest-result.json
```

For raw-tensor gRPC, use the same JSON request:

```bash
poetry run python benchmarking/inference.py \
  --model rfdetr-small --request "$HOME/rfdetr-small-bundle/sample/request.json" \
  --protocol grpc --port 18081 --requests 1000 --concurrency 16 \
  --output grpc-result.json
```

The client performs 32 warm-up requests, then measures 1,000 completed requests
with at most 16 outstanding. It prints and optionally saves requests/second,
p50/p95/p99 latency, payload size and hash, and the completed request count.
Change `--requests`, `--concurrency` or `--warmup` as needed. Use a new result
filename for each run.

`--host` defaults to `127.0.0.1`. Without an explicit port, REST uses 8080 and
gRPC uses 8081; the prepared RF-DETR repository instead uses the 18080/18081
ports shown above. `--model` and `--request` can name another loaded model and
its V2 request; the client is not specific to RF-DETR.

Request preparation and JSON-to-protobuf conversion happen once before timing.
gRPC sends raw binary tensors. Both paths read the complete response without
decoding its contents in the measured loop. Request errors make the command
fail; outstanding requests drain before it exits. Validate numerical results
separately before benchmarking.

To compare an optimization, run with it disabled, enable it in the server or
model settings, restart MLServer, and repeat the same command. Hold the model,
request, worker/thread counts, client concurrency and hardware resources fixed.
Alternate configurations for at least three rounds and also check concurrency
1. Test REST and gRPC separately. Use representative payload sizes: the RF-DETR
input is 3 MiB as FP32 or about 15 MB as JSON. A tiny request does not exercise
the same conversion costs.

The client leaves the server running. It measures serving throughput and client
latency; it does not measure server CPU, GPU utilization or GIL waiting. Reserve
CPU capacity for the client, or run it on a separate machine, when checking a
fast server. The [reports](../reports/README.md) describe the fuller studies.

For timed sweeps, per-request samples, multiple clients, HTTPS routes, server
accounting, and worker/thread tuning, use the [study guide](STUDY_GUIDE.rst).
It also links the root profiling and OpenShift how-tos.
