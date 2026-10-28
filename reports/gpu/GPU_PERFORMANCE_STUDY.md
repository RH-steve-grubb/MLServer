# RF-DETR GPU serving: REST and gRPC performance

## 1. Purpose

This study asks how much MLServer's serving path costs when an object detector
runs on a GPU, and which changes help. A fast model is not enough if the server
cannot keep it supplied with work. We measured the original path, changed one
part at a time, and repeated the measurements.

The largest improvement comes from preserving binary gRPC input as numeric
buffers instead of expanding it into Python floats. REST improves by moving
JSON work into model workers, but its much larger messages and numeric
conversion remain expensive. CUDA tuning reduces host CPU use; it does not
solve the remaining REST bottleneck.

All reported serving results use one instance and one RTX 4070. Multiple
workers belong to that instance; there is one frontend endpoint, rather than
several complete servers behind a load balancer. Python remains 3.12.13 with
the GIL enabled throughout.

## 2. Original request flow

Uvicorn receives REST connections and drives the ASGI application. FastAPI
routes inference requests; orjson creates Python dictionaries, lists and
numbers, and Pydantic validates the resulting request. Declaring an input
`FP32` does not make its JSON representation binary floats.

gRPC receives a Protocol Buffer request instead. Its raw input field already
contains binary FP32 values, but the original converter calls
`list(struct.unpack(...))`. That creates the same kind of Python-number
collection that the REST path produces.

Both paths then send an inference message through a multiprocessing queue.
Pickle serializes the Python objects; a pipe carries the message to a worker.
The worker reconstructs NumPy arrays, calls ONNX Runtime, and copies inputs to
the GPU. After inference, outputs return to CPU memory, become Python lists,
and travel back through IPC. The frontend formats JSON or a gRPC response.

ONNX inference is synchronous within each worker. Its Python method being
`async` does not make the native inference call asynchronous. Separate
workers provide request concurrency; a busy worker does not stop the other
workers. The frontend can await their results, but its own conversion and
copying still take CPU time.

Python object creation, validation, serialization and cleanup encounter each
process's GIL. Queue feeder and response-reader threads can compete for the
frontend's lock. Native inference can release the GIL while the calling
worker still waits for the GPU. These are different costs and need separate
measurements.

**Diagram placeholder: original flow**

Render a left-to-right engineering diagram with two input lanes. REST enters
Uvicorn/FastAPI, then JSON parsing and Pydantic validation. Raw gRPC enters
protobuf conversion, then binary-to-Python-float expansion. Both converge on
Python lists, Queue/pickle/pipe IPC, worker NumPy construction, ONNX Runtime,
CPU-to-GPU transfer and GPU execution. Draw the reverse response path through
CPU outputs, Python lists and IPC. Mark frontend and worker process boundaries,
each with its own GIL. Label JSON's large text payload separately from raw
gRPC's binary tensor. Do not depict the GIL as a single lock shared by processes.

## 3. Workload and profiling

The baseline is ODH master at commit `11e878839379`. Production MLServer and
ONNX runtime source remains unchanged; opt-in study adapters supply the
experimental paths.

RF-DETR Small uses a roughly 114 MB ONNX model and normalized RGB FP32 input
of shape `[1,3,512,512]`. The sample shows two cats and two remote controls
on a pink couch. Image decoding, resizing and normalization occur before
timing. The timed response includes both raw outputs: `dets [1,300,4]` and
`labels [1,300,91]`. Detection postprocessing is checked outside timing.

The same input contains 786,432 values, occupying 3 MiB in memory. Its REST
request is 15.26 MB; raw gRPC needs 3.15 MB. Responses are approximately
542 KB and 114 KB respectively. This study therefore measures normalized
tensor serving, rather than JPEG ingestion or an application returning only
a few filtered detections.

One instance shares fourteen physical CPU cores and their SMT siblings, a
16 GiB RAM limit and one GPU. Load generators use two separate physical
cores. ONNX Runtime GPU is 1.23.2. Model, batch size 1, FP32 tensors and CUDA
Execution Provider stay fixed, including its existing TF32 policy. There is
no adaptive batching, TensorRT, MIG/MPS, KServe, ingress proxy or TLS.

Root controls resource limits and profiling; servers and clients run as the
normal user. Unprofiled, alternating-order trials supply throughput, latency
and CPU/request. CPU stack sampling, validated GIL probes, correlated host
stages and Nsight CUDA/NVTX timelines explain the costs. Each profiling pass
has ordinary before/after controls. Startup, pilot traffic and graph capture
are excluded from measured traces.

CPU milliseconds per request sum server CPU consumption across processes;
they are not request latency. CUDA activity is the union of this model's
recorded kernels, copies and memsets, not SM occupancy. Nested stage spans
overlap and must not be added.

The original model performs 638 CUDA kernels per image, totaling about
6.3 ms. Yet model device activity occupies only 10.3% of the REST capture
and 18.7% of gRPC's. The GPU is mostly waiting for the serving path.

Stock gRPC request conversion costs 17.75 CPU ms, including 13.61 ms in raw
unpacking. Worker NumPy input conversion costs another 19.28 ms. REST workers
pay a similar NumPy cost. Allocations, float/list lifetimes, serialization
and memory copying accompany that work.

GIL contention is observable, especially in the stock gRPC frontend. However,
counts of waiting acquisitions and summed thread-wait time are not percentages
of request latency. The evidence points first to representation and process
boundaries, rather than assuming that removing the GIL is sufficient.

**Chart placeholder: initial bottleneck**

Create two side-by-side horizontal bar charts with independent axes.

Left panel: title "Model CUDA activity". Show REST 10.28% and gRPC 18.66%.
Label the horizontal axis "Capture time with model CUDA activity (%)", with
a 0--100% scale. Add the note "Recorded kernels, copies and memsets; not SM
occupancy."

Right panel: title "Stock gRPC CPU conversion costs". Show request conversion
17.75 and worker NumPy input conversion 19.28. Label the horizontal axis
"CPU ms/request", with a 0--25 scale. Display each bar's exact value. Add only
this annotation: "Request conversion includes 13.61 CPU ms/request of raw-input
unpacking."

Renderer instructions, not image text: raw unpacking is part of the 17.75
value, so it must not appear as an additional bar or stacked cost. Each panel
has its own units; there is no secondary axis or GPU milliseconds axis.

## 4. How the design evolved

We explored representation changes, worker-side JSON, Queue versus ZeroMQ,
worker/thread sizing, I/O binding, CUDA Graph replay and GPU wait policy.
The measurements progressively separated work worth removing from work worth
distributing. They did not produce one transport or engine that wins under
every load.

The first change preserved raw gRPC numeric buffers across dispatch. Workers
still perform their ordinary final NumPy decoding, but the large float-list
round trip disappears. At identical W2/T1 settings and concurrency 16,
throughput rises from 25.64 to 117.30 RPS, while CPU/request falls from
91.37 to 24.67 ms. This is work removal, rather than extra replicas.

For REST, we moved the process boundary earlier: the frontend dispatches the
original JSON bytes before parsing their numeric contents. The selected
worker performs parsing, validation, inference preparation and response JSON
encoding. At identical W4/T2 settings, throughput rises from 14.72 to
27.34 RPS. CPU/request instead rises from 115.60 to 183.27 ms. Distributing
work across processes relieves the frontend, but does not eliminate that work.

ZeroMQ then replaces inference Queue/pipe transport with local IPC sockets.
It retains pickle and round-robin dispatch; it is not shared-memory or
zero-copy transport. Its native I/O threads avoid the Python queue/executor
thread contention observed earlier. Confirmed gRPC throughput rises from
117.72 to 144.02 RPS at W2/T2. REST gains little sustained throughput, but
saves CPU. At concurrency 1 it regresses from about 10.44 to 8.89 RPS, an
important reason to retain transport choice rather than mandate ZeroMQ.

I/O binding alone gives no useful model-only gain. Combining persistent
device buffers with CUDA Graph replay reduces host launch work. Blocking
synchronization lets the calling thread wait without consuming as much CPU.
For gRPC, graph/blocking raises throughput from 142.50 to 150.38 RPS and
reduces CPU/request from 21.17 to 13.67 ms in matched trials.

Provider receipts already show full ORT graph optimization, enabled CPU
arena/memory patterns and maximum cuDNN convolution workspace. Those settings
are retained, along with NCHW input. An internal NHWC-kernel preference fails
the strict raw-output parity check and is excluded from accepted timings.

Two protobuf-copy alternatives are also tested. Metadata-only copying
regresses under load. Direct injection cuts conversion CPU roughly in half,
but short serving trials are unstable and mostly worse. Both remain disabled.
The full protobuf clone they target belongs to our experimental adapter,
rather than being a newly discovered upstream operation.

Finally, the matched REST CUDA comparison finds nearly identical throughput.
Graph/blocking saves about 4% CPU at concurrency 16, with additional GPU
memory. JSON and frontend byte handling continue to dominate. We retain the
ordinary engine as the REST throughput reference.

## 5. Resulting architecture and performance

For REST, the frontend resolves the requested model/version, receives the
body and dispatches bytes to its worker pool. Workers parse/validate JSON,
construct NumPy input, run the model and return encoded response bytes.
FastAPI and Uvicorn remain in place.

For raw gRPC, conversion preserves numeric buffers and protocol metadata.
Those buffers cross ZeroMQ IPC to workers without constructing hundreds of
thousands of Python floats. The selected GPU engine retains device buffers,
refreshes their contents for every request and replays the captured graph.
Both outputs return to CPU memory; response construction still has costs.
Required fresh-input H2D transfers are not eliminated.

**Diagram placeholder: final flow**

Draw REST and gRPC as two separate protocol panels. Within each panel, use
columns for frontend process, ZeroMQ IPC, worker process and GPU. Draw requests
left to right on the upper row and responses right to left on a second row.
Make process boundaries and arrow directions explicit. Label REST W8/T2 and
gRPC W2/T2 as separate measured configurations, each with one frontend endpoint
and one shared GPU; they are not concurrent pools in one measurement.

REST request row: frontend receives JSON bytes and resolves model/version;
pickle and ZeroMQ carry the request to a worker. Inside the worker, show JSON
parsing, Pydantic validation, NumPy input construction and ONNX Runtime,
followed by fresh CPU-to-GPU input transfer and GPU inference.

REST response row: GPU output returns to worker CPU memory. Inside the worker
boundary, show output arrays to inference response, response validation and
JSON encoding, then a reply containing finished JSON bytes and headers.
The worker pickles that reply for ZeroMQ IPC. The frontend unpickles it and
performs only HTTP response wrapping and byte delivery on this return row.
Use the visible frontend label "HTTP wrapper + send JSON bytes". All REST
response JSON encoding belongs visibly inside the worker boundary.

gRPC request row: frontend converts metadata while preserving raw FP32 buffers;
pickle and ZeroMQ carry them to the worker. Show worker NumPy decoding, ONNX
Runtime and fresh CPU-to-GPU input transfer before GPU inference.

gRPC response row: GPU output returns to worker CPU arrays, then becomes output
lists and an inference response inside the worker. That response crosses
pickle/ZeroMQ IPC. Inside the frontend, show raw-output packing, protobuf
response construction and gRPC delivery.

Each worker owns its model/session and CUDA context. Show persistent device
buffers, CUDA Graph replay and blocking waits as optional GPU engine settings.
Keep CPU-to-GPU input and GPU-to-CPU output transfers visible.

Renderer instructions, not image text: the lower row must show the return
path through worker-side REST JSON encoding before IPC. Keep the REST frontend
return row limited to its HTTP wrapper and delivery box. The gRPC frontend
return row retains its response conversion boxes. Pickle serializes IPC
messages; it is separate from JSON encoding and protobuf conversion.

The following gRPC progression uses repeated C16 results at W2/T2. It combines
staged matched experiments, rather than claiming one interleaved four-way
stock-versus-final trial.

**gRPC progression, sixteen concurrent requests**

| Configuration                         | RPS    | CPU ms/request |
|---------------------------------------|--------|----------------|
| Stock, Queue                          | 23.68  | 92.67          |
| Compact input, Queue                  | 117.72 | 24.42          |
| Compact input, ZeroMQ                 | 144.02 | 20.92          |
| Compact input, ZeroMQ, graph/blocking | 150.38 | 13.67          |

The selected gRPC path also reaches 46.13 RPS with one outstanding request,
at 16.50 CPU ms/request and pooled p95 23.00 ms. Its C16 pooled p95 is
approximately 116 ms. The 46 RPS figure describes gRPC, not REST.

The final REST engine comparison holds W8/T2, worker JSON and ZeroMQ fixed.
Rates and CPU costs are medians of three trials; p95 pools measured requests.

**REST CUDA comparison**

| Concurrent requests | Engine         | RPS   | CPU ms/request | p95       |
|---------------------|----------------|-------|----------------|-----------|
| 1                   | Ordinary Run   | 8.81  | 114.07         | 116.23 ms |
| 1                   | Graph/blocking | 8.93  | 106.79         | 114.76 ms |
| 16                  | Ordinary Run   | 29.87 | 167.36         | 560.87 ms |
| 16                  | Graph/blocking | 29.70 | 159.96         | 570.96 ms |

Four-client controls give similar rates. CUDA captures show roughly 21%
model device activity for both REST engines. ONNX calling-thread CPU falls
9.04 to 1.77 ms/request, while JSON decoding/validation and NumPy input
conversion still cost roughly 75--79 ms. Frontend CPU remains near
39 ms/request, with its main thread approximately fully occupied. Body-receive
wall time includes async suspension and competing work; it is not a pure
network or CPU timer.

The selected gRPC CUDA profile shows about 83% model activity. Better feeding
also raises its natural GPU boost clocks compared with lightly loaded REST;
the shorter kernel times do not represent a new model or precision policy.
Validated ZeroMQ GIL captures observe no GIL condition waits. The lock remains
enabled, and serial frontend work remains even without observed contention.

**Chart placeholder: throughput and CPU progression**

Plot the four gRPC rows above in aligned panels: RPS and CPU ms/request, with
exact W2/T2/C16 labels. Add a separate REST panel showing ordinary versus
graph/blocking at C1 and C16 from the REST table. Avoid joining results across
protocols or calling the gRPC progression one simultaneous paired trial.
Annotate REST's principal gain as worker-side JSON and its CUDA increment as
CPU savings with essentially unchanged throughput.

### Worker and thread sizing

We screened 1, 2, 4 and 8 workers with 1, 2 or 4 ORT intra-op threads, plus
one worker retaining native defaults. Explicit thread settings use inter-op
1 and disable ORT spinning. Short-screen winners were followed by repeated
trials, client-topology controls and resource measurements.

REST benefits from extra parsing processes, but the return diminishes. In
the Queue comparison, W8/T2 adds only 5.5% throughput over W4/T2 while cgroup
peak memory rises from 4.16 to 7.58 GiB. Four workers retain most capacity;
eight are the tested higher-throughput setting. Each worker also loads its
own GPU model/session. Graph/blocking adds roughly 288 MiB of GPU memory
across eight workers without raising REST throughput materially.

gRPC needs only two workers here. Four or eight do not provide a clear gain.
One versus two intra-op threads is effectively tied in the earlier repeated
tests; T2 is held fixed for later transport/engine comparisons. More workers
are not automatically more GPU concurrency or useful capacity.

### Delivery sequence

These are measured prototypes, not a complete production patch. Deliver the
shared representation/dispatch changes first, preserving model/version
selection, request IDs, headers, errors, requested outputs and unsupported
input behavior. Keep numeric buffers internal rather than hand-editing the
generated public protocol types.

Land the adaptive-batching work before integrating this redesign. Forwarding
raw REST bodies changes where parsed requests exist; it must not silently
bypass main-process batching. Retain the existing path where batching,
streaming or custom codecs require it until that integration is implemented.
The readiness work has little interaction with this request path and need
not determine the delivery order.

Make ZeroMQ and GPU engine policy separate changes so their tradeoffs remain
reviewable. Graph replay needs stable shapes, buffers owned by the session,
fresh-input validation and serialization of concurrent calls sharing those
buffers. Original/mirrored/original REST checks passed on every one of eight
workers under both engines, with exact agreement for both raw outputs.

The shared serving improvements may benefit other runtimes, but the measured
GPU engine result applies to this ONNX workload. Customer GPUs, mixed-model
traffic and the 20--100 loaded-model requirement remain separate qualification
work; this study deliberately isolates one resident model.

## 6. Conclusion

The main limitation is how data is represented and moved around inference.
The original raw gRPC path unnecessarily turns binary numbers into Python
objects. Preserving buffers removes most of that work. ZeroMQ improves its
dispatch path, and graph replay with blocking waits reduces host overhead.
The result is approximately 150 gRPC requests/second from one instance.

REST reaches approximately 30 requests/second. Moving JSON to workers
substantially improves throughput, but its 15 MB requests still pass through
a busy frontend and require expensive numeric conversion. Faster CUDA calls
save some CPU without materially changing that rate. For normalized image
tensors, raw gRPC is the stronger measured path when the deployment can use it.

## Appendix A. Where further work could help

For REST, the next substantial change must address input byte handling or
numeric representation. A native frontend or parser that constructs numeric
buffers directly could remove work that moving JSON between Python processes
merely redistributes. Shared-memory dispatch could reduce large IPC copies,
but would introduce explicit ownership and reclamation. Neither was measured
in this GPU study.

gRPC still has copying, output-list conversion and response/IPC work. These
are smaller targets now that the GPU is much busier. Removing them cannot
remove the detector's remaining GPU work. Larger total capacity can then use
additional serving replicas, but no replica scaling is included in these
single-instance results.

No GPU OVMS comparison was run. The earlier CPU study uses a different model
and workload; its OpenVINO figures must not be reused as a GPU comparison.
TensorRT, other precision policies and GPU sharing are also separate
experiments, not explanations for the gains measured here.
