# MLServer performance redesign

## 1. Introduction

This report investigates performance limitations in MLServer’s REST inference path, focusing on internal queueing, serialization, and how work is divided between the frontend and inference workers. It traces requests through the implementation, presents profiling evidence, and evaluates architectural changes through controlled experiments.

The objective is to improve throughput and latency within one serving instance while preserving MLServer’s model-selection capabilities. Testing uses CPU-based ONNX inference and includes a resource-matched comparison with OpenVINO Model Server.

Not every experiment helped. The report explains the improvements and the unsuccessful alternatives, identifies the strongest tested configuration, and outlines the engineering and validation needed for production delivery.

This report focuses on REST’s JSON-specific costs. gRPC carries numeric values in binary Protocol Buffer messages, avoiding JSON scanning, decimal-string-to-floating-point conversion on input, and floating-point-to-string formatting on output. It also bypasses the FastAPI REST pipeline. However, MLServer still converts gRPC messages into Python objects and uses the same parallel-worker dispatch machinery. gRPC avoids those REST-specific costs, not every source of GIL contention or internal queueing. It was not benchmarked here.

## 2. The original request path

This walkthrough describes REST/JSON inference with parallel workers enabled and adaptive batching disabled. The data changes representation several times before the client receives a prediction.

### 2.a Architectural dataflow diagram

[Diagram placeholder]

[Illustration prompt: Create a clean technical architecture diagram on a light background. Show one “Frontend process” and three representative “Inference worker processes,” labelled 1, 2, and N. Inside the frontend, show “HTTP JSON bytes → Parse and validate → Python request objects → Model lookup and round-robin dispatch.” Outgoing arrows pass through “Pickle / per-worker request queue / pipe.” Each worker shows “Unpickle → Python objects → NumPy tensors → ONNX Runtime → NumPy outputs → Python response objects.” Return arrows pass through “Pickle / shared response queue / pipe” to the SAME frontend process, showing “Unpickle → Validate and format → JSON bytes → Client.” Label outgoing serialization “Queue feeder thread” and frontend response reception “Executor thread.” Note that each process has its own GIL; frontend threads share one GIL. Mark native ONNX computation “GIL released; worker event loop still occupied.” Caption: Most of the serving path moves Python objects, not compact numeric tensors.]

### 2.b Walking through a prediction

1. **Receive and validate.** The client sends JSON to `/v2/models/<model-name>/infer`. Uvicorn accepts the connection and parses HTTP; FastAPI routes the request to MLServer’s inference endpoint. MLServer’s JSON handler creates Python dictionaries, lists, and numbers, which Pydantic validates into an `InferenceRequest`. Declaring `FP32` does not make the transmitted data binary floats.

2. **Select the model and worker.** MLServer looks up an already-loaded model. It does not reload weights for each request. A parallel-model proxy packages the model name, version, request, and internal response ID. The dispatcher selects a ready worker using round robin.

3. **Cross the process boundary.** A multiprocessing queue’s feeder thread pickles the request objects and sends bytes through a pipe. The worker unpickles them into its own memory. The frontend waits for the response asynchronously, allowing its event loop to handle other requests. No shared tensor buffer is used.

4. **Build tensors and run inference.** The worker converts the numeric lists into NumPy arrays and calls ONNX Runtime. Native computation releases the GIL, but the call is synchronous: that worker’s event-loop thread remains occupied until inference finishes. Other worker processes can continue independently. [ONNX Runtime binding](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/python/onnxruntime_pybind_state.cc#L2438).

5. **Return the prediction.** The worker converts output arrays into Python lists and response objects, then pickles them into the pool’s shared response queue. A frontend executor thread reads and unpickles the reply, allowing the waiting request to resume. FastAPI/Pydantic validates and converts the response into JSON-compatible objects, MLServer’s renderer encodes the JSON bytes, and Uvicorn sends the HTTP response to the client.

The GIL matters around those representation changes. Each process has its own global interpreter lock. Within the frontend, the event-loop thread, queue feeder threads, and response-reader threads share that lock for Python-object work. Using a compiled library does not automatically avoid it: `orjson` holds the GIL during JSON parsing and encoding. [orjson documentation](https://github.com/ijl/orjson#deserialize).

That leaves two separate limits: how quickly workers execute predictions, and how quickly the frontend prepares and returns their work. Adding inference workers can improve the first without fixing the second.

## 3. Profiling results

### 3.a Method

The initial investigation used Python 3.12 on an AMD Ryzen 9 5950X: 16 physical cores, 32 logical CPUs, and 64 GiB RAM.

The synthetic FP32 ONNX model was a dense neural network with layer widths `54 → 1024 → 512 → 256 → 128 → 7`: 54 input features and seven output values per sample.

| Rows/request | Input shape | Approximate JSON body size |
|---|---|---:|
| 1 | `[1, 54]` | 1.2 KB |
| 100 | `[100, 54]` | 112 KB |
| 1,000 | `[1000, 54]` | 1.11 MB |

Rows are samples within one HTTP request. A 1,000-row payload is one request containing 1,000 samples, not 1,000 requests. Adaptive batching was disabled; these were client-provided multirow inputs, not batches formed by the server.

Three measurements answered different questions:

- Request timestamps showed where elapsed time accumulated.
- Linux `perf` CPU sampling identified work executing in the frontend.
- Validated probes recorded GIL acquisitions and whether they entered a wait.

Profiling runs were bracketed by unprofiled controls. GIL tracing reduced throughput by up to approximately 14%, so traced rates are not treated as serving-capacity measurements. The later resource-matched benchmarks appear in section 5.

### 3.b Numbers

**Requests backed up without a corresponding throughput gain.** In a one-worker diagnostic using 100-row requests:

| Concurrent requests | Requests/second | Mean response time | Dispatch-to-worker-entry |
|---|---:|---:|---:|
| 2 | 639.5 | 3.1 ms | 0.74 ms |
| 128 | 620.8 | 200.8 ms | 196.4 ms |

The native ONNX call remained approximately 1.15 ms. Most of the additional latency accumulated before the worker began processing.

With six workers and 64 concurrent 100-row requests, one instrumented run measured a 1.46 ms ONNX call within a 69.4 ms client round trip. Approximately 64.7 ms elapsed between the worker queueing its reply and the frontend finishing its response-queue read. That interval includes serialization, scheduling, and unpickling—not just time moving bytes through a pipe.

**Frontend threads competed for the GIL.** At concurrency 64, with six workers:

| Rows/request | Frontend GIL acquisitions/second | Acquisitions entering a wait |
|---|---:|---:|
| 1 | 25,462 | 29.0% |
| 100 | 15,935 | 34.0% |
| 1,000 | 4,354 | 22.8% |

These percentages describe acquisition calls, not the percentage of request time spent waiting. The traces do not establish a GIL-held-time percentage.

**Data handling consumed substantial CPU.** For 1,000-row requests, selected frontend CPU hotspots included JSON scanning at 20.0% of samples, Python object allocation/free at 10.8%, memory copying at 8.0%, and pickle serialization at 7.6%.

A separate Python 3.14 diagnostic, with the frontend GIL disabled, still placed approximately 90% of frontend CPU samples on its event-loop thread. Decimal-number parsing, formatting, and JSON scanning accounted for 35% in just three routines. That experiment used standard-library JSON, not the original orjson stack; it identifies remaining work, not an isolated Python-version effect.

### 3.c Charts

[Chart placeholder: More concurrency, more waiting]

[Illustration prompt: Create a clean two-panel technical chart on a light background. Title: “More concurrent requests increased latency, not throughput.” Both panels compare concurrency 2 and 128. Left panel: throughput, 639.5 and 620.8 requests/second. Right panel: mean response time, 3.1 and 200.8 milliseconds. Use bars with zero-based axes and exact value labels. Caption: Python 3.12, one inference worker, 100-row requests; illustrative early diagnostic. Do not invent intermediate measurements or error bars.]

[Chart placeholder: Frontend CPU hotspots]

[Illustration prompt: Create a horizontal bar chart titled “Selected frontend CPU hotspots.” Values: JSON scanner 20.0%; Python object allocation/free 10.8%; Python execution 9.7%; Memory copying 8.0%; Pickle serialization 7.6%. Label the axis “Share of frontend CPU samples.” Add a note: “Selected leaf-function hotspots—not complete processing-stage totals. Excludes blocked time and inference-worker CPU.” Caption: Python 3.12, six workers, 64 concurrent requests, 1,000 rows per request. Use a light background, readable labels, and no decorative imagery.]

### 3.d Interpretation

There were two limits. Synchronous inference occupied each worker’s event loop, while the shared frontend approached one CPU core doing request preparation, serialization, and response handling.

The GIL added competition between frontend threads, but it was not the whole problem. Removing it did not distribute CPU-heavy handlers across threads or eliminate conversion and copying.

The practical distinction is important: a queue shows where requests accumulate. It does not necessarily identify the work limiting capacity. The redesign therefore needed to address work placement and representation changes, not simply replace a queue or add inference workers.

## 4. Architectural evolution

We explored several promising options: distributing JSON processing, replacing multiprocessing queues with ZeroMQ, using a lighter HTTP frontend, and combining free-threaded Python with frontend JSON threads. Fused parsing and balanced worker scheduling offered further opportunities.

Testing led to one preferred configuration for this workload: retain FastAPI and Python 3.12, move JSON processing into the inference workers, and use asynchronous ZeroMQ communication. It was not the fastest for every request size, but it offered the strongest tested balance of throughput and tail latency with one fixed frontend.

### 4.a Follow the bottleneck

The starting question was whether internal queueing meant the transport was too expensive. MessagePack and compact NumPy buffers looked worth investigating. However, smaller messages did not translate into faster serving. Converting inputs into arrays before dispatch added work to the already-saturated frontend, while the workers still performed their normal tensor conversion.

That result changed the question: could we move the process boundary earlier, before the frontend constructed the large Python object graph?

The first experiment forwarded the original JSON bytes to workers for parsing and validation. A second also moved response encoding into workers. With the original FastAPI frontend and Queue/pickle transport still in place, the complete worker-JSON variant improved throughput by approximately 52%, 99%, and 40% for 1-, 100-, and 1,000-row requests.

This established the central design idea independently of a transport or framework change. The work had not disappeared; it was now distributed across processes instead of concentrated in one frontend.

### 4.b Build on the result

ZeroMQ had already shown an incremental improvement with the original request representation. We then tested it with worker-side JSON rather than assuming the gains would add together. It helped small and medium requests in that configuration, while large requests increasingly depended on worker capacity. Pickle and memory copies remained; this was not zero-copy dispatch.

Architecturally, ZeroMQ replaced the pipe-backed inference request and response queues with asynchronous local sockets—one request channel per worker and one shared response receiver in the frontend. This replaced the queue feeder-thread and executor-thread handoffs with sends and receives on the event loops. Model-lifecycle control queues were unchanged.

The aiohttp prototype offered another useful comparison. It improved small- and medium-request throughput, but some of its advantage came from compiled HTTP parsing. Enabling Uvicorn’s `httptools` parser recovered part of that benefit without replacing FastAPI. Later testing favoured the FastAPI configuration for large-request throughput and tail latency, making it the preferred compromise across the three request sizes.

As more work reached the workers, we tested worker counts and ONNX Runtime thread counts together. Simply adding processes was not enough. The final tuning improved the large-request result without sacrificing the small-request performance of the selected configuration.

### 4.c Keep the alternatives honest

Free-threaded Python with explicit frontend JSON threads was a genuine alternative: it substantially improved larger requests within its own controls. Removing the GIL alone was insufficient because CPU-heavy handlers still ran on one event-loop thread. However, this approach did not beat the worker-JSON configuration in the later resource-matched comparison.

Other improvements were conditional. Fused parsing reduced work in several controls but was slower in the Python 3.12/orjson inline comparison. Balanced scheduling improved tail latency in earlier threaded-frontend workloads, but showed no consistent benefit in the homogeneous worker-JSON tests. Likewise, ZeroMQ helped one architecture but reduced throughput when paired with the free-threaded frontend JSON pool.

The final comparisons used one serving instance, fixed CPU and memory budgets, a native load generator, and fresh repeated measurements. The tested Python 3.14 stack did not outperform the selected Python 3.12 configuration; dependency versions also changed, so that was not an isolated interpreter comparison.

The design emerged from those measurements: distribute expensive conversion work first, then tune communication and execution around it. Section 5 describes the resulting architecture and its benchmark results.

## 5. Final architecture

The selected prototype retains Uvicorn, FastAPI, and Python 3.12. The main change is where request processing happens: the frontend handles HTTP and routing, while inference workers perform JSON parsing, validation, tensor conversion, prediction, and response encoding.

### 5.a Architectural dataflow diagram

[Diagram placeholder]

[Illustration prompt: Create a clean technical architecture diagram on a light background. Enclose all server components within “One MLServer instance.” Show one “Frontend process: Uvicorn + FastAPI” containing “Receive JSON bytes → Select model/version and check readiness → Round-robin dispatch.” Connect it through “ZeroMQ IPC: pickled envelope containing JSON bytes” to three representative inference workers labelled “Worker 1,” “Worker 2,” and “Worker N.” Inside each worker show “Parse JSON and validate → Python request objects → NumPy tensors → ONNX Runtime → Validate and encode response JSON.” Return arrows pass through “ZeroMQ IPC: response envelope containing JSON bytes” to the frontend’s “Match response to waiting request → Send HTTP response.” Show a separate dashed control path labelled “Existing model loading, unloading, and worker lifecycle queues.” Note: “Tested configuration: 12 workers, two ONNX Runtime inference threads per worker, spinning disabled.” Caption: JSON processing is distributed across workers; pickle and memory copies remain.]

### 5.b Walking through the new flow

1. **Receive and route.** Uvicorn handles HTTP, and FastAPI selects the inference endpoint. The frontend resolves the requested model and version through the existing registry and checks readiness. It forwards the original body bytes without first constructing the numeric Python lists.

2. **Dispatch.** Round robin selects a ready worker. A small routing envelope carries the body, headers, and internal response ID through ZeroMQ. The frontend awaits the reply asynchronously.

3. **Process in the worker.** The worker parses and validates the JSON, performs the inference preparation, converts inputs into NumPy tensors, and runs ONNX Runtime. Each worker has its own GIL, so conversion work can proceed independently across processes.

4. **Encode and return.** The worker validates the prediction response and encodes it into JSON bytes. The frontend matches the reply to its waiting request and sends those bytes without rebuilding and formatting the prediction objects.

This removes substantial Python-object processing from the shared frontend. It does not eliminate JSON conversion, pickle, copying, or synchronous inference within each worker.

Worker-side JSON processing is a REST-specific optimization. ZeroMQ replaces a transport shared by REST and gRPC when parallel workers are enabled, so it could also improve gRPC performance by replacing queue feeder-thread and response-reader handoffs with asynchronous IPC. Whether that helps depends on the workload and remaining bottlenecks; a separate gRPC comparison is needed before claiming a benefit.

The main improvement targets MLServer’s shared REST/JSON serving path, not ONNX’s inference engine. Scikit-learn, XGBoost, and LightGBM also receive an `InferenceRequest` and return an `InferenceResponse`. In principle, moving JSON parsing, validation, and response encoding into their workers should distribute the same frontend work without changing their prediction algorithms.

The benefit depends on what limits the workload:

- **Frontend-limited:** fast predictions or substantial JSON payloads could benefit significantly.
- **Inference-limited:** expensive predictions may see little throughput improvement because workers are already fully occupied.
- **Resource-limited:** additional model copies and each backend’s native threads could exhaust memory or oversubscribe CPUs.

The tested worker and thread counts are tuning choices, not proposed universal defaults. More workers also mean more model copies and memory consumption—particularly important when one instance holds many models. The ONNX choice of 12 workers with two inference threads each does not transfer automatically; each backend needs its own tuning and benchmarks. Likewise, ZeroMQ’s contribution must be measured rather than assumed.

This approach does not require free-threading support from those libraries: the selected design uses separate, GIL-enabled Python 3.12 processes. The architectural idea applies across runtimes, but the measured speedups are established only for the tested CPU ONNX workload.

### 5.c Benchmark results

The comparison used the same synthetic FP32 ONNX model and identical REST/JSON requests. Each server received a budget of 14 physical cores, 28 logical CPUs, and 8 GiB RAM. The native load generator used two separate physical cores. Adaptive batching was disabled.

| Configuration | HTTP serving stack and parser | Inference execution |
|---|---|---|
| Original MLServer | FastAPI/Uvicorn with h11 | Six worker processes; one ONNX Runtime inference thread per worker |
| Redesigned MLServer | FastAPI/Uvicorn with httptools | Twelve worker processes; two ONNX Runtime inference threads per worker |
| Tuned OVMS | Native C++ REST interface | Six execution streams; fourteen CPU inference threads in total |

Both MLServer configurations used GIL-enabled Python 3.12 and round-robin dispatch, with ONNX Runtime spinning disabled. The original path used Queue/pickle with frontend JSON processing; the redesigned path used ZeroMQ/pickle with worker-side JSON processing.

The following results are means of three fresh confirmation runs at 64 concurrent requests. Each implementation used one serving instance; there were no horizontal replicas.

| Rows/request | Original MLServer requests/s | Redesigned MLServer requests/s | Tuned OVMS requests/s |
|---|---:|---:|---:|
| 1 | 1,886 | 4,540 | 67,184 |
| 100 | 978 | 2,992 | 5,033 |
| 1,000 | 179 | 371 | 504 |

The combined architectural and tuning changes improved MLServer throughput by approximately **2.4×, 3.1×, and 2.1×**, respectively.

Mean per-run p95 response times also improved:

| Rows/request | Original MLServer | Redesigned MLServer | Tuned OVMS |
|---|---:|---:|---:|
| 1 | 35.4 ms | 14.8 ms | 1.3 ms |
| 100 | 73.2 ms | 23.4 ms | 17.1 ms |
| 1,000 | 384.5 ms | 195.1 ms | 183.1 ms |

A later five-repeat confirmation reproduced the redesigned Python 3.12 throughput within approximately 1% for every request size.

These improvements do not remove overload queueing. In that later test, increasing concurrency to 512 produced little additional throughput, while large-request p95 rose to approximately 2.21 seconds. Admission limits remain necessary; accepting more outstanding requests is not the same as increasing capacity.

Appendix A discusses OVMS’s remaining advantage and the options for further improvement.

### 5.d Delivery plan

The readiness changes have limited interaction with the performance redesign and can land independently, before or after it. Integration testing should confirm that startup completion and model-readiness checks remain authoritative when accepting traffic.

Adaptive batching should land before the performance work. That change moves batching onto the main-process model before worker dispatch, while the raw-JSON prototype dispatches directly to workers and would bypass the batching wrapper. Landing batching first establishes the behaviour the redesign must preserve.

Performance development and benchmarking can continue in parallel, but production integration should follow this sequence:

1. **Establish the correctness baseline.** Land and validate the adaptive-batching fixes first. Verify cancellation, streaming fallback, and safe teardown of active predictions.

2. **Introduce worker-side JSON as an opt-in REST path.** Initially support non-streaming requests to parallel models with adaptive batching disabled. Keep the existing path for batched models, streaming, non-parallel models, and unsupported custom handling. Enforce body-size and outstanding-request limits.

3. **Introduce the production ZeroMQ transport separately.** Preserve response correlation and lifecycle controls. Test worker failure and restart, cancellation, concurrent model load/unload, and shutdown with inference in flight. The benchmark exposed shutdown callback errors that must be resolved before rollout.

4. **Qualify and roll out gradually.** Verify error responses, headers, middleware, metrics aggregation, multi-model routing, and all supported runtimes. Test gRPC for regressions in shared worker infrastructure, without claiming a gRPC performance improvement. Compare canary deployments against the existing path and retain a configuration rollback.

Python migration should remain a separate qualification step. The immediate design benefit comes from distributing work, not from requiring free-threaded Python.

## 6. Conclusion

The investigation showed that much of MLServer’s REST queueing was a symptom of concentrated frontend work. JSON parsing, validation, serialization, and response formatting competed for one process’s CPU capacity. Adding inference workers could not remove that bottleneck while the frontend remained responsible for preparing and returning their work.

The selected prototype distributed JSON processing across inference workers and refined the transport around that change. Within the same resource budget, it delivered approximately 2.1–3.1× the original throughput and lower p95 response times at 64 concurrent requests. Those improvements retained FastAPI and GIL-enabled Python 3.12; free threading was not required.

These measurements cover CPU-based ONNX inference over REST. The architectural idea extends to the shared serving path used by other runtimes, but their benefits require separate validation and tuning. The gRPC path was not benchmarked.

The result is a demonstrated design direction, not a production-ready replacement. Delivery should establish the adaptive-batching correctness baseline first, then introduce worker-side JSON and ZeroMQ separately. Compatibility, lifecycle handling, observability, and bounded admission remain part of that work.

Higher aggregate capacity can then come from additional serving pods, each with its own frontend and workers. That requires additional resources and does not erase the remaining per-instance performance gap. Appendix A discusses that gap and the more fundamental changes that could address it.

## Appendix A. The remaining OVMS gap

### A.1 Why OVMS remains faster

At 64 concurrent requests, tuned OVMS delivered approximately 14.8×, 1.7×, and 1.36× the selected MLServer configuration’s throughput for 1-, 100-, and 1,000-row requests.

Both servers received REST/JSON requests. OVMS therefore still had to parse decimal numbers and format responses; its advantage was not obtained by switching to gRPC.

The serving architectures differ substantially. OVMS implements its REST interface in C++ with multithreaded request handling. Its native model-serving path does not require MLServer’s Python/Pydantic object processing or frontend-to-worker pickle boundary. OpenVINO also provides execution streams for concurrent inference within the native runtime. [OVMS performance tuning](https://github.com/openvinotoolkit/model_server/blob/main/docs/performance_tuning.md).

The redesigned MLServer distributes conversion work more effectively, but still constructs Python objects, copies messages between processes, and coordinates requests through one Python frontend. Those costs matter particularly when predictions are fast.

The much larger gap for small requests is consistent with serving overhead dominating their cost. However, the comparison also changes the inference engine—from ONNX Runtime to OpenVINO. It does not establish how much of the remaining difference comes from the frontend, data conversion, IPC, or native inference.

### A.2 The practical capacity step: more pods

After improving one instance, the straightforward way to increase aggregate capacity is to distribute requests across additional MLServer pods. Each pod has its own frontend and inference workers, so frontend capacity grows alongside backend capacity.

That requires additional CPU and memory. Replicas must also expose the appropriate models and versions, and readiness must prevent traffic reaching instances that cannot serve them. Scaling is not guaranteed to be linear.

This is a deployment strategy, not a way to close the resource-matched benchmark gap. OVMS can also scale horizontally. Adding MLServer replicas increases total capacity but does not make each instance as efficient as OVMS.

### A.3 More fundamental changes

The ZeroMQ experiment in this study was inspired by vLLM’s separation of its frontend and inference engine. vLLM is now exploring an experimental Rust frontend while retaining Python engine processes and the existing ZeroMQ/MessagePack boundary. [vLLM Rust frontend architecture](https://github.com/vllm-project/vllm/blob/main/rust/README.md).

Its stated motivation includes Python’s GIL restrictions, garbage-collection overhead, and frontend event loops becoming bottlenecks. That is the same class of architectural problem encountered here, although the workloads differ. It provides a useful precedent, not a prediction of MLServer’s potential speedup. [vLLM Rust frontend RFC](https://github.com/vllm-project/vllm/issues/40846).

Further single-instance improvements could take three directions:

- **Replace more of the serving path with native code.** A C, C++, or Rust frontend could process requests across native threads while retaining Python model runtimes. The important change would extend beyond HTTP handling to parsing, validation, response formatting, and a language-independent worker protocol. Preserving model selection, custom handlers, errors, streaming, metrics, and lifecycle behaviour makes this a substantial engineering project.

- **Keep numeric data in binary buffers.** A binary tensor path could avoid decimal conversion and repeated construction of Python numeric lists. Existing gRPC already avoids decimal conversion, but MLServer still converts its tensor contents into Python objects. A more complete design would preserve compact buffers through dispatch and runtime codecs. Changing the wire format alone is not zero-copy serving.

- **Use shared memory for large payloads.** Workers could receive buffer descriptors instead of copied tensor contents. This requires explicit ownership and cleanup across cancellation, unload, and worker failure. It would reduce copying, but would not remove JSON processing unless paired with a different conversion path.

These options should be evaluated as a second design phase. The current experiments establish a useful improvement without replacing the frontend or tensor representation. Going further means removing more work—not merely moving the same work to another queue.

### A.4 Native inference without Python

A native frontend could be taken further by calling the inference libraries directly, without Python in the supported prediction path. This is feasible for ONNX Runtime, XGBoost, and LightGBM because they already expose native APIs. Scikit-learn is the important exception.

| Backend | Native serving option |
|---|---|
| ONNX Runtime | Load the same ONNX model and run inference through its [C/C++ API](https://onnxruntime.ai/docs/api/c/). |
| XGBoost | Load native model files and predict through its [C API](https://xgboost.readthedocs.io/en/stable/c.html). |
| LightGBM | Load native model files and pass numeric buffers through its [C API](https://lightgbm.readthedocs.io/en/stable/C-API.html). |
| Scikit-learn | Export supported models and pipelines to ONNX. Arbitrary joblib/pickle models still require Python; conversion coverage is incomplete. [Model persistence](https://scikit-learn.org/stable/model_persistence.html). |

Much of the expensive inference arithmetic is already compiled. The opportunity is to remove Python from the surrounding validation, object conversion, and dispatch—not rewrite the numerical kernels. A native server could select loaded model handles by name/version and execute predictions through a thread pool using numeric buffers. Keeping that path in one process could eliminate frontend-to-worker pickling, although REST/JSON would still require decimal conversion.

This would be a serving-layer reimplementation, not simply packaging MLServer into an executable. It must preserve preprocessing, prediction and output conventions, thread safety, and safe model unload. A hybrid design could retain Python workers for unsupported scikit-learn pipelines and custom runtimes.

A focused future experiment would compare native serving against MLServer while holding the ONNX Runtime build, model, JSON requests, and resource budget constant. Unlike the OVMS comparison, that would isolate serving-layer differences without also changing the inference engine. No performance gain from this design has been measured here.
