Reproducible serving measurements
=================================

This guide complements the existing `quick benchmark <README.md>`_ and
`model preparation <../model-preparation/README.md>`_. Those remain the
starting points for installing the Poetry environments, obtaining the public
RF-DETR assets, exporting ONNX, and checking a first prediction. The scripts
here add timed load, multiple client processes, raw latency samples, a sweep
plan, and paired analysis. They work against an already provisioned endpoint;
they do not deploy a cluster or install profiling privileges.

Tools and responsibilities
--------------------------

* ``benchmarking/inference.py``: existing quick fixed-count check. It saves
  aggregate latency/throughput, but not per-request timings or server CPU.
* ``benchmarking/study_tools/preflight.py``: read-only inventory and bundle
  hash check. Run on the client and inside the server environment when possible.
  CUDA provider availability is not proof that a worker executed on the GPU.
* ``benchmarking/study_tools/launch_server.py``: production CLI launcher, using the
  existing model-preparation CUDA library-path helper for spawned workers.
* ``benchmarking/study_tools/payloads.py``: prebuild original JSON (J0), shorter
  FP32 JSON (J1), and raw protobuf (G), plus eight deterministic finite inputs.
* ``benchmarking/study_tools/client.py``: one timed load-generator process.
  It consumes complete responses, records every attempted request, and drains
  in-flight work after an error. It does not check numerical output values.
* ``benchmarking/study_tools/sweep.py``: executes an explicit restart/readiness
  command before each trial, warms clients, coordinates their start, and saves
  results. Multiple clients run on the same host or in the same client pod;
  this is not a distributed clock-synchronization protocol.
* ``benchmarking/study_tools/analyze.py``: candidate/control ratios with
  paired-trial bootstrap intervals. It reports missing/invalid pairs, and
  deliberately does not decide whether an optimization is acceptable.

Run these from the resources checkout root with its Poetry environment. No
package version changes or extra client dependencies are needed. Run
``make lint`` and ``make test`` after changing the tools. Local loopback tests
validate accounting, REST body consumption, gRPC byte transport and the
multi-process start gate; they do not establish GPU or OpenShift performance.

Portable paths and preflight
----------------------------

Use sibling checkouts as described in the `root guide <../README.md>`_. All
paths below are examples; resolve them before creating a plan::

    resources_root="$PWD"
    study_store="$HOME/mlserver-durable-store/serving-study"
    bundle_dir="$HOME/mlserver-durable-store/rfdetr-small-bundle"
    mkdir -p "$study_store"
    poetry run python benchmarking/study_tools/preflight.py \
      --bundle "$bundle_dir" --output "$study_store/preflight.json"

The output file must not already exist. Verify Python 3.12, the installed
packages, actual editable MLServer/ONNX source paths, the model hash, NVIDIA
access, and the selected CPU topology. On a remote client, missing NVIDIA or
perf is expected: inventory the predictor separately. Treat missing required
client imports or a model hash mismatch as a setup failure. ``preflight.py``
does not install packages, download models, test credentials, or certify an
endpoint. Check Poetry/git/tool exit statuses in its report; successful
inventory creation does not mean every optional executable exists.

The original fixture/model hashes and export requirements are already in
``model-preparation/REFERENCE_MANIFEST.json`` and its guide. A new export can
have different ONNX bytes. Use one immutable bundle for both sides of a new
comparison and report its hash; do not mix a new export with historical rates.
For air-gapped clusters, prepare/export externally, transfer the complete
bundle with checksums, and use an approved offline package/image mirror.

Production launch versus historical launch
------------------------------------------

The existing ``model-preparation/serve.py`` launches ``root()`` directly. It is
useful for its existing smoke checks, but bypasses the normal CLI's uvloop
installation. The production console entrypoint calls ``main()``. For
production-path measurements use::

    poetry run python benchmarking/study_tools/launch_server.py \
      --repository "$bundle_dir/repositories/cuda"

Or launch ``poetry run mlserver start <repository>`` with the same CUDA library
paths available to frontend and spawned workers. Never infer the active loop
or HTTP parser merely from installed packages. Record the running loop class
and Uvicorn's resolved HTTP protocol in a short diagnostic pass, including
worker loop/provider settings, then remove diagnostics from timed trials.
Do not use experimental adapter launchers to represent integrated code.

Copy a model repository into an experiment-owned directory before changing
settings. The original fixture settings are smoke-test defaults, not a tuned
profile. Configure the supported public options in
`MLServer's performance guide
<https://github.com/RH-steve-grubb/MLServer/blob/winning-combination/docs/user-guide/performance.md>`_.
The source revision and settings used in a measurement must be frozen together.
The following is an illustrative server configuration, not a recommendation
for every GPU or cluster::

    {
      "host": "0.0.0.0", "http_port": 8080, "grpc_port": 8081,
      "parallel_workers": 2,
      "grpc_raw_input_buffers": true, "rest_worker_json": true,
      "parallel_inference_transport": "zmq",
      "grpc_max_message_length": 67108864,
      "access_log": false, "gzip_enabled": false,
      "metrics_endpoint": null
    }

Keep logging, metrics, compression, proxies and tracing identical across
comparisons. If an organization's deployment requires metrics/logging,
retain them on both sides and describe the difference from historical tests.
Under model ``parameters.extra``, ORT settings use this structure::

    {
      "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
      "provider_options": [{"device_id": 0, "use_tf32": "1"}, {}],
      "session_options": {
        "intra_op_num_threads": 2, "inter_op_num_threads": 1
      },
      "session_config_entries": {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0"
      },
      "cuda_graph": false, "cuda_blocking_sync": false
    }

These keys are inside ``parameters.extra`` alongside a correct model URI;
retain the implementation/name fields and disable adaptive batching with
``max_batch_size=0`` and ``max_batch_time=0``. The graph and blocking-wait
options require the corresponding implementation. Check actual settings after
load; merely writing a JSON key is not proof that it took effect.

Preparing wire payloads
-----------------------

Use a new destination::

    poetry run python benchmarking/study_tools/payloads.py \
      --bundle "$bundle_dir" --output "$study_store/payloads"

``j0-0.bin`` is byte-identical to the fixture's request.json. ``j1-0.bin``
serializes the same FP32 numbers directly using orjson's NumPy support.
``g-0.bin`` is a serialized raw ModelInferRequest. All request both ``dets``
and ``labels``. The tool checks FP32 input bit parity; its manifest records
wire/input hashes and sizes. J0/J1 names describe client representation,
not a different model or precision. Do not round to a fixed decimal count.

Inputs 1--7 are the horizontal mirror and six horizontal cyclic shifts of
the original model-ready tensor. They provide distinct finite buffers for
stale-input checks, not an accuracy evaluation set. The tool saves arrays and
all three wire encodings. Timed corpus mode cycles these prebuilt bodies;
IDs are stable per body and repeat across cycles. Separate correctness tests
must use unique request IDs and original/mirror/original ordering, overlap,
and per-worker correlation to detect misrouted or stale results.

The existing ``model-preparation/validate.py`` checks serving with tolerances
and detection agreement. It is a smoke test, not an exact raw-output oracle
for a representation-only change. Capture repeated raw baseline outputs from
the actual GPU/engine and compare complete arrays, names, shapes, dtypes and
IDs. Recheck every corpus member, not only the original image. Keep numerical
validation outside timed load; the timed driver reads bytes without parsing.

Sweep plans
-----------

Copy `the example plan <templates/sweep.example.json>`_ to durable storage.
Replace every ``REPLACE_...`` value and use absolute paths. The plan is not
executable until real endpoints and restart hooks are supplied. Invoke::

    poetry run python benchmarking/study_tools/sweep.py \
      --plan "$study_store/plan.json" --output "$study_store/session-a"

Each ``prepare_argv`` is an argv array, not shell source. Use an explicitly
reviewed script if multiple commands are necessary. The driver passes
``BENCHMARK_VARIANT`` and a unique ``BENCHMARK_TRIAL_DIR`` to the hook. Before
returning success, the hook must:

1. Stop/restart only the experiment-owned server, or roll out the selected
   immutable image/settings in the authorized test namespace. Start a fresh
   instance even when repeating the same variant. Do not change a shared
   editable checkout underneath a running process.
2. Wait for rollout completion, expected replica count, and model readiness;
   prove that the old instance is gone. Save image/source/config digests,
   worker/provider/resource receipts and pod UID or PID/start time in the
   trial directory. Record declared resources and their observed limits.
3. Run a short complete-output correctness check with both outputs, and
   verify that graphs/lazy initialization are ready on every worker.
4. Exit nonzero on any mismatch. Do not return success on readiness alone.

Implement this small environment-specific hook once, version it, and qualify
it with a short trial. Do not put credentials in the plan, hook arguments or
saved receipts. Use service-account mounts or ``bearer_token_env``. ``headers``
is for nonsecret protocol/routing headers. TLS verification remains enabled;
``ca_file`` supplies a private CA. REST ``url`` is the full inference URL;
gRPC ``target`` is host:port and ``tls`` selects a secure channel.

``payloads`` lists prebuilt wire files. Use only index 0 for an invariant-body
measurement, or indices 0--7 for corpus cycling. Select ``protocol=grpc`` and
``g-*.bin`` for raw gRPC. The driver also sends arbitrary prepared binary
HTTP bodies if supplied with the matching Content-Type and
Inference-Header-Content-Length headers. It neither creates binary V2 payloads
nor certifies server support for that extension.

``clients`` is the maximum number of processes; C1 activates one. Total
``concurrency`` is divided across active processes, not multiplied by them.
For a four-client control use the same total client CPU allocation. All child
processes inherit the coordinator's affinity/cgroup. Run the coordinator on
reserved client CPUs (for example ``taskset -c <client-CPU-list> ...``), or in
a dedicated, resource-limited client pod. Do not pin it to server CPUs.

Each client completes both a minimum warmup count and a minimum warm duration.
This does not prove round-robin worker coverage: establish that in separate
diagnostics. Measured admission lasts ``seconds``; admitted requests then
fully drain. RPS uses successful responses divided by actual synchronized
start-to-last-drain time, not the requested interval. Start lateness is saved
and bounded by the plan. Errors stop new admissions in the affected client,
drain its existing requests, and invalidate the trial. Other clients finish;
the aggregate remains invalid. Failures and partial files remain on disk.
A failed hook also produces an invalid entry in SUMMARY.json.

Use a separate ``session`` name for each measurement session, nonoverlapping
``round_offset`` values, and a fresh output directory. Alternating order is
based on round number. Keep the predeclared trial count and every failed
trial. Copy the plan and selected source before results exist. No automatic
retries, winner selection, retuning, server teardown or cluster cleanup occur.
The experiment owner must stop its last server after the sweep.

Accounting and interpretation
-----------------------------

Per-case files include the prepare log, client specs, client warmup records,
start timestamp, client result files, raw ``samples.jsonl``, and aggregate
``result.json``. SUMMARY.json is updated after every case. Success means all
responses were consumed without transport errors, not model correctness.
The load is closed-loop; do not describe its p95 as an open-loop production
SLO or overload result. No token, request body or response body is copied
into sample logs. Protect the payload files themselves as appropriate.

For a same-host cgroup v2 predictor, ``cgroup`` can name the complete serving
cgroup. A hook can instead write the fresh cgroup path into ``cgroup_file``.
The coordinator captures CPU/memory counters just before the start gate and
after all clients finish. ``cpu_ms_per_request_envelope`` includes gate and
result-collection idle time; it is not an exactly aligned profiler span.
Inspect the snapshots and subtract only the recorded CPU counters. If the
server is remote, leave these fields absent and collect server-side counters
separately; null is missing evidence, not zero CPU cost. Do not use client CPU
as server CPU. Timestamped cgroup samples around warmup, admission and drain
provide a tighter accounting interval when needed. Memory.peak may include
startup and must not be described as steady-state memory.

For paired analysis::

    poetry run python benchmarking/study_tools/analyze.py \
      --summary "$study_store/session-a/SUMMARY.json" \
      --summary "$study_store/session-b/SUMMARY.json" \
      --control before --candidate after \
      --output "$study_store/paired.json"

The output includes individual candidate/control ratios and bootstrap 95%
intervals for RPS, trial p95/p99 and available CPU envelopes. Resampling units
are whole paired trials. Duplicate identities fail; missing or invalid pairs
are listed. The result does not waive missing trials or count partial evidence
as a pass. Small sample counts can yield misleadingly narrow intervals; inspect
session effects and all raw pairs. Do not average p95 values and call that a
pooled p95: concatenate actual sample arrays for a pooled percentile.

Worker/thread tuning
--------------------

W is the number of independent model workers/sessions. T is each session's
ORT intra-op thread setting; inter-op threads and spinning policy are separate.
W*T is not the total server thread count and is not a GPU concurrency limit.
The frontend, runtime libraries, IPC and client processes also need CPU.
More workers duplicate GPU memory, graph buffers and CPU arenas.

Screen W1/W2/W4/W8 and T1/T2/T4 under one fixed CPU/RAM/GPU budget. Keep the
wire encoding, engine, model, precision and client placement fixed; restart
and qualify every cell. Avoid OOM by measuring one-worker memory first and
rejecting infeasible combinations, rather than provoking cluster eviction.
Repeat cells in rotated order, including concurrency 1 and a higher load.
Select a small shortlist, then independently confirm it. Historical REST
W8/T2 and gRPC W2/T2 choices are starting hypotheses for RF-DETR on an RTX
4070; they are not defaults for another GPU or cluster CPU quota.

For every promising profile sweep C1/C2/C4/C8/C16. A useful descriptive knee
is the smallest C achieving 95% of the best stable RPS observed. If RPS is
still rising at C16, the sweep has not demonstrated capacity; label that
limit or extend the range before a capacity claim. Evaluate tail latency and
CPU/request alongside RPS. First compare code at identical W/T settings;
report independently retuned before/after configurations separately.

Additional guides
-----------------

* `Root and CUDA profiling <PROFILING.rst>`_: tools, captures, analysis and
  historical source recovery.
* `OpenShift measurements <OPENSHIFT.rst>`_: predictor versus route tests,
  resources, permissions, images, persistence and remote accounting.
