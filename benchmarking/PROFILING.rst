Profiling a serving instance
============================

Use the `measurement guide <STUDY_GUIDE.rst>`_ for ordinary load. Profile a
separate run and bracket it with unprofiled controls. A profiler can change
GIL scheduling, allocations and CUDA timing; its RPS is diagnostic, not the
final capacity number. Startup, warmup and graph capture are not prediction
measurements. Keep raw captures, exact commands, tool versions, warnings,
process identities and analysis outputs on persistent storage.

Prerequisites and privilege separation
--------------------------------------

On the Linux GPU host, check the actual executables::

    python3 --version
    perf --version
    nsys --version
    nsys profile --help
    nsys export --help
    nvidia-smi
    cc --version
    nm --version
    objdump --version
    lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
    cat /proc/sys/kernel/perf_event_paranoid

Use the selected Poetry interpreter, not host Python, for serving and Python
analysis. The historical study used CPython 3.12.13 with the GIL, ORT GPU
1.23.2, NumPy 2.2.6 and one RTX 4070. Its functioning Nsight executable was
``/usr/local/sbin/nsys``; a CUDA-toolkit launcher existed but its payload was
missing. Test ``--version`` successfully, not just ``which`` or an RPM record.
Do not copy the old Fedora package-repair command onto an OpenShift node.
An administrator should provide the tools appropriate to that node/image.

Root is used for perf probes and resource enforcement. Application servers
and clients remain the normal test user. Do not change perf sysctls, GPU
clocks, power limits, audit policy, driver/toolkit versions, SCCs or node
configuration merely to make a capture run. Use approved profiling access or
record the unavailable measurement. Unavailable evidence is not zero overhead.
OpenShift's namespace/PID/security differences are covered in
`the cluster guide <OPENSHIFT.rst>`_.

Resource enforcement on a standalone host
-----------------------------------------

Discover physical-core sibling groups; reserve complete groups for clients.
The original host used fourteen physical cores plus their SMT siblings for
one server and two other physical cores for clients, with 16 GiB server RAM
and no swap. That topology is historical, not portable CPU numbering.

A test administrator can launch an owned scope like this after choosing
absolute paths and actual CPU lists. The shell variables must be set; the
unit name must be unique to this experiment::

    sudo systemd-run --scope --unit="$study_unit" \
      --property="AllowedCPUs=$server_cpus" \
      --property=MemoryMax=17179869184 --property=MemorySwapMax=0 \
      taskset -c "$server_cpus" runuser -u "$study_user" -- \
      "$serving_python" "$resources_root/benchmarking/study_tools/launch_server.py" \
      --repository "$model_repository"

Supply needed library paths via the launcher and ensure the normal user can
read the code/model and write owned IPC/log locations. A root umask must not
hide fixtures from the application. The scope command remains attached; use
a dedicated terminal or an owned process supervisor. Record the scope and
all frontend/worker PIDs. Check each ``/proc/<pid>/cgroup`` and the actual
``cpu.max``, ``cpuset.cpus.effective``, ``memory.max``, ``memory.swap.max``
and ``memory.events``. CPU affinity alone does not enforce a memory budget.
Only stop the named owned scope. Do not use broad ``pkill`` commands.

CPU sampling
------------

Warm and qualify the server first. Obtain its frontend PID and every model
worker PID; a worker count in settings is not proof of running processes.
Save ``/proc/<pid>/stat`` start times to detect PID reuse, ``cmdline``,
``status``/``NSpid``, ``cgroup`` and ``maps`` for target processes. Avoid saving
unrelated processes or complete environments containing credentials.

Run the client as the normal user. In a separate authorized root shell,
attach only to the selected serving PIDs::

    sudo perf record --clockid mono -F 99 -g --call-graph dwarf \
      -p "$serving_pids_csv" -o "$capture_dir/perf.data" -- sleep 30
    sudo perf script --ns --show-lost-events \
      -i "$capture_dir/perf.data" > "$capture_dir/perf-script.txt"
    sudo perf report --stdio --no-children --sort comm,tgid,pid,dso,symbol \
      -i "$capture_dir/perf.data" > "$capture_dir/cpu-report.txt"

Choose a diagnostic load duration that fits inside capture; coordinate the
start gate and save its monotonic start/drain times. Restrict analysis with
``perf report --time START_SECONDS,END_SECONDS`` to those same-host
CLOCK_MONOTONIC boundaries. Save stderr, lost events and unwind warnings too.
If capture does not contain the full measured interval, it cannot explain the
whole trial. Do not subtract monotonic timestamps from different hosts.
Python 3.12 ``-X perf``/``PYTHONPERFSUPPORT=1`` may improve Python frame names
where supported; qualify it separately and never enable it in only one
capacity control. Missing Python symbols do not imply no Python cost.

Interpret on-CPU samples separately for frontend main thread, IPC/helper
threads and workers. Samples in memcpy/pickle/json/NumPy routines motivate a
copy/representation hypothesis; they do not directly measure bytes copied.
Use allocation/page-fault/cgroup counters to test that hypothesis. Summed
server CPU/request is different from request latency. Nested spans overlap.

Host stages and CUDA timelines
------------------------------

The historical sources below provide ``gpu_stages.py``, ``gpu_stage_summary.py``,
``gpu_nvtx.c``, ``gpu_cuda_summary.py`` and ``gpu_cuda_analyze.py``. Stage
wrappers record process/thread IDs, request correlation, monotonic wall time
and thread CPU time. The C bridge emits NVTX ranges without creating a CUDA
context. Compile that diagnostic bridge against the installed NVTX3 headers::

    cc -O2 -fPIC -shared -I "$nvtx_include_dir" \
      "$harness_dir/gpu_nvtx.c" -o "$capture_dir/gpu-nvtx.so" -ldl

This is one compilation unit; larger builds should use available parallelism.
The old include path was ``/usr/local/cuda-13.3/include``. Discover the current
NVTX header location rather than assuming that toolkit version is present.

Instrument only a diagnostic launcher. Install wrappers before the frontend
and spawned workers import their serving code. Invoke the normal CLI main()
after installation. A top-level spawn-safe installation hook or a temporary
sitecustomize module can do this; keep it out of the unprofiled environment.
Qualify every wrapped function against the installed source. The historical
hooks target prototype functions: integrated REST worker parsing now lives
in ``mlserver.rest.worker`` rather than ``raw_worker``. Raw gRPC buffer helpers
and CUDA graph wrappers also need actual-path coverage. Do not conclude a
stage is cheap because an obsolete hook recorded nothing. Record drops,
unpaired spans, request IDs and worker coverage; fail a broken correlation.

Launch Nsight around the actual server process tree, not around the load
client. Example switches, subject to the installed CLI's help::

    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --wait=all --output "$capture_dir/cuda-timeline" \
      "$serving_python" "$diagnostic_launcher" start "$model_repository"

Run the application as the ordinary user, even if root is required to arrange
its resource scope. For CUDA Graph investigations, qualify
``--cuda-graph-trace=node`` if supported. Collect only short diagnostic runs;
stop the owned server cleanly so Nsight and Python stage buffers flush. Save
``*.nsys-rep`` and export without overwriting previous captures::

    nsys export --type=sqlite --output "$capture_dir/cuda-timeline.sqlite" \
      "$capture_dir/cuda-timeline.nsys-rep"
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
      "$capture_dir/cuda-timeline.nsys-rep" > "$capture_dir/nsys-stats.txt"

Verify report names with the installed version. Generic stats include all
captured activity, including startup; they are not automatically per-request
steady-state summaries. Use the measured NVTX/stage window and correlation
IDs to identify this model's kernels, copies and memsets. Validate event
coverage before computing request costs, especially graph replays. Union
of device-active intervals is activity, not SM occupancy. A CUPTI schema or
correlation mismatch means unsupported analysis, not a zero-cost operation.

GIL probes
----------

Use GIL tracing only for a specific contention question. The historical
``gil_profile.py`` resolves take_gil/drop_gil and pthread condition-wait
symbols and file offsets, installs uniquely named perf uprobes, runs quiet
and known-contention controls, and removes only its own probes afterward.
``rfdetr/gpu_perf.py`` extends the capture/analysis to all selected processes.
Read its ``self_test`` and ``collect`` entrypoints before using them. They
expect CPython 3.12 with suitable symbols and CLOCK_MONOTONIC load boundaries.
Run the known-contention control on the exact interpreter/libc before model
capture. Symbol stripping, probe permissions, lost events or unmatched calls
invalidate the interpretation; do not report zero GIL waits as success.
The older standalone gil_profile CLI targets its own CPU benchmark, not an
arbitrary modern predictor. Use the GPU harness or adapt its collector,
keeping source and qualification artifacts.

GIL wait time is summed thread time, not a percentage of request latency.
Independent worker processes have independent GILs. Validate acquisition and
condition-wait pairing per process/thread and restrict to the measured load
window. Preserve raw events and the analysis validity flags.

Historical harness recovery and exact commands
-----------------------------------------------

`legacy-queueing-sources.tar.gz <profiling/legacy-queueing-sources.tar.gz>`_
contains the original untracked Python/C/shell harness and its Poetry
manifests/locks, including supporting CPU utilities imported by GPU tools.
`SOURCE_MANIFEST.json <profiling/SOURCE_MANIFEST.json>`_ records every source
hash. It deliberately omits duplicate reports, model weights, binary build
artifacts and large raw traces. Read the existing
`GPU report <../reports/gpu/GPU_PERFORMANCE_STUDY.md>`_ for historical results.

Extract into a disposable source checkout of historical stock commit
``11e8788393795c44edb661d03897b463457a07e6``; the archive's paths start with
``benchmarking/queueing/``. Inspect ``tar -tzf`` first, refuse overwrites, and
verify each extracted file against SOURCE_MANIFEST.json. The original
``gpu_study.py`` infers the repository from its filesystem location and copies
that runtime source into each run. Extraction into the resources checkout
alone will not supply an MLServer implementation.

Create an isolated Poetry environment using the archived transport-env
manifest/lock after resolving its relative editable dependencies to that
checkout. Do not reuse a historic environment against another checkout without
verifying parent and spawned-worker imports. Use absolute paths for all of::

    historical_python=/absolute/isolated/environment/bin/python
    harness_dir=/absolute/historical/MLServer/benchmarking/queueing/rfdetr
    bundle_dir=/absolute/durable/model-bundle
    capture_store=/absolute/durable/historical-reproduction
    study_user=normal-test-user

The following commands describe the old adapter-based method, not capacity
of integrated production code. They use explicit paths so no local defaults
are required::

    "$historical_python" "$harness_dir/gpu_study.py" --help
    "$historical_python" "$harness_dir/gpu_study.py" \
      --phase preflight --bundle "$bundle_dir" --store "$capture_store" \
      --python "$historical_python" --user "$study_user"
    sudo "$historical_python" "$harness_dir/gpu_study.py" \
      --phase tune --protocol both --workers 2 --threads 2 \
      --concurrency 16 --seconds 20 --bundle "$bundle_dir" \
      --store "$capture_store" --python "$historical_python" --user "$study_user"
    sudo "$historical_python" "$harness_dir/gpu_study.py" \
      --phase profile --capture cpu --protocol grpc --workers 2 --threads 2 \
      --concurrency 16 --seconds 10 --bundle "$bundle_dir" \
      --store "$capture_store" --python "$historical_python" --user "$study_user"

For GIL change ``--capture cpu`` to ``--capture gil``. For host stages use
``--phase stages``; for CUDA/NVTX use ``--phase cuda``. ``--phase tune`` runs
the harness's built-in grid; inspect its actual grid in source before treating
``--workers`` as a one-case selection. ``--phase smoke`` selects an individual
variant but is explicitly qualification, not resource-controlled capacity.
``gpu_baseline.py`` is the older complete screen/profile driver, and
``gpu_load.py`` is its per-process fixed-count client. ``gpu_experiment.py``,
``gpu_transport_compare.py`` and engine comparison scripts add historical
adapters. Inspect ``--help``; do not blindly run their large default plans.

The old root driver uses systemd/cgroup v2, an automatically discovered budget,
a normal-user account and an explicit RTX 4070 device-name check.
Other GPUs require a separately qualified adaptation, not bypassing the
check and labeling the result historical reproduction. It has host-specific Nsight
include paths and source/receipt expectations. It is not an OpenShift pod
launcher. Its ``SUMMARY.json`` points to per-case result/receipt/client/stage/
perf/CUDA files. The driver calls ``gpu_stage_summary.collect(case, result, server)`` with
the case result and frontend/worker receipts; it is a library, not a CLI.
The CUDA analyzer has this CLI::

    "$historical_python" "$harness_dir/gpu_cuda_analyze.py" \
      --case /absolute/case-directory --output /absolute/new-analysis.json

CUDA analysis expects the original case layout, stage files and request
correlation. A generic Nsight SQLite export alone is insufficient. Preserve
``valid``/coverage failures, raw logs, command manifests and snapshots. If
adapting this machinery to an integrated implementation, qualify its source
hooks and receipt schema first; never silently install old adapters while
claiming to profile new production code.

Technical references
--------------------

* `NVIDIA Nsight Systems User Guide
  <https://docs.nvidia.com/nsight-systems/UserGuide/>`_: installed-version CLI,
  process-tree CUDA/NVTX capture and export.
* `Nsight post-collection analysis
  <https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html>`_: report
  definitions and interpretation.
