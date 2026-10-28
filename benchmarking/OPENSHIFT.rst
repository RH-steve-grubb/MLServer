Serving measurements on OpenShift
=================================

Use the `study tools <STUDY_GUIDE.rst>`_ from a dedicated client pod or test
machine, and read the `profiling guide <PROFILING.rst>`_ before requesting
node access. The historical RF-DETR study used direct localhost traffic on
one RTX 4070; its absolute rates are not OpenShift capacity predictions.
Cluster measurements need their own frozen control and environment record.

Inventory before deployment
---------------------------

Use an existing authorized test namespace. Do not create cluster-wide roles,
change SCCs, modify operators, or alter production predictors to obtain
benchmark access. Record the context and namespace without exporting tokens
or kubeconfig contents. Example read-only commands::

    oc version
    oc whoami --show-server
    oc project -q
    oc get inferenceservices.serving.kserve.io -n "$test_namespace"
    oc get servingruntimes.serving.kserve.io -n "$test_namespace"
    oc get pods -n "$test_namespace" -o wide
    oc explain inferenceservice.spec.predictor
    oc explain servingruntime.spec.containers
    oc auth can-i get pods -n "$test_namespace"
    oc auth can-i create pods/exec -n "$test_namespace"

Use the installed CRDs and current operator documentation. Do not assume a
KServe version, RawDeployment/Standard/Serverless mode, service mesh, ingress
implementation, auth mechanism or gRPC support from another environment.
Record OpenShift/KServe/operator versions, runtime CR YAML, predictor image
ID/digest, deployment mode, route and sidecars. Sanitize inline secrets before
sharing configuration dumps. Server and client source/package inventories
must identify their actual image or editable source paths.

Deployment and image controls
-----------------------------

Use the organization's existing MLServer ServingRuntime/InferenceService
and approved image build/model-storage procedure. Resolve all placeholders
before applying anything. The model artifact must be the same hash in both
predictors; an image digest alone does not establish model equality.
Freeze one replica, resource requests/limits, node/GPU class, runtime command,
settings, sidecars, route/TLS, autoscaling configuration and client placement.
Check the installed CRD's fields for minimum/maximum replicas; confirm the
observed pod count and restarts throughout each trial. One replica still
contains multiple MLServer model workers.

GPU allocation normally requires the GPU operator/device plugin and a
schedulable resource such as ``nvidia.com/gpu``. Read the site's actual
resource names and sharing policy. A request for one GPU does not prove
exclusive hardware: record MIG, time-slicing/MPS, device UUID and other
compute users. Compare the same GPU class/partition and isolation policy.
A Guaranteed pod QoS class does not by itself prove exclusive CPU placement;
record CPU-manager policy where available, effective cpusets, quota and
throttling. Explicitly include proxy/sidecar CPU costs when reporting total
service cost; distinguish predictor-only from pod-total counters.

Use a PVC or approved object storage for model/configuration and artifacts.
OpenShift arbitrary UIDs need read access to the model and writable runtime
IPC/temp directories. Do not solve permission problems by switching the
predictor to privileged/root. Record the selected repository path and CUDA
library resolution in each worker. A container that advertises a CUDA provider
but silently runs on CPU is not a qualified GPU predictor.

Build the baseline and candidate with the same approved recipe/dependency
locks, changing only intended source/options. Use immutable digests and
source labels; do not compare a stale pull-cached tag with a freshly built tag.
If production build/CI/Dockerfile changes require approval, leave those paths
unchanged and use an already authorized experiment-image workflow. A local
wheel or startup-mounted source tree can aid diagnosis, but do not describe
it as a production image measurement without equivalent packaging evidence.

A rollout hook for the sweep must update only the owned test workload, wait
for rollout and model readiness, record the new pod UID/image ID, and perform
a correctness check. For a plain, explicitly owned Deployment, the basic
operations are::

    oc -n "$test_namespace" set image deployment/"$test_deployment" \
      "$predictor_container=$selected_image_digest"
    oc -n "$test_namespace" rollout status deployment/"$test_deployment" \
      --timeout=600s

Do not use that example to patch an operator-managed KServe Deployment:
change its owning CR/runtime through the approved procedure or the controller
may revert it. The sweep's prepare hook must restart even if the selected
image digest is unchanged; a no-op rollout is not a fresh process. Save
readiness/correctness/config/resource receipts in BENCHMARK_TRIAL_DIR.
If no deployment authorization is available, benchmark an existing qualified
endpoint or prepare the artifacts without claiming deployment completion.

Clients and network paths
-------------------------

Run the load generators on separate reserved CPU resources. Install the
resources environment in an approved client image or use an approved Python
3.12 environment with the same client packages. Copy model-ready payloads
onto persistent storage visible to the client pod. The client does not need
a GPU. Keep authentication/TLS verification and connection reuse identical.
Run the entire sweep coordinator and its children in one client pod; it uses
one host monotonic clock and a shared filesystem start gate. For multiple
client pods, use a separately qualified distributed load generator and report
its coordination/accounting policy.

Distinguish these paths in every filename, table and claim:

* Direct predictor service/pod endpoint from a client pod. This measures
  cluster networking and serving; record whether sidecars are on the path.
* The actual KServe/OpenShift route with normal TLS/auth/proxies. This is the
  deployment path. Save the resolved endpoint and configured timeouts/body
  limits without saving credentials.
* ``oc port-forward`` or an SSH tunnel. Useful for correctness/debugging, but
  an extra process/tunnel bottleneck makes it unsuitable for a capacity claim.

Do not compare a loopback baseline to a routed candidate. Keep client node,
network path, DNS/service selection and resource allocation stable. Do not
silently bypass a router because a 15 MB JSON body fails. Save the status and
request size, identify the failing hop, and report that path as unsupported
until an authorized configuration change is separately qualified.

Example endpoint specifications for the study plan::

    {
      "protocol": "rest",
      "url": "https://authorized-route/v2/models/rfdetr-small/infer",
      "payloads": ["/durable/payloads/j0-0.bin"],
      "bearer_token_env": "BENCHMARK_BEARER_TOKEN",
      "ca_file": "/mounted/site-ca.pem"
    }

Omit token/CA entries when unnecessary. Populate the token through the site's
secret delivery mechanism; never embed it in a saved plan or shell history.
Do not disable TLS verification. A client TLS failure is not model latency.
Raw gRPC can still be qualified against a direct predictor where routing lacks
support. Mark routed gRPC unavailable instead of silently substituting REST.

Binary tensor HTTP is an independent capability check. The standard appends
little-endian tensor bytes to a JSON header and identifies the header's byte
length with Inference-Header-Content-Length. Test the server and every route
hop with known tensor/output parity, mixed metadata, exact lengths and content
types. HTTP support alone does not guarantee this extension is passed through.
Keep JSON-output and binary-output comparisons separate. Do not infer feature
support from the presence of a v2 route or from a local-only success.
See the `KServe binary tensor specification
<https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol/binary-tensor-data-extension>`_.

Server-side accounting and profiling
------------------------------------

``oc adm top`` and dashboard averages are useful context but often too coarse
for a short trial's CPU/request. Prefer read-only cgroup v2 CPU/memory counters
inside the predictor's cgroup, including descendants, sampled around the
whole admission/drain interval. Record timestamp, units, counter resets, pod
UID, collection duration and the accounting envelope. If collecting through
``oc exec``, the exec process may contribute CPU; characterize that overhead
or use a persistent sampler. Include sidecars separately if outside the
predictor cgroup. Prometheus/DCGM data must retain scrape intervals and query
ranges; do not present a five-minute average as a sixty-second request cost.
Container restarts, throttling, OOMs or changing replicas invalidate a paired
trial until explained. Save pod events, restarts and relevant logs.

Root in a normal pod is not host profiling permission. Perf access depends
on host policy, capabilities, seccomp/SELinux and PID visibility. Use the
platform team's approved diagnostic pod/node workflow. If node debugging is
explicitly authorized, ``oc debug node/<node>`` plus ``chroot /host`` is a
common entry to host inspection; it creates a privileged diagnostic pod and
must not be run as a routine benchmark requirement. Obtain the target
container's host PIDs through the runtime/node tools, map them to the
predictor container/pod UID and NSpid values, and attach only to those PIDs.
Do not sample the entire production node to avoid doing PID mapping.

For CUDA/NVTX, run Nsight in an approved diagnostic predictor image/process
launch so injection reaches model workers. The image must provide Nsight and
NVTX headers/libraries appropriate to the driver. Perf CPU/GIL and CUPTI CUDA
tracing have different permission requirements. Do not assume that one
working tool certifies the others. Launching Nsight in a client pod profiles
the client, not the remote GPU server. Follow the profiling guide's stage
coverage and measured-window validation. Persist reports to a PVC and copy
verified artifacts to the final artifact store before deleting the pod.

If privileged diagnostics are prohibited, retain unprofiled results,
application stages and accessible cgroup/provider telemetry. State which
causal claims remain unsupported. Missing root access need not prevent an
honest before/after transport benchmark; it does prevent asserting unmeasured
GIL wait time or CUDA kernel activity.

Cleanup and reproducibility
---------------------------

Keep source/image/model/config hashes, plan and exact commands, observed pod
identities/resources, raw samples, counters and all failures together in a
persistent artifact directory. Save timestamps in UTC and monotonic clocks
with their host identity. Export a checksummed archive to approved storage;
a pod's /tmp or emptyDir is not durable evidence. The resources branch stores
reusable tools, not cluster credentials or large captures. Delete only owned
test resources after export and record any intentionally retained resources.

References
----------

* `KServe control-plane API
  <https://kserve.github.io/website/docs/reference/crd-api>`_: verify against
  the CRDs installed in the target cluster.
* `KServe custom predictor/runtime guide
  <https://kserve.github.io/website/docs/model-serving/predictive-inference/frameworks/custom-predictor>`_:
  deployment concepts; adapt to the site's supported MLServer runtime.
* `Red Hat: debugging inside OpenShift containers
  <https://developers.redhat.com/blog/2020/01/09/debugging-applications-within-red-hat-openshift-containers>`_:
  why container privilege and kernel profiler access differ.
* `Kubernetes pod debugging
  <https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/>`_:
  PID/security context and diagnostic workload concepts.
