# Lifecycle ownership

Lifecycle operations remain serialized by model name and by inference pool.
Acquiring the public operation lock accepts a mutation; cancellation while waiting
for that lock has no effect on models. After acceptance, the existing deferred
cancellation boundary returns the operation result or exception despite caller
cancellation. The configured deadline requests cancellation; it is not evidence
that native code, a thread, or a worker process stopped.

| Transition | Publication and ownership |
| --- | --- |
| Fresh load starts | The registry may expose a not-ready loading entry. |
| Hooks return | Ownership transfers to the returned model, including wrappers. A hook which raises must release resources it has not transferred. |
| Preparation completes | The registry marks the model ready. A replacement remains staged until the old generation is removed. |
| Replacement commits | The new ready instance is registered before old-instance cleanup. Cleanup failure preserves this committed instance. |
| Load fails before commit | Only the failed instance is removed. The previous instance and settings remain committed. |
| Runtime preparation exceeds the deadline | The exact preparation task stays owned; the instance is removed from serving, and cleanup waits for preparation to settle. Later mutations join this recovery before changing that name. |
| Unload starts | Serving eligibility is removed before cleanup. A retained cleanup task owns the exact instance independently of its waiter. |
| Cleanup fails | The error and instance remain owned. Loads report that failure; explicit unload retries cleanup. A load failure keeps its primary cause, and exposes `ModelLoadError.cleanup_error` when synchronous cleanup also fails. |

Local runtime preparation runs in an owned asyncio task. A blocked event loop
cannot deliver deadlines; a native call must return before Python can act. If an
owned preparation never settles, cleanup and subsequent mutations of that model
name cannot complete. The registry keeps that ownership and does not declare its
resources released. This does not introduce a server termination policy.

Extension hooks must not leave unowned detached native work after returning or
raising. Wrappers must own the resources they replace. Arbitrary detached tasks
created by extensions are not covered by the nested timeout contract.
