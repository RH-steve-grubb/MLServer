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

Unload-all first joins unfinished preparation and cleanup. It retries every
settled failed owner and attempts every committed version even if a retry fails.
Failed owners remain tracked; the first retained failure is reported after the
independent attempts finish. A deadline while waiting for uncertain work does not
start those independent mutations. Once instances are selected for bulk removal,
each receives a cleanup owner before the next await, so interruption cannot drop
an instance whose unload coroutine has not started.

Extension hooks must not leave unowned detached native work after returning or
raising. Wrappers must own the resources they replace. Arbitrary detached tasks
created by extensions are not covered by the nested timeout contract.

Custom REST routes bind directly to instances. Load abandonment revokes those
routes immediately, independently of cleanup waiting for native preparation.
Removal matches the instance's bound handlers, preserving committed routes for
other instances. Partial route-registration failure also revokes routes it added.

## Worker recovery

A pool retains its dispatched update waiter after frontend timeout. It immediately
removes all workers in that pool from routing. New pool mutations and replacement
worker replay join this recovery. Committed replay settings are copied; a fresh
load is added only after worker preparation succeeds, and failed replacements
never enter pending replay. The recovery owner waits for the delivered operation
to settle, then sends an unload (or replacement rollback) and requires its
acknowledgement before restoring surviving workers to routing. Per-message UUIDs
keep late responses separate from later operations. Retired frontend instances
cannot issue a second unload against a newer committed instance.

Worker unload updates target an exact version, including an unversioned instance.
If that attempted instance is absent, cleanup is a no-op; it cannot select a
healthy versioned default. Public versionless unload still selects the default.

A failed rollback or unacknowledged unload leaves the pool quarantined, with its
failure retained and reported by subsequent mutations. Close remains available.
An already dispatched operation which never settles remains owned and quarantined
until close confirms process exit. Other models in the same pool lose availability
during this conservative recovery; their committed configurations are retained.
Other pools remain independent. This adds no work to successful predictions.

Automatic pool retirement also counts staged frontend preparation and pending
replacements. Removing the last committed model cannot close
a pool still needed by an accepted load. Publication, or acknowledged cleanup of
an abandoned load, transfers/releases that owner before automatic retirement is
checked again. An otherwise empty abandoned pool may retire through its retained
close operation, which owns recovery through settlement or confirmed worker exit.
Explicit registry close remains terminal for all preparation.

## Close

Both pool and pool-registry close set closure requested before waiting for locks.
They create one retained cleanup task. Concurrent or repeated callers join it;
cancelling a caller stops only that caller's wait. Completed errors are reported
again. Pool close attempts independent cleanup even after a failure. A failed
pool is retained separately from pools eligible for new loads.

Lock draining uses the existing model-operation timeout. Worker shutdown uses
`parallel_workers_timeout` for graceful stop, then kill and bounded join. A worker
reference is retained if exit cannot be confirmed. These are cooperative Python
deadlines, not hard interruption of blocking native code or extension hooks.
Normal server shutdown and startup-error cleanup await registry close. Explicit
cancellation of the close waiter was the interrupted-close reproduction; that
probe did not establish a failure of normal, uninterrupted server shutdown.

Worker restart and recovery tasks use an explicit independent timeout context.
Awaited nested operations retain the existing shared timeout. Extension authors
spawning detached lifecycle work must give it an independent boundary and retain
its owner; inheriting a nested context does not create independent ownership.

Restart obligations wait for staged publication and recovery outside the pool
lock and initialization deadline. Actual spawn/replay retains its configured
deadline and is attempted once; failure is logged without an automatic retry
loop. Close cancels pending dependency/lock waiters and joins accepted startup,
retaining any unresolved worker owner instead of allowing late publication.
After startup settles, a final bounded sweep collects children which registered
after the first sweep. Close preserves earlier cleanup failures even if this
additional sweep succeeds.

A replacement PID is owned immediately after spawn, before dispatcher registration
can wait. This startup inventory is separate from routing and lifecycle broadcasts.
SIGCHLD records confirmed exit there even before registration; close includes those
children in its cleanup inventory. Retirement completion is recorded only after
parent endpoints and feeder threads are released, and failed retirement can retry.
An exit code alone never proves retirement. Empty replay also checks for exit before
readiness and uses the normal worker-stop path if polling observes death first.
Forced registration waits exercise defensive hardening; current serialized callers
do not normally prolong that interval. Initialization still has one bounded attempt.

If a killed worker leaves the response reader blocked, dispatcher shutdown reports
that reader as unrecoverable after a bounded thread join and retains the owner.
Python cannot forcibly terminate such a thread. This change does not add a policy
for killing the server process; an operator must handle an unrecoverable shutdown.

Environment extraction uses a retained task because its thread cannot be cancelled.
Retries join that task; they cannot treat a partial directory as a usable environment.
An incomplete marker also prevents reuse after the owner process exits. Failed
extractions retain their error and require removal of the incomplete directory
before retrying in a new registry. Close bounds its wait for extraction and reports
any unsettled owner without cancelling the underlying thread.

Successful pool close explicitly releases its extracted directory after resource
cleanup, before the pool key can be reused. Release revokes deletion authority on
retained old environment objects; collection cannot delete a later extraction.
Failed close retains that authority and blocks reuse. Borrowed directories are
never deleted, and worker copies cannot release their parent's directory.

Cancelling an owned local cleanup task also preserves its preparation reference.
A retry joins preparation before releasing resources. If preparation itself is
cancelled by external code, native completion cannot be established: cleanup
reports the uncertain state and keeps the owner rather than unloading concurrently.

SIGCHLD scheduling explicitly wakes the event loop. Scheduling a task from a raw
Python signal handler alone can leave a standard asyncio selector asleep until
unrelated IO or a timer, delaying worker exit handling and reconciliation.

When a frontend hook substitutes a model or changes its version, registration and
rollback follow instance identity. Earlier keys are removed, the previous committed
instance survives precommit failure, and default selection includes healthy sibling
versions. If preparation targets an already committed version, that target also
survives precommit failure. Publication retires both displaced instances and gives
each a cleanup owner before awaiting either; cleanup failure preserves publication.
Hooks must finalize the worker-visible identity before worker preparation;
renaming a worker model after dispatch is outside this ownership contract.

Confirmed worker exit releases child resources but does not settle parent queue
writers. Pool shutdown and worker retirement close obsolete parent readers only
after exit, then join each feeder with a bound before releasing its owner. A feeder
which cannot stop remains owned and reported. This also prevents interpreter-exit
queue finalizers from silently waiting on a dead consumer.

The feeder and retirement owner serialize closure of each parent reader. Native
descriptor close can release the GIL between checking and clearing the handle;
concurrent close calls could otherwise close the same descriptor twice. The lock
is installed before the feeder starts and adds no work to prediction dispatch.
