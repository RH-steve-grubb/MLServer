"""Optional local inference IPC using ZeroMQ and Queue's pickle serializer.

Only endpoint strings belong on a spawned Worker. Contexts and sockets are
created in their owning process; model control messages keep their queues.
"""

import asyncio
from collections.abc import Callable
from functools import partial
from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
from typing import Any
import uuid
from weakref import WeakKeyDictionary

import zmq
import zmq.asyncio

from .logging import logger
from .errors import WorkerStop
from .messages import ModelRequestMessage, ModelResponseMessage
from .worker import Worker


def create_endpoint(directory: Path, kind: str) -> str:
    """Choose a short, unique IPC path in the parent's private directory."""
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("ZeroMQ IPC directory must already exist and be absolute")
    if kind not in ("q", "r"):
        raise ValueError("ZeroMQ endpoint kind must be q or r")
    path = directory / f"{kind}-{uuid.uuid4().hex[:12]}.sock"
    if len(os.fsencode(path)) >= 100:
        raise ValueError("ZeroMQ IPC pathname must be shorter than 100 bytes")
    return f"ipc://{path}"


def _close_socket(sock: Any, owned: set[Any]) -> None:
    sock.close(linger=0)
    # Release ownership only after native close succeeds. Incomplete sockets
    # cannot be discovered through the usable request/reply fields.
    owned.discard(sock)


def _close_sockets(owned: set[Any]) -> None:
    failures: list[BaseException] = []
    for sock in list(owned):
        try:
            _close_socket(sock, owned)
        except BaseException as error:
            failures.append(error)
    if failures:
        raise failures[0]


def _socket(context: zmq.asyncio.Context, kind: int, owned: set[Any]) -> Any:
    sock = context.socket(kind)
    # A returned native handle needs an actionable owner before configuration
    # or its rollback can fail. Publication into usable routing happens later.
    owned.add(sock)
    # Preserve Queue's unbounded admission policy. Resource limits and request
    # admission remain the server's responsibility, rather than an IPC change.
    try:
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.setsockopt(zmq.LINGER, 0)
        if kind == zmq.PUSH:
            sock.setsockopt(zmq.IMMEDIATE, 1)
    except BaseException as primary:
        try:
            _close_socket(sock, owned)
        except BaseException as cleanup:
            setattr(primary, "cleanup_error", cleanup)
        raise
    return sock


class ParentTransport:
    def __init__(self, reply_endpoint: str) -> None:
        self._reply_endpoint = reply_endpoint
        self._context: zmq.asyncio.Context | None = None
        self._replies: Any = None
        self._requests: dict[Worker, Any] = {}
        self._unregistered_sockets: set[Any] = set()
        self._pending_sends: dict[Worker, set[asyncio.Future[None]]] = {}
        # Queued cross-loop coroutines retain their selected Worker. Keep only
        # its small exit status here, never an exception/traceback retaining it.
        self._stopped_workers: WeakKeyDictionary[Worker, int] = WeakKeyDictionary()
        self._receiver: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._failure: BaseException | None = None
        self._closed = False
        self._close_task: asyncio.Task | None = None

    def start(
        self,
        resolve: Callable[[ModelResponseMessage], None],
        failed: Callable[[BaseException], None],
    ) -> None:
        """Start receiving only after the dispatcher can resolve pending IDs."""
        if self._closed or self._failure is not None:
            raise RuntimeError("ZeroMQ response transport is unavailable")
        if self._receiver is not None:
            return
        self._loop = asyncio.get_running_loop()
        # The dispatcher retains this owner before any native acquisition.
        # A partially successful start is retired by its normal close path.
        try:
            self._context = zmq.asyncio.Context(io_threads=1)
            self._replies = _socket(self._context, zmq.PULL, self._unregistered_sockets)
            self._unregistered_sockets.discard(self._replies)
            self._replies.bind(self._reply_endpoint)
            self._receiver = asyncio.create_task(self._receive(resolve, failed))
        except BaseException as error:
            # A partial start must close through this owner, not overwrite its
            # context on another start while earlier sockets remain unreleased.
            self._failure = error
            raise

    async def _receive(
        self,
        resolve: Callable[[ModelResponseMessage], None],
        failed: Callable[[BaseException], None],
    ) -> None:
        try:
            while True:
                response = ForkingPickler.loads(await self._replies.recv(copy=True))
                if not isinstance(response, ModelResponseMessage):
                    raise TypeError("Unexpected ZeroMQ inference response")
                resolve(response)
                # A ready socket need not suspend; let control operations run.
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure = error
            try:
                self._close_requests()
            except (Exception, asyncio.CancelledError) as cleanup:
                setattr(error, "cleanup_error", cleanup)
                logger.error(
                    "ZeroMQ request cleanup failed; owner retained", exc_info=True
                )
            try:
                failed(error)
            except Exception:
                logger.exception("ZeroMQ response failure callback raised")

    async def send(self, worker: Worker, message: ModelRequestMessage) -> None:
        """Send a request whose response future the dispatcher already tracks."""
        owner = self._loop
        if owner is None or owner.is_closed() or not owner.is_running():
            raise RuntimeError("ZeroMQ response transport is unavailable")
        if asyncio.get_running_loop() is owner:
            await self._send(worker, message)
            return
        # ParallelModel may be called from another thread's event loop. Keep
        # every socket operation on its owner, while forwarding cancellation.
        operation = self._send(worker, message)
        try:
            submitted = asyncio.run_coroutine_threadsafe(operation, owner)
        except BaseException:
            operation.close()
            raise
        try:
            await asyncio.wrap_future(submitted)
        except asyncio.CancelledError:
            submitted.cancel()
            raise

    async def _send(self, worker: Worker, message: ModelRequestMessage) -> None:
        # This first step runs on the socket owner. A stop can precede it even
        # when submission and response registration already happened elsewhere.
        exit_code = self._stopped_workers.get(worker)
        if exit_code is not None:
            raise WorkerStop(exit_code)
        if (
            self._closed
            or self._failure is not None
            or self._receiver is None
            or self._receiver.done()
        ):
            raise RuntimeError(
                "ZeroMQ response transport is unavailable"
            ) from self._failure
        pid = worker.pid
        if pid is None:
            raise ValueError("ZeroMQ inference requires a started worker")
        if worker not in self._requests:
            assert self._context is not None
            sock = _socket(self._context, zmq.PUSH, self._unregistered_sockets)
            try:
                sock.connect(getattr(worker, "_zmq_request_endpoint"))
            except BaseException as primary:
                try:
                    _close_socket(sock, self._unregistered_sockets)
                except BaseException as cleanup:
                    setattr(primary, "cleanup_error", cleanup)
                raise
            self._requests[worker] = sock
            self._unregistered_sockets.discard(sock)
        # Use multiprocessing.Queue's exact default serializer/protocol,
        # including the existing Pydantic and NumPy pickle reducers.
        sending = self._requests[worker].send(ForkingPickler.dumps(message), copy=True)
        if sending.done():
            sending.result()
            return
        # A socket close cancels PyZMQ's send Future. Keep a distinct waiter so
        # worker/transport failure cannot masquerade as caller cancellation.
        # Ready sends keep the original direct path without an extra Future.
        completed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        pending = self._pending_sends.setdefault(worker, set())
        pending.add(completed)
        sending.add_done_callback(partial(self._complete_send, completed))
        try:
            await completed
        finally:
            pending.discard(completed)
            if not pending:
                self._pending_sends.pop(worker, None)
            if not sending.done():
                sending.cancel()
            # Explicit task cancellation may win after worker failure has set
            # an exception but before the waiter resumes. Consume that stored
            # exception without changing the caller's cancellation outcome.
            if completed.done() and not completed.cancelled():
                completed.exception()

    def _complete_send(self, completed: asyncio.Future[None], sending: Any) -> None:
        if sending.cancelled():
            error = self._failure or RuntimeError("ZeroMQ inference send was stopped")
        else:
            error = sending.exception()
        if completed.done():
            return
        if error is not None:
            completed.set_exception(error)
        else:
            completed.set_result(None)

    def forget_worker(self, worker: Worker, exit_code: int = -1) -> None:
        """Revoke this incarnation on the owner loop, even before its first send."""
        previous = self._stopped_workers.get(worker)
        if previous is None or previous == -1:
            self._stopped_workers[worker] = exit_code
        else:
            exit_code = previous
        for completed in self._pending_sends.get(worker, ()):
            if not completed.done():
                completed.set_exception(WorkerStop(exit_code))
        sock = self._requests.get(worker)
        if sock is not None:
            sock.close()
            # A close request which raises does not establish release. Keep
            # the socket reachable for terminal cleanup, separately from routing.
            self._requests.pop(worker, None)

    def _close_requests(self) -> None:
        # A caller can still be awaiting send rather than its response future.
        # Settle that wait independently of native socket closure, which can fail.
        error = self._failure or RuntimeError("ZeroMQ inference send was stopped")
        for pending in self._pending_sends.values():
            for completed in pending:
                if not completed.done():
                    completed.set_exception(error)
        failures: list[BaseException] = []
        for worker, sock in list(self._requests.items()):
            try:
                sock.close()
            except (Exception, asyncio.CancelledError) as error:
                failures.append(error)
            else:
                self._requests.pop(worker, None)
        if failures:
            raise failures[0]

    async def close(self) -> None:
        """Cancel reception and discard unsent IPC messages during shutdown."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._close_done)
        await asyncio.shield(self._close_task)

    def _close_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "ZeroMQ cleanup failed; owner retained", exc_info=task.exception()
            )

    async def _close(self) -> None:
        if self._receiver is not None:
            self._receiver.cancel()
        failures: list[BaseException] = []
        try:
            self._close_requests()
        except (Exception, asyncio.CancelledError) as error:
            failures.append(error)
        try:
            if self._replies is not None:
                self._replies.close()
        except (Exception, asyncio.CancelledError) as error:
            failures.append(error)
        try:
            _close_sockets(self._unregistered_sockets)
        except BaseException as error:
            failures.append(error)
        if self._receiver is not None:
            await asyncio.gather(self._receiver, return_exceptions=True)
        if failures:
            # Include sockets whose setup never reached a usable field.
            # term() can block with an unclosed socket. Retain that context
            # and its failure rather than blocking the loop or claiming release.
            raise failures[0]
        if self._context is not None:
            self._context.term()


async def _reply(worker: Worker, request: ModelRequestMessage, replies: Any) -> None:
    response = await worker._process_request(request)
    if not isinstance(response, ModelResponseMessage):
        raise TypeError("Unexpected worker inference response")
    await replies.send(ForkingPickler.dumps(response), copy=True)


async def serve_requests(worker: Worker) -> None:
    """Serve data-plane requests alongside the worker's control-plane loop."""
    context = zmq.asyncio.Context(io_threads=1)
    requests: Any = None
    replies: Any = None
    pending: set[asyncio.Task] = set()
    owned: set[Any] = set()
    # Child-only native resources never cross spawn. A failed cleanup remains
    # owned by the worker until release or its parent's confirmed process exit.
    worker._zmq_resources = (context, owned, pending)
    failure = asyncio.get_running_loop().create_future()
    incoming: Any = None
    primary: BaseException | None = None

    def completed(task: asyncio.Task) -> None:
        pending.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None and not failure.done():
                failure.set_exception(error)

    try:
        requests = _socket(context, zmq.PULL, owned)
        replies = _socket(context, zmq.PUSH, owned)
        requests.bind(getattr(worker, "_zmq_request_endpoint"))
        replies.connect(getattr(worker, "_zmq_reply_endpoint"))
        while True:
            incoming = requests.recv(copy=True)
            done, _ = await asyncio.wait(
                (incoming, failure), return_when=asyncio.FIRST_COMPLETED
            )
            if failure in done:
                failure.result()
            request = ForkingPickler.loads(incoming.result())
            if not isinstance(request, ModelRequestMessage):
                raise TypeError("Unexpected ZeroMQ inference request")
            # Queue workers also schedule async methods independently. Keep
            # that concurrency instead of serializing models that await I/O.
            task = asyncio.create_task(_reply(worker, request, replies))
            pending.add(task)
            task.add_done_callback(completed)
            await asyncio.sleep(0)
    except BaseException as error:
        primary = error
        raise
    finally:
        if incoming is not None:
            incoming.cancel()
        for task in pending:
            task.cancel()
        # _process_request may turn cancellation into an error response.
        # Closing first prevents that reply from waiting for a missing peer.
        if failure.done() and not failure.cancelled():
            failure.exception()
        else:
            failure.cancel()
        try:
            _close_sockets(owned)
            await asyncio.gather(*pending, return_exceptions=True)
            # Do not enter blocking termination with an unreleased handle.
            # A failure here is also cleanup, not a replacement setup error.
            context.term()
        except BaseException as cleanup:
            logger.error(
                "Worker socket cleanup failed; resources remain owned", exc_info=True
            )
            if primary is None:
                raise
            setattr(primary, "cleanup_error", cleanup)
        else:
            worker._zmq_resources = None
