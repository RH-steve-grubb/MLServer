"""Optional local inference IPC using ZeroMQ and Queue's pickle serializer.

Only endpoint strings belong on a spawned Worker. Contexts and sockets are
created in their owning process; model control messages keep their queues.
"""

import asyncio
from collections.abc import Callable
from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
from typing import Any
import uuid

import zmq
import zmq.asyncio

from .logging import logger
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


def _socket(context: zmq.asyncio.Context, kind: int) -> Any:
    sock = context.socket(kind)
    # Preserve Queue's unbounded admission policy. Resource limits and request
    # admission remain the server's responsibility, rather than an IPC change.
    try:
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.setsockopt(zmq.LINGER, 0)
        if kind == zmq.PUSH:
            sock.setsockopt(zmq.IMMEDIATE, 1)
    except BaseException:
        sock.close(linger=0)
        raise
    return sock


class ParentTransport:
    def __init__(self, reply_endpoint: str) -> None:
        self._context = zmq.asyncio.Context(io_threads=1)
        self._replies: Any = None
        self._requests: dict[int, Any] = {}
        self._receiver: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._failure: BaseException | None = None
        self._closed = False
        try:
            self._replies = _socket(self._context, zmq.PULL)
            self._replies.bind(reply_endpoint)
        except BaseException:
            if self._replies is not None:
                self._replies.close()
            self._context.term()
            raise

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
        self._receiver = asyncio.create_task(self._receive(resolve, failed))

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
            self._close_requests()
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
        if pid not in self._requests:
            sock = _socket(self._context, zmq.PUSH)
            try:
                sock.connect(getattr(worker, "_zmq_request_endpoint"))
            except BaseException:
                sock.close()
                raise
            self._requests[pid] = sock
        # Use multiprocessing.Queue's exact default serializer/protocol,
        # including the existing Pydantic and NumPy pickle reducers.
        await self._requests[pid].send(ForkingPickler.dumps(message), copy=True)

    def forget_worker(self, pid: int) -> None:
        """Release the request socket when its worker leaves the pool."""
        sock = self._requests.pop(pid, None)
        if sock is not None:
            sock.close()

    def _close_requests(self) -> None:
        for sock in self._requests.values():
            sock.close()
        self._requests.clear()

    async def close(self) -> None:
        """Cancel reception and discard unsent IPC messages during shutdown."""
        if self._closed:
            return
        self._closed = True
        if self._receiver is not None:
            self._receiver.cancel()
        self._close_requests()
        self._replies.close()
        if self._receiver is not None:
            await asyncio.gather(self._receiver, return_exceptions=True)
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
    failure = asyncio.get_running_loop().create_future()
    incoming: Any = None

    def completed(task: asyncio.Task) -> None:
        pending.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None and not failure.done():
                failure.set_exception(error)

    try:
        requests = _socket(context, zmq.PULL)
        replies = _socket(context, zmq.PUSH)
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
    finally:
        if incoming is not None:
            incoming.cancel()
        for task in pending:
            task.cancel()
        # _process_request may turn cancellation into an error response.
        # Closing first prevents that reply from waiting for a missing peer.
        if requests is not None:
            requests.close()
        if replies is not None:
            replies.close()
        await asyncio.gather(*pending, return_exceptions=True)
        if failure.done() and not failure.cancelled():
            failure.exception()
        else:
            failure.cancel()
        context.term()
