"""A native worker must exit when its independent inference loop fails."""

import asyncio
from multiprocessing import Queue
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import tempfile

import pytest

zmq = pytest.importorskip("zmq")
from mlserver.parallel.utils import configure_inference_pool  # noqa: E402
from mlserver.parallel.messages import ModelResponseMessage  # noqa: E402
from mlserver.parallel.worker import Worker  # noqa: E402
from mlserver.parallel.zmq import create_endpoint  # noqa: E402
from mlserver.settings import Settings  # noqa: E402


async def test_native_worker_exits_on_data_failure_without_control_sentinel(
    settings: Settings,
) -> None:
    settings.parallel_workers = 1
    settings.parallel_inference_transport = "zmq"
    settings.metrics_endpoint = None
    configure_inference_pool(settings)
    with tempfile.TemporaryDirectory(prefix="mls-exit-") as location:
        directory = Path(location)
        endpoint = create_endpoint(directory, "r")
        responses: Queue[ModelResponseMessage] = Queue()
        setattr(responses, "_zmq_reply_endpoint", endpoint)
        worker = Worker(settings, responses)
        context = zmq.asyncio.Context()
        replies = context.socket(zmq.PULL)
        replies.setsockopt(zmq.LINGER, 0)
        replies.bind(endpoint)
        requests = context.socket(zmq.PUSH)
        requests.setsockopt(zmq.LINGER, 0)
        requests.setsockopt(zmq.IMMEDIATE, 1)
        try:
            worker.start()
            requests.connect(getattr(worker, "_zmq_request_endpoint"))
            # IMMEDIATE waits until the child's data socket has bound. Its
            # ordinary control coroutine is already waiting in executor poll.
            await asyncio.wait_for(
                requests.send(ForkingPickler.dumps({"not": "ModelRequestMessage"})),
                timeout=5,
            )
            # There is deliberately no Pool/SIGCHLD reaper or shutdown sentinel
            # before this assertion: data-loop failure must end the process.
            await asyncio.to_thread(worker.join, 5)
            assert worker.exitcode is not None, (
                f"Worker {worker.pid} remained alive after malformed ZeroMQ input; "
                "its idle control poll was not released by data-loop failure"
            )
            assert worker.exitcode != 0
        finally:
            # A sentinel is permitted only for cleanup, after measuring exit.
            await worker.stop()
            if worker.pid is not None:
                await asyncio.to_thread(worker.join, 2)
                if worker.is_alive():
                    worker.kill()
                    await asyncio.to_thread(worker.join, 2)
                assert not worker.is_alive()
                worker.close()
            requests.close()
            replies.close()
            context.term()
            responses.close()
            responses.join_thread()
            worker._requests.join_thread()
            worker._model_updates.join_thread()
