import os
import uuid
import asyncio
import inspect
import urllib.parse
from contextvars import ContextVar

from asyncio import Task
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar
from functools import wraps

from .logging import logger
from .types import InferenceRequest, InferenceResponse, Parameters
from .settings import ModelSettings, DEFAULT_MODEL_OPERATION_TIMEOUT
from .errors import InvalidModelURI
from .version import __version__


T = TypeVar("T")
P = ParamSpec("P")


_deferred_cancellation_active: ContextVar[bool] = ContextVar(
    "deferred_cancellation_active", default=False
)


async def get_model_uri(
    settings: ModelSettings, wellknown_filenames: list[str] = []
) -> str:
    if not settings.parameters:
        raise InvalidModelURI(settings.name)

    model_uri = settings.parameters.uri
    if not model_uri:
        raise InvalidModelURI(settings.name)

    model_uri_components = urllib.parse.urlparse(model_uri, scheme="file")
    if model_uri_components.scheme != "file":
        return model_uri

    full_model_path = to_absolute_path(settings, model_uri_components.path)
    if os.path.isfile(full_model_path):
        return full_model_path

    if os.path.isdir(full_model_path):
        # If full_model_path is a folder, search for a well-known model filename
        for fname in wellknown_filenames:
            model_path = os.path.join(full_model_path, fname)
            if os.path.isfile(model_path):
                return model_path

        # If none, return the folder
        return full_model_path

    # Otherwise, the uri is neither a file nor a folder
    raise InvalidModelURI(settings.name, full_model_path)


def to_absolute_path(model_settings: ModelSettings, uri: str) -> str:
    source = model_settings._source
    if source is None:
        # Treat path as either absolute or relative to the working directory of
        # the MLServer instance
        return uri

    parent_folder = os.path.dirname(source)
    unnormalised = os.path.join(parent_folder, uri)
    return os.path.normpath(unnormalised)


def get_wrapped_method(f: Callable) -> Callable:
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__  # type: ignore

    return f


def generate_uuid() -> str:
    return str(uuid.uuid4())


def insert_headers(
    inference_request: InferenceRequest, headers: dict[str, str]
) -> InferenceRequest:
    # Ensure parameters are present
    if inference_request.parameters is None:
        inference_request.parameters = Parameters()

    parameters = inference_request.parameters

    if parameters.headers is not None:
        # TODO: Raise warning that headers will be replaced and shouldn't be used
        logger.warning(
            f"There are {len(parameters.headers)} entries present in the"
            "`headers` field of the request `parameters` object."
            "The `headers` field of the `parameters` object "
            "SHOULDN'T BE USED directly."
            "These entries will be replaced by the actual headers (REST) "
            "or metadata (gRPC) of the incoming request."
        )

    parameters.headers = headers
    return inference_request


def extract_headers(inference_response: InferenceResponse) -> dict[str, str] | None:
    if inference_response.parameters is None:
        return None

    parameters = inference_response.parameters
    if parameters.headers is None:
        return None

    headers = parameters.headers
    parameters.headers = None
    return headers


def _check_current_event_loop_policy() -> str:
    policy = (
        "uvloop"
        if type(asyncio.get_event_loop_policy()).__module__.startswith("uvloop")
        else "asyncio"
    )
    return policy


def install_uvloop_event_loop():
    if "uvloop" == _check_current_event_loop_policy():
        return

    try:
        import uvloop

        uvloop.install()
    except ImportError:
        # else keep the standard asyncio loop as a fallback
        pass

    policy = _check_current_event_loop_policy()

    logger.debug(f"Using asyncio event-loop policy: {policy}")


def schedule_with_callback(coro, cb) -> Task:
    task = asyncio.create_task(coro)
    task.add_done_callback(cb)
    return task


def get_normalized_version(version: str | None = None) -> str:
    """
    Return a public version string without local build metadata.

    Example:
    - 1.7.1+rhaiv.8 -> 1.7.1
    - 1.7.1 -> 1.7.1
    - 1.7.0.dev0 -> 1.7.0.dev0
    """
    resolved_version = version or __version__
    return resolved_version.split("+", 1)[0]


async def _defer_cancellation(
    operation: Awaitable[T], timeout: float = DEFAULT_MODEL_OPERATION_TIMEOUT
) -> T:
    """Wait for an operation to settle while deferring caller cancellation.

    Caller cancellation is suppressed during the timeout window. If timeout expires,
    the underlying operation is cancelled.

    If the operation completes during the timeout window its result is returned or
    any exception is raised. If timeout expires a ``TimeoutError`` is raised.
    """
    operation = asyncio.wait_for(operation, timeout)
    task = asyncio.ensure_future(operation)
    caller = asyncio.current_task()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # This helper intentionally defers caller cancellation. Catching
            # CancelledError does not clear asyncio's recorded requests, so
            # clear every pending request before the operation resumes. A
            # task can receive one CancelledError after multiple cancel() calls.
            if (
                caller is not None
                and hasattr(caller, "cancelling")
                and hasattr(caller, "uncancel")
            ):
                while caller.cancelling():
                    caller.uncancel()
            continue
    return task.result()


def defer_cancellation(
    timeout: float | Callable[..., float] = DEFAULT_MODEL_OPERATION_TIMEOUT,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Apply one timeout and deferred-cancellation boundary to an operation.

    Nested decorated calls reuse the outer boundary and timeout.
    ``timeout`` may be a value or a callable resolved at invocation time.
    """

    def decorate(
        operation_method: Callable[P, Awaitable[T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(operation_method)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            # Only the outermost operation owns the timeout and deferred
            # cancellation. Nested decorated methods run inside that boundary.
            if _deferred_cancellation_active.get():
                return await operation_method(*args, **kwargs)

            operation_timeout = timeout
            if callable(operation_timeout):
                operation_timeout = operation_timeout(*args, **kwargs)

            active_token = _deferred_cancellation_active.set(True)
            try:
                return await _defer_cancellation(
                    operation_method(*args, **kwargs), operation_timeout
                )
            finally:
                _deferred_cancellation_active.reset(active_token)

        return wrapped

    return decorate


def with_operation_lock(
    lock_for: Callable[..., asyncio.Lock | Awaitable[asyncio.Lock]],
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Acquire an operation lock before running the decorated method.

    ``lock_for`` may return a lock directly or an awaitable.
    """

    def decorate(
        method: Callable[P, Awaitable[T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(method)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            lock = lock_for(*args, **kwargs)
            if inspect.isawaitable(lock):
                lock = await lock
            async with lock:
                return await method(*args, **kwargs)

        return wrapped

    return decorate
