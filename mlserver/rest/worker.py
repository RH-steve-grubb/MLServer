"""Opt-in REST JSON processing in the selected model's inference worker.

The frontend forwards the original JSON bytes, keeping the large Python input
object graph inside one worker. Parsing, validation and response encoding retain
the normal REST algorithms; inference still passes through the DataPlane.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from typing import Any, TYPE_CHECKING

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import TypeAdapter, ValidationError
from starlette.requests import Request
from starlette.responses import Response

from ..codecs.json import decode_from_bytelike_json_to_dict, encode_to_json_bytes
from ..errors import ModelNotFound, ModelNotReady
from ..handlers import DataPlane
from ..types import InferenceRequest, InferenceResponse
from ..utils import extract_headers, insert_headers

if TYPE_CHECKING:
    from ..parallel.messages import ModelRequestMessage, ModelResponseMessage
    from ..parallel.worker import Worker

RAW_INFER_METHOD = "__mlserver_rest_worker_infer__"
_RESPONSE_ADAPTER = TypeAdapter(InferenceResponse)


@dataclass
class RawReply:
    body: bytes
    headers: dict[str, str]
    status: int = 200
    worker_pid: int = field(default_factory=os.getpid)


def _http_response(reply: RawReply) -> Response:
    return Response(
        reply.body,
        status_code=reply.status,
        headers=reply.headers,
        media_type="application/json",
    )


def _validation_reply(errors: list[dict[str, Any]]) -> RawReply:
    return RawReply(
        encode_to_json_bytes({"detail": jsonable_encoder(errors)}), {}, status=422
    )


def _missing_body() -> RawReply:
    return _validation_reply(
        [{"type": "missing", "loc": ("body",), "msg": "Field required", "input": None}]
    )


def _json_content(headers: dict[str, str]) -> bool:
    content_type = headers.get("content-type")
    if content_type is None:
        return True
    media_type = content_type.partition(";")[0].strip().lower()
    return media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )


def parse_body(body: bytes, json_content: bool = True) -> InferenceRequest | RawReply:
    """Use the existing JSON-object/schema path and FastAPI error locations."""
    if not body:
        return _missing_body()
    decoded: Any = body
    if json_content:
        try:
            decoded = decode_from_bytelike_json_to_dict(body)
        except json.JSONDecodeError as error:
            return _validation_reply(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body", error.pos),
                        "msg": "JSON decode error",
                        "input": {},
                        "ctx": {"error": error.msg},
                    }
                ]
            )
    if decoded is None:
        # FastAPI treats JSON null as an absent required request body.
        return _missing_body()
    try:
        return InferenceRequest.model_validate(decoded, from_attributes=True)
    except ValidationError as error:
        errors: list[dict[str, Any]] = [
            dict(item) for item in error.errors(include_url=False)
        ]
        for item in errors:
            item["loc"] = ("body", *tuple(item["loc"]))
        return _validation_reply(errors)


def encode_response(response: InferenceResponse) -> bytes:
    """Apply the same validated schema and serialization flags as REST."""
    checked = _RESPONSE_ADAPTER.validate_python(response, from_attributes=True)
    content = _RESPONSE_ADAPTER.dump_python(
        checked, mode="json", by_alias=True, exclude_unset=True, exclude_none=True
    )
    return encode_to_json_bytes(content)


async def process_raw_request(
    worker: Worker, request: ModelRequestMessage
) -> ModelResponseMessage:
    """Handle the dedicated raw REST message inside its ordinary model worker."""
    # Import at the dispatch boundary: parallel startup imports REST app setup.
    from ..parallel.errors import WorkerError
    from ..parallel.messages import ModelResponseMessage

    try:
        if request.method_name != RAW_INFER_METHOD:
            raise ValueError("Unexpected method for raw REST inference")
        body, headers = request.method_args
        if not isinstance(body, bytes) or not isinstance(headers, dict):
            raise TypeError("Raw REST inference requires bytes and HTTP headers")
        payload = parse_body(body, _json_content(headers))
        if isinstance(payload, RawReply):
            return ModelResponseMessage(id=request.id, return_value=payload)
        insert_headers(payload, headers)
        data_plane = getattr(worker, "_rest_raw_data_plane", None)
        if data_plane is None:
            # Keep middleware, model context and inference metrics on the worker
            # where inference happens, rather than bypassing the DataPlane.
            data_plane = DataPlane(worker._settings, worker._model_registry)
            setattr(worker, "_rest_raw_data_plane", data_plane)
        response = await data_plane.infer(
            payload, request.model_name, request.model_version
        )
        response_headers = extract_headers(response) or {}
        reply = RawReply(encode_response(response), response_headers)
        return ModelResponseMessage(id=request.id, return_value=reply)
    except (Exception, asyncio.CancelledError) as error:
        return ModelResponseMessage(id=request.id, exception=WorkerError(error))


class RawEndpoints:
    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane

    async def infer(
        self, request: Request, model_name: str, model_version: str | None = None
    ) -> Response:
        from ..parallel.messages import ModelRequestMessage
        from ..parallel.model import ParallelModel

        try:
            body = await request.body()
        except HTTPException:
            raise
        except Exception as error:
            # A Request-only endpoint bypasses FastAPI's body parser, including
            # its conversion of read/decompression errors into a JSON 400.
            raise HTTPException(
                status_code=400, detail="There was an error parsing the body"
            ) from error
        headers = dict(request.headers)
        try:
            model = await self._data_plane._model_registry.get_model(
                model_name, model_version
            )
            if not model.ready:
                raise ModelNotReady(model_name, model_version)
        except (ModelNotFound, ModelNotReady):
            # FastAPI validates the body before calling the ordinary endpoint.
            # Preserve that 422 precedence on lookup/readiness failures without
            # building input objects on the valid parallel-worker hot path.
            invalid = parse_body(body, _json_content(headers))
            if isinstance(invalid, RawReply):
                return _http_response(invalid)
            raise
        if (
            isinstance(model, ParallelModel)
            and not model.settings.max_batch_size
            and not model.settings.max_batch_time
            and _json_content(headers)
        ):
            message = ModelRequestMessage(
                model_name=model.name,
                model_version=model.version,
                method_name=RAW_INFER_METHOD,
                method_args=[body, headers],
            )
            response = await model._dispatcher.dispatch_request(message)
            reply = response.return_value
            if (
                not isinstance(reply, RawReply)
                or not isinstance(reply.body, bytes)
                or reply.worker_pid == os.getpid()
            ):
                raise TypeError("Raw REST inference did not return a worker reply")
        else:
            # Batching may wrap predict() on the frontend. Retain that entry
            # point for batching and for models that have no inference worker.
            payload = parse_body(body, _json_content(headers))
            if isinstance(payload, RawReply):
                reply = payload
            else:
                insert_headers(payload, headers)
                prediction = await self._data_plane.infer(
                    payload, model_name, model_version
                )
                response_headers = extract_headers(prediction) or {}
                reply = RawReply(encode_response(prediction), response_headers)
        return _http_response(reply)
