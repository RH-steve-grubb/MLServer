from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from time import perf_counter
from prometheus_client import (
    Counter,
    Summary,
)
from collections.abc import AsyncIterator, Iterator

from ..model import MLModel
from ..errors import ModelNotReady
from ..context import model_context
from ..settings import Settings
from ..registry import MultiModelRegistry
from ..types import (
    MetadataModelResponse,
    MetadataServerResponse,
    InferenceRequest,
    InferenceResponse,
    RuntimeSecurityResponse,
    RuntimeSecurityMode,
)
from ..middleware import InferenceMiddlewares
from ..cloudevents import CloudEventsMiddleware
from ..utils import generate_uuid


@dataclass
class _InferenceMetrics:
    enabled: bool = True
    excluded_duration: float = 0.0
    inference_succeeded: bool = False


class DataPlane:
    """
    Internal implementation of handlers, used by both the gRPC and REST
    servers.
    """

    def __init__(
        self,
        settings: Settings,
        model_registry: MultiModelRegistry,
        *,
        inference_metrics: bool = True,
    ):
        self._settings = settings
        self._model_registry = model_registry
        self._inference_metrics_enabled = inference_metrics
        self._inference_middleware = InferenceMiddlewares(
            CloudEventsMiddleware(settings)
        )

        # TODO: Update to standardised set of labels
        self._ModelInferRequestSuccess = Counter(
            "model_infer_request_success",
            "Model infer request success count",
            ["model", "version"],
        )
        self._ModelInferRequestFailure = Counter(
            "model_infer_request_failure",
            "Model infer request failure count",
            ["model", "version"],
        )
        self._ModelInferRequestDuration = Summary(
            "model_infer_request_duration",
            "Model infer request duration",
            ["model", "version"],
        )

    async def live(self) -> bool:
        return True

    async def ready(self) -> bool:
        # Return false if startup has not completed
        if not self._model_registry.is_startup_complete:
            return False

        # Fetch models from the registry
        models = list(await self._model_registry.get_models())

        # Apply empty readiness mode based on setting
        if not models:
            return self._settings.empty_registry_readiness

        # Apply strict or lenient readiness mode based on setting
        if self._settings.strict_readiness:
            # Strict mode: ALL models must be ready
            return all(model.ready for model in models)
        else:
            # Lenient mode: AT LEAST ONE model must be ready
            return any(model.ready for model in models)

    async def model_ready(self, name: str, version: str | None = None) -> bool:
        model = await self._model_registry.get_model(name, version)
        return model.ready

    async def metadata(self) -> MetadataServerResponse:
        # Built-in extensions that are always available
        builtin_extensions = ["model_repository", "runtime_security"]
        # Combine with user-configured extensions
        all_extensions = builtin_extensions + self._settings.extensions
        return MetadataServerResponse(
            name=self._settings.server_name,
            version=self._settings.server_version,
            extensions=all_extensions,
        )

    @staticmethod
    async def runtimes() -> RuntimeSecurityResponse:
        from ..settings import (
            _load_image_baked_allowed_model_implementations,
            _get_trusted_runtimes_artifact_path,
        )

        allowed = _load_image_baked_allowed_model_implementations(
            _get_trusted_runtimes_artifact_path()
        )

        if allowed is not None:
            # Production mode: file exists
            return RuntimeSecurityResponse(
                mode=RuntimeSecurityMode.PRODUCTION,
                allowed_model_implementations=sorted(allowed),
            )
        else:
            # Development mode: no file
            return RuntimeSecurityResponse(
                mode=RuntimeSecurityMode.DEVELOPMENT,
                allowed_model_implementations=None,
            )

    async def model_metadata(
        self, name: str, version: str | None = None
    ) -> MetadataModelResponse:
        model = await self._model_registry.get_model(name, version)
        # TODO: Make await optional for sync methods
        with model_context(model.settings):
            return await model.metadata()

    async def infer(
        self,
        payload: InferenceRequest,
        name: str,
        version: str | None = None,
    ) -> InferenceResponse:
        async with self._infer_contextmanager(name, version) as model:
            payload = self._prepare_payload(payload, model)

            prediction = await model.predict(payload)

            # Ensure ID matches
            prediction.id = payload.id
            self._inference_middleware.response_middleware(prediction, model.settings)
            return prediction

    async def infer_stream(
        self,
        payloads: AsyncIterator[InferenceRequest],
        name: str,
        version: str | None = None,
    ) -> AsyncIterator[InferenceResponse]:

        async with self._infer_contextmanager(name, version) as model:
            # we need to get the first payload to get the ID
            async for payload in payloads:
                break

            payloads_decorated = self._prepare_payloads_generator(
                payload, payloads, model
            )
            async for prediction in model.predict_stream(payloads_decorated):
                prediction.id = payload.id  # Ensure ID matches
                self._inference_middleware.response_middleware(
                    prediction, model.settings
                )
                yield prediction

    def _prepare_payload(
        self, payload: InferenceRequest, model: MLModel
    ) -> InferenceRequest:
        if self._settings.grpc_raw_input_buffers and (
            model.settings.max_batch_size or model.settings.max_batch_time
        ):
            from ..grpc.buffers import restore_list_inputs

            # Batching merges the public list representation. Keep compact
            # NumPy buffers on the measured unbatched path, while preserving
            # the existing batching contract without changing the batcher.
            restore_list_inputs(payload)
        if payload.id is None:
            payload.id = generate_uuid()

        self._inference_middleware.request_middleware(payload, model.settings)
        return payload

    async def _prepare_payloads_generator(
        self,
        first_payload: InferenceRequest,
        subsequent_payloads: AsyncIterator[InferenceRequest],
        model: MLModel,
    ) -> AsyncIterator[InferenceRequest]:
        # yield the first payload after preparing it
        first_payload = self._prepare_payload(first_payload, model)
        yield first_payload

        # Yield the rest of the payloads after preparing them
        # and set the ID to match the first payload. Note that
        # we don't make any assumptions about how many inputs and
        # outputs there are. Thus, everything gets the same ID, cause
        # otherwise we could have one to many, many to one, or many to
        # many id mappings.
        async for payload in subsequent_payloads:
            payload.id = first_payload.id
            payload = self._prepare_payload(payload, model)
            yield payload

    @contextmanager
    def _infer_metrics(
        self, name: str, version: str | None
    ) -> Iterator[_InferenceMetrics]:
        # Raw REST learns whether the body is valid from the worker's reply.
        # Defer recording until that is known; schema errors never entered the
        # standard DataPlane and must not become inference failures here.
        observation = _InferenceMetrics(enabled=self._inference_metrics_enabled)
        started = perf_counter()
        try:
            yield observation
        except BaseException as error:
            # Match count_exceptions(): caller cancellation is not an inference
            # failure, but a completed timing observation still belongs to it.
            if observation.enabled:
                if observation.inference_succeeded:
                    self._ModelInferRequestSuccess.labels(
                        model=name, version=version
                    ).inc()
                elif isinstance(error, Exception):
                    self._ModelInferRequestFailure.labels(
                        model=name, version=version
                    ).inc()
            raise
        else:
            if observation.enabled:
                self._ModelInferRequestSuccess.labels(model=name, version=version).inc()
        finally:
            if observation.enabled:
                elapsed = max(
                    0.0, perf_counter() - started - observation.excluded_duration
                )
                self._ModelInferRequestDuration.labels(
                    model=name, version=version
                ).observe(elapsed)

    @asynccontextmanager
    async def _infer_contextmanager(
        self,
        name: str,
        version: str | None = None,
    ) -> AsyncIterator[MLModel]:
        with self._infer_metrics(name, version):
            model = await self._model_registry.get_model(name, version)
            if not model.ready:
                raise ModelNotReady(name, version)

            with model_context(model.settings):
                yield model
