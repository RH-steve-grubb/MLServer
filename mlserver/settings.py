import importlib
import inspect
import json
import logging
import os
import re
import sys
import uuid
import warnings
from contextlib import contextmanager
from functools import lru_cache

from typing import (
    Any,
    ClassVar,
    Literal,
    no_type_check,
    TYPE_CHECKING,
)
from typing_extensions import Self
from pydantic import (
    ImportString,
    Field,
    AliasChoices,
    field_validator,
    ValidationError,
)
from pydantic import model_validator
from pydantic._internal._validators import import_string
import pydantic_settings
from pydantic_settings import SettingsConfigDict

from .version import __version__
from .types import MetadataTensor

ENV_FILE_SETTINGS = ".env"
ENV_PREFIX_SETTINGS = "MLSERVER_"
ENV_PREFIX_MODEL_SETTINGS = "MLSERVER_MODEL_"

DEFAULT_PARALLEL_WORKERS = 1

DEFAULT_ENVIRONMENTS_DIR = os.path.join(os.getcwd(), ".envs")
DEFAULT_METRICS_DIR = os.path.join(os.getcwd(), ".metrics")
TRUSTED_RUNTIMES_ARTIFACT_PATH = "/etc/mlserver/trusted-runtimes.json"
# Canonical runtime import-path regex used by runtime and CLI validation.
# Require explicit dotted paths (`module.ClassName`) and disallow leading
# underscores on each segment to keep runtime declarations explicit.
# Hyphens are allowed in module segments but NOT in the final class-name
# segment (which must be a valid Python identifier).
RUNTIME_IMPORT_PATH_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*\.[A-Za-z][A-Za-z0-9_]*$"
)

logger = logging.getLogger(__name__)


def is_valid_runtime_import_path(value: object) -> bool:
    if not isinstance(value, str) or not RUNTIME_IMPORT_PATH_PATTERN.fullmatch(value):
        return False
    _, _, attr = value.rpartition(".")
    return attr[:1].isupper()


ALLOWED_MODEL_IMPLEMENTATIONS = {
    "mlserver_sklearn.SKLearnModel",
    "mlserver_xgboost.XGBoostModel",
    "mlserver_lightgbm.LightGBMModel",
    "mlserver_onnx.OnnxModel",
}


_BUILTIN_RUNTIME_IMPORT_PATH_ALIASES = {
    "mlserver_sklearn.sklearn.SKLearnModel": "mlserver_sklearn.SKLearnModel",
    "mlserver_xgboost.xgboost.XGBoostModel": "mlserver_xgboost.XGBoostModel",
    "mlserver_lightgbm.lightgbm.LightGBMModel": "mlserver_lightgbm.LightGBMModel",
    "mlserver_onnx.onnx.OnnxModel": "mlserver_onnx.OnnxModel",
}


def _is_missing_or_empty_field_error(exc: ValidationError, field: str) -> bool:
    """Return whether a field error represents a missing or empty value."""
    return any(
        error.get("loc") == (field,)
        and (error.get("type") == "missing" or error.get("input") == "")
        for error in exc.errors()
    )


def canonicalize_runtime_import_path(import_path: str) -> str:
    return _BUILTIN_RUNTIME_IMPORT_PATH_ALIASES.get(import_path, import_path)


def log_runtime_security_mode() -> None:
    """Log the runtime security mode at server startup."""
    allowed = _load_image_baked_allowed_model_implementations(
        _get_trusted_runtimes_artifact_path()
    )

    # Development mode: no trusted runtimes allowlist file exists
    if allowed is None:
        logger.info(
            "Runtime security: DEVELOPMENT - all model implementations "
            "allowed (no trusted runtimes allowlist file found)"
        )
        return

    # Production mode: trusted runtimes allowlist file exists
    logger.info(
        "Runtime security: PRODUCTION - %d model implementations allowed "
        "from trusted runtimes allowlist file: %s",
        len(allowed),
        sorted(allowed),
    )

    if not allowed:
        logger.warning(
            "Trusted runtimes allowlist file exists but is empty - "
            "no models can be loaded! Either add model implementation "
            "entries or remove the file for development mode."
        )


def clear_trusted_runtime_caches() -> None:
    """Clear trusted runtimes allowlist cache.

    Useful for tests or runtime reconfiguration where trusted runtime sources
    may change during process lifetime.
    """
    _load_image_baked_allowed_model_implementations.cache_clear()


def _assert_trusted_runtime_import_path(import_path: str) -> None:
    if not is_valid_runtime_import_path(import_path):
        raise ValueError("Model implementation has an invalid import path.")

    allowed = _load_image_baked_allowed_model_implementations(
        _get_trusted_runtimes_artifact_path()
    )

    # If allowlist file does not exist, allow any runtime (development mode)
    if allowed is None:
        logger.debug(
            "No trusted runtimes allowlist configured - "
            "allowing model implementation %s",
            import_path,
        )
        return

    # If trusted runtimes allowlist file does exist, enforce allowlist (production mode)
    if import_path not in allowed:
        logger.warning(
            "Rejected untrusted model implementation %r.",
            import_path,
        )
        raise ValueError(
            f"Model implementation {import_path!r} is not included in the "
            "trusted runtimes allowlist configuration."
        )


# Conditionally imported due to cyclic dependencies
if TYPE_CHECKING:
    from ..model import MLModel


@contextmanager
def _extra_sys_path(extra_path: str):
    """Context manager to temporarily add a path to sys.path for dynamic imports.

    This is used in development mode to allow loading custom runtimes from
    model folders without requiring them to be installed packages.
    """
    sys.path.insert(0, extra_path)
    try:
        yield
    finally:
        try:
            sys.path.remove(extra_path)
        except ValueError:
            pass


def _reload_module(import_path: str):
    """Reload a module to ensure fresh import in dynamic loading scenarios.

    This is used in development mode when loading runtimes from model folders
    to ensure we get the latest version of the module.
    """
    if not import_path:
        return

    module_path, _, _ = import_path.rpartition(".")
    module = importlib.import_module(module_path)
    importlib.reload(module)


def _get_import_path(klass: type[Any]):
    import_path = f"{klass.__module__}.{klass.__name__}"
    return canonicalize_runtime_import_path(import_path)


def _get_trusted_runtimes_artifact_path() -> str:
    return TRUSTED_RUNTIMES_ARTIFACT_PATH


@lru_cache(maxsize=32)
def _load_image_baked_allowed_model_implementations(
    artifact_path: str,
) -> frozenset[str] | None:
    if not os.path.lexists(artifact_path):
        return None
    if not os.path.isfile(artifact_path):
        raise ValueError(
            f"Trusted runtimes artifact {artifact_path!r} must be a regular file."
        )

    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            runtimes = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Trusted runtimes artifact {artifact_path!r} could not be loaded."
        ) from exc

    if not isinstance(runtimes, list):
        raise ValueError(
            "Trusted runtimes artifact must be a JSON list of import paths."
        )

    allowed = set()
    for runtime in runtimes:
        if not isinstance(runtime, str) or runtime != runtime.strip():
            raise ValueError(
                "Trusted runtimes artifact contains an invalid runtime import path."
            )
        runtime = canonicalize_runtime_import_path(runtime)
        if not is_valid_runtime_import_path(runtime):
            raise ValueError(
                "Trusted runtimes artifact contains an invalid runtime import path."
            )
        allowed.add(runtime)

    return frozenset(allowed)


class BaseSettings(pydantic_settings.BaseSettings):
    @no_type_check
    def __setattr__(self, name, value):
        """
        Patch __setattr__ to be able to use property setters.
        From:
            https://github.com/pydantic/pydantic/issues/1577#issuecomment-790506164
        """
        try:
            super().__setattr__(name, value)
        except ValueError as e:
            setters = inspect.getmembers(
                self.__class__,
                predicate=lambda x: isinstance(x, property) and x.fset is not None,
            )
            for setter_name, func in setters:
                if setter_name == name:
                    object.__setattr__(self, name, value)
                    break
            else:
                raise e

    def dict(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().dict(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )

    def json(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().json(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )


class CORSSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_SETTINGS,
        env_prefix=ENV_PREFIX_SETTINGS,
        # > For compatibility with pydantic 1.x BaseSettings you
        # > should use extra=ignore: [1]
        #
        # [1] https://docs.pydantic.dev/2.7/concepts/pydantic_settings/#dotenv-env-support  # noqa: E501
        extra="ignore",
    )

    allow_origins: list[str] | None = []
    """
    A list of origins that should be permitted to make
    cross-origin requests. E.g. ['https://example.org', 'https://www.example.org'].
    You can use ['*'] to allow any origin
    """

    allow_origin_regex: str | None = None
    """
    A regex string to match against origins that
    should be permitted to make cross-origin requests.
    e.g. 'https:\\/\\/.*\\.example\\.org'
    """

    allow_credentials: bool | None = False
    """Indicate that cookies should be supported for cross-origin requests"""

    allow_methods: list[str] | None = ["GET"]
    """A list of HTTP methods that should be allowed for cross-origin requests"""

    allow_headers: list[str] | None = []
    """A list of HTTP request headers that should be supported for
    cross-origin requests"""

    expose_headers: list[str] | None = []
    """Indicate any response headers that should be made accessible to the browser"""

    max_age: int | None = 600
    """Sets a maximum time in seconds for browsers to cache CORS responses"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        protected_namespaces=(),
        env_file=ENV_FILE_SETTINGS,
        env_prefix=ENV_PREFIX_SETTINGS,
        # > For compatibility with pydantic 1.x BaseSettings you
        # > should use extra=ignore: [1]
        #
        # [1] https://docs.pydantic.dev/2.7/concepts/pydantic_settings/#dotenv-env-support  # noqa: E501
        extra="ignore",
    )

    debug: bool = False
    """
    Enable debug mode.  When True, the application log level is set to DEBUG,
    the FastAPI inference and metrics servers run in debug mode, and
    health-check endpoints are included in access logs.
    Use ``access_log`` to control REST / gRPC request logging.

    If a ``logging_settings`` file/dict is also provided, it takes final
    precedence over the level set by ``debug``.
    """

    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    """When parallel inference is enabled, number of workers to run inference
    across."""

    parallel_inference_transport: Literal["queue", "zmq"] = "queue"
    """Inference IPC transport; ZeroMQ requires the optional ``zmq`` extra.

    Model-control messages retain multiprocessing queues in either mode.
    """

    parallel_workers_timeout: int = 5
    """Grace timeout to wait until the workers shut down when stopping MLServer."""

    environments_dir: str = DEFAULT_ENVIRONMENTS_DIR
    """
    Directory used to store custom environments.
    By default, the `.envs` folder of the current working directory will be
    used.
    """

    # Custom model repository class implementation
    model_repository_implementation: ImportString | None = None
    """*Python path* to the inference runtime to model repository (e.g.
    ``mlserver.repository.repository.SchemalessModelRepository``)."""

    # Model repository settings
    model_repository_root: str = "."
    """Root of the model repository, where we will search for models."""

    # Model Repository parameters are meant to be set directly by the MLServer runtime.
    model_repository_implementation_args: dict = {}
    """Extra parameters for model repository."""

    load_models_at_startup: bool = True
    """Flag to load all available models automatically at startup."""

    empty_registry_readiness: bool = True
    """
    Flag to control readiness check behavior when no models are loaded.
    If True (default), the server reports ready when the model registry is empty.
    If False, the server reports not ready when the model registry is empty.

    Note: This flag only applies AFTER server startup completes. During startup,
    an empty registry always reports not ready to prevent race conditions.
    """

    strict_readiness: bool = True
    """
    Flag to control readiness check behavior for loaded models.
    If True (default), ALL models must be ready for health check to pass.
    If False, AT LEAST ONE model must be ready for health check to pass.

    Note: During server startup, readiness is always strict (all models must be ready)
    regardless of this setting. This prevents race conditions during startup failures.
    After startup completes, this setting takes effect.
    """

    # Server metadata
    server_name: str = "mlserver"
    """Name of the server."""

    server_version: str = __version__
    """Version of the server."""

    extensions: list[str] = []
    """Server extensions loaded."""

    # HTTP Server settings
    host: str = "0.0.0.0"
    """Host where to listen for connections."""

    http_port: int = 8080
    """Port where to listen for HTTP / REST connections."""

    rest_worker_json: bool = False
    """Parse and encode unary REST inference JSON in parallel model workers.

    Nonparallel models, adaptive batching and other media types retain local
    request preparation. Generate and streaming routes are unchanged.
    """

    root_path: str = ""
    """Set the ASGI root_path for applications submounted below a given URL path."""

    grpc_port: int = 8081
    """Port where to listen for gRPC connections."""

    grpc_max_message_length: int | None = None
    """Maximum length (i.e. size) of gRPC payloads."""

    grpc_raw_input_buffers: bool = False
    """Preserve raw FP32 buffers for unary gRPC and NumPy-compatible runtimes.

    Enable only for runtimes whose input codecs accept NumPy-backed data.
    Streaming and unsupported input representations retain normal conversion.
    """

    # CORS settings
    cors_settings: CORSSettings | None = None

    # Metrics settings
    metrics_endpoint: str | None = "/metrics"
    """
    Endpoint used to expose Prometheus metrics. Alternatively, can be set to
    `None` to disable it.
    """

    metrics_port: int = 8082
    """
    Port used to expose metrics endpoint.
    """

    metrics_rest_server_prefix: str = "rest_server"
    """
    Metrics rest server string prefix to be exported.
    """

    metrics_dir: str = DEFAULT_METRICS_DIR
    """
    Directory used to share metrics across parallel workers.
    Equivalent to the `PROMETHEUS_MULTIPROC_DIR` env var in
    `prometheus-client`.
    Note that this won't be used if the `parallel_workers` flag is disabled.
    By default, the `.metrics` folder of the current working directory will be
    used.
    """

    # Logging settings
    log_level: str = "INFO"
    """
    Application log level.  Accepts any Python logging level name:
    ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    When ``debug`` is True this is overridden to ``DEBUG``.

    If a ``logging_settings`` file/dict is also provided, it takes final
    precedence and may override both ``log_level`` and ``debug``.
    """

    _VALID_LOG_LEVELS: ClassVar[frozenset[str]] = frozenset(
        {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        normalized = v.strip().upper()
        if normalized not in cls._VALID_LOG_LEVELS:
            warnings.warn(
                f"Unrecognised log_level {v!r}; defaulting to 'INFO'",
                UserWarning,
                stacklevel=2,
            )
            return "INFO"
        return normalized

    access_log: bool = True
    """
    Enable access logging for REST and gRPC endpoints.
    Independent of ``debug``. When enabled, logs each request except successful
    health-check GETs (/v2/health/live, /v2/health/ready, and
    /v2/models/\\*/ready), which are filtered unless ``debug`` is also True.
    Failed health-check responses are always logged.
    """

    use_structured_logging: bool = False
    """Use JSON-formatted structured logging instead of default format."""
    logging_settings: str | dict | None = None
    """Path to logging config file or dictionary configuration."""

    # OpenTelemetry Tracing settings
    tracing_server: str | None = None
    """Server name used to export OpenTelemetry tracing to collector service."""

    # Custom server settings
    _custom_rest_server_settings: dict | None = None
    _custom_metrics_server_settings: dict | None = None
    _custom_grpc_server_settings: dict | None = None

    gzip_enabled: bool = True
    """Enable GZipMiddleware."""

    @model_validator(mode="after")
    def validate_no_wildcard_cors_in_production_mode(self) -> Self:
        """
        Prevent wildcard CORS configurations in PRODUCTION mode.

        Wildcard CORS origins allow any website to make cross-origin requests,
        which is inappropriate for production deployments.
        """
        allowed = _load_image_baked_allowed_model_implementations(
            _get_trusted_runtimes_artifact_path()
        )

        # Only enforce in PRODUCTION mode (when allowlist file exists)
        if allowed is not None and self.cors_settings is not None:
            if "*" in (self.cors_settings.allow_origins or []):
                raise ValueError(
                    "Wildcard CORS origins ['*'] not allowed in PRODUCTION mode. "
                    "Specify explicit allowed origins or disable CORS settings."
                )
            if self.cors_settings.allow_origin_regex is not None:
                raise ValueError(
                    "CORS origin regex patterns not allowed in PRODUCTION mode. "
                    "Specify explicit allowed origins or disable CORS settings."
                )

        return self


class ModelParameters(BaseSettings):
    """
    Parameters that apply only to a particular instance of a model.
    This can include things like model weights, or arbitrary ``extra``
    parameters particular to the underlying inference runtime.
    The main difference with respect to ``ModelSettings`` is that parameters
    can change on each instance (e.g. each version) of the model.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_SETTINGS,
        env_prefix=ENV_PREFIX_MODEL_SETTINGS,
        extra="allow",
    )

    uri: str | None = None
    """
    URI where the model artifacts can be found.
    This path must be either absolute or relative to where MLServer is running.
    """

    version: str | None = None
    """Version of the model."""

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_empty_version(cls, v: str | None) -> str | None:
        if v == "":
            return None
        return v

    environment_path: str | None = None
    """Path to a directory that contains the python environment to be used
    to load this model."""

    environment_tarball: str | None = None
    """Path to the environment tarball which should be used to load this
    model."""

    inference_pool_gid: str | None = None
    """Inference pool group id to be used to serve this model."""

    autogenerate_inference_pool_gid: bool = False
    """Flag to autogenerate the inference pool group id for this model."""

    format: str | None = None
    """Format of the model (only available on certain runtimes)."""

    content_type: str | None = None
    """Default content type to use for requests and responses."""

    extra: dict | None = {}
    """Arbitrary settings, dependent on the inference runtime
    implementation."""

    @model_validator(mode="after")
    def set_inference_pool_gid(self) -> Self:
        if self.autogenerate_inference_pool_gid and self.inference_pool_gid is None:
            self.inference_pool_gid = str(uuid.uuid4())
        return self

    @model_validator(mode="after")
    def validate_no_custom_environments_in_production_mode(self) -> Self:
        """
        Prevent use of custom environments in PRODUCTION mode:
        - environment_tarball: Prevents unsafe tarball extraction
        - environment_path: Prevents sys.path injection attacks

        In PRODUCTION mode (when trusted runtimes allowlist exists), all dependencies
        must be pre-installed in the container image.
        """
        allowed = _load_image_baked_allowed_model_implementations(
            _get_trusted_runtimes_artifact_path()
        )

        # Only enforce in PRODUCTION mode (when allowlist file exists)
        if allowed is not None:
            if self.environment_tarball is not None:
                raise ValueError(
                    "environment_tarball is not allowed in PRODUCTION mode. "
                    "All dependencies must be pre-installed in the container image. "
                    "Remove the trusted runtimes allowlist file to use custom envs."
                )
            if self.environment_path is not None:
                raise ValueError(
                    "environment_path is not allowed in PRODUCTION mode. "
                    "All dependencies must be pre-installed in the container image. "
                    "Remove the trusted runtimes allowlist file to use custom envs."
                )

        return self


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_SETTINGS,
        env_prefix=ENV_PREFIX_MODEL_SETTINGS,
        # > For compatibility with pydantic 1.x BaseSettings you
        # > should use extra=ignore: [1]
        #
        # [1] https://docs.pydantic.dev/2.7/concepts/pydantic_settings/#dotenv-env-support  # noqa: E501
        extra="ignore",
    )

    # Source points to the file where model settings were loaded from
    _source: str | None = None

    def __init__(self, *args, **kwargs):
        # Ensure we still support inline init, e.g.
        # `ModelSettings(implementation=SumModel)`.
        implementation = kwargs.get("implementation", None)
        if inspect.isclass(implementation):
            kwargs["implementation"] = _get_import_path(implementation)

        super().__init__(*args, **kwargs)

    @classmethod
    def parse_file(cls, path: str) -> Self:  # type: ignore
        path = os.path.abspath(path)
        with open(path, "r") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                obj["_source"] = path

        try:
            return cls.model_validate(obj)
        except ValidationError as exc:
            # Repository settings may omit the name and derive it from the
            # containing directory. Retry only for that file-backed case;
            # environment-only settings must still provide a name.
            if isinstance(obj, dict) and _is_missing_or_empty_field_error(exc, "name"):
                model_directory = os.path.dirname(path)
                obj["name"] = os.path.basename(model_directory)
                return cls.model_validate(obj)
            raise

    @classmethod
    def model_validate(cls, obj: Any) -> Self:  # type: ignore
        source = None
        if isinstance(obj, dict):
            obj = obj.copy()
            source = obj.pop("_source", None)
        model_settings = super().model_validate(obj)
        if source:
            model_settings._source = source

        return model_settings

    @model_validator(mode="after")
    def validate_trusted_runtime(self) -> Self:
        # Step 1 (early validation): reject untrusted runtime import paths
        # while parsing settings so repository discovery can skip bad model
        # entries without taking down server startup.
        self.implementation_ = canonicalize_runtime_import_path(self.implementation_)
        _assert_trusted_runtime_import_path(self.implementation_)
        return self

    # Custom model class implementation
    #
    # NOTE: The `implementation_` attr will only point to the string import.
    #
    # The actual import will occur within the `implementation` property - think
    # of this as a lazy import.
    #
    # You should always use `model_settings.implementation` and treat
    # `implementation_` as a private attr.

    @property
    def implementation(self) -> type["MLModel"]:
        # Step 2 (defense in depth): validate again at access time in case
        # implementation_ was mutated programmatically after model validation.
        implementation = self.implementation_
        if not isinstance(implementation, str):
            raise ValueError("Model implementation has an invalid import path.")
        implementation = canonicalize_runtime_import_path(implementation)
        _assert_trusted_runtime_import_path(implementation)
        self.implementation_ = implementation

        # Check if we're in development mode (no allowlist file)
        allowed = _load_image_baked_allowed_model_implementations(
            _get_trusted_runtimes_artifact_path()
        )

        # In development mode, support dynamic loading from model folder
        if allowed is None and self._source:
            # Get a nice path to the model's (disk) location
            model_folder = os.path.dirname(self._source)

            # Temporarily inject the model's module into the Python system path
            logger.debug(
                "Development mode: attempting dynamic load of %s from %s",
                implementation,
                model_folder,
            )
            with _extra_sys_path(model_folder):
                _reload_module(implementation)
                return import_string(implementation)  # type: ignore

        # Production mode or no source file: use standard import path only
        return import_string(implementation)  # type: ignore

    @implementation.setter
    def implementation(self, value: type["MLModel"]):
        import_path = _get_import_path(value)
        import_path = canonicalize_runtime_import_path(import_path)
        _assert_trusted_runtime_import_path(import_path)
        self.implementation_ = import_path

    implementation_: str = Field(
        validation_alias=AliasChoices(
            "implementation", "MLSERVER_MODEL_IMPLEMENTATION"
        ),
        serialization_alias="implementation",
    )

    @property
    def version(self) -> str | None:
        params = self.parameters
        if params is not None:
            return params.version
        return None

    name: str
    """Name of the model."""

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model name must not be blank or whitespace-only")
        return value

    # Model metadata
    platform: str = ""
    """Framework used to train and serialise the model (e.g. sklearn)."""

    versions: list[str] = []
    """Versions of dependencies used to train the model (e.g.
    sklearn/0.20.1)."""

    inputs: list[MetadataTensor] = []
    """Metadata about the inputs accepted by the model."""

    outputs: list[MetadataTensor] = []
    """Metadata about the outputs returned by the model."""

    # Parallel settings
    parallel_workers: int | None = Field(
        None,
        deprecated=True,
        description=(
            "Use the `parallel_workers` field the server wide settings instead."
        ),
    )

    warm_workers: bool = Field(
        False,
        deprecated=True,
        description="Inference workers will now always be `warmed up` at start time.",
    )

    # Adaptive Batching settings (disabled by default)
    max_batch_size: int = 0
    """When adaptive batching is enabled, maximum number of requests to group
    together in a single batch."""

    max_batch_time: float = 0.0
    """When adaptive batching is enabled, maximum amount of time (in seconds)
    to wait for enough requests to build a full batch."""

    """*Python path* to the inference runtime to use to serve this model (e.g.
    ``mlserver_sklearn.SKLearnModel``)."""

    # Model parameters are meant to be set directly by the MLServer runtime.
    # However, it's also possible to override them manually.
    parameters: ModelParameters | None = None
    """Extra parameters for each instance of this model."""
