import logging
import os
import signal
import sys
import pytest
import json
from unittest.mock import patch
from pydantic import ValidationError

from mlserver.settings import CORSSettings, Settings, ModelSettings, ModelParameters
from mlserver.repository import DEFAULT_MODEL_SETTINGS_FILENAME
import mlserver.settings as mlserver_settings

from .conftest import TESTDATA_PATH, TESTS_PATH


def test_model_settings_rejects_empty_name():
    with pytest.raises(ValidationError):
        ModelSettings(name="", implementation="tests.fixtures.SumModel")


def test_model_settings_rejects_whitespace_name():
    with pytest.raises(ValidationError):
        ModelSettings(name="   ", implementation="tests.fixtures.SumModel")


def test_model_settings_requires_name_when_loaded_from_environment(monkeypatch):
    monkeypatch.delenv("MLSERVER_MODEL_NAME", raising=False)
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "tests.fixtures.SumModel")

    with pytest.raises(ValidationError):
        ModelSettings()


def test_model_settings_parse_file_derives_name_from_directory(tmp_path):
    model_dir = tmp_path / "directory-model"
    model_dir.mkdir()
    settings_path = model_dir / "model-settings.json"
    settings_path.write_text(json.dumps({"implementation": "tests.fixtures.SumModel"}))

    model_settings = ModelSettings.parse_file(str(settings_path))

    assert model_settings.name == model_dir.name
    assert model_settings._source == str(settings_path)


def test_model_settings_parse_file_derives_name_for_empty_name(tmp_path, monkeypatch):
    model_dir = tmp_path / "directory-model"
    model_dir.mkdir()
    settings_path = model_dir / "model-settings.json"
    settings_path.write_text(
        json.dumps({"name": "", "implementation": "tests.fixtures.SumModel"})
    )
    monkeypatch.delenv("MLSERVER_MODEL_NAME", raising=False)

    model_settings = ModelSettings.parse_file(str(settings_path))

    assert model_settings.name == model_dir.name


def test_model_settings_parse_file_rejects_whitespace_environment_name(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "model-settings.json"
    settings_path.write_text(json.dumps({"implementation": "tests.fixtures.SumModel"}))
    monkeypatch.setenv("MLSERVER_MODEL_NAME", "   ")

    with pytest.raises(ValidationError):
        ModelSettings.parse_file(str(settings_path))


def test_model_settings_parse_file_empty_name_uses_directory_with_valid_env(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "directory-model"
    model_dir.mkdir()
    settings_path = model_dir / "model-settings.json"
    settings_path.write_text(
        json.dumps({"name": "", "implementation": "tests.fixtures.SumModel"})
    )
    monkeypatch.setenv("MLSERVER_MODEL_NAME", "environment-model")

    model_settings = ModelSettings.parse_file(str(settings_path))

    assert model_settings.name == model_dir.name


def test_model_settings_parse_file_derives_name_for_relative_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    settings_path = tmp_path / "model-settings.json"
    settings_path.write_text(json.dumps({"implementation": "tests.fixtures.SumModel"}))

    model_settings = ModelSettings.parse_file("model-settings.json")

    assert model_settings.name == tmp_path.name


@pytest.mark.parametrize("name", [0, False, None, "   "])
def test_model_settings_parse_file_rejects_invalid_name_instead_of_fallback(
    tmp_path, name
):
    model_dir = tmp_path / "directory-model"
    model_dir.mkdir()
    settings_path = model_dir / "model-settings.json"
    settings_path.write_text(
        json.dumps({"name": name, "implementation": "tests.fixtures.SumModel"})
    )

    with pytest.raises(ValidationError):
        ModelSettings.parse_file(str(settings_path))


def test_settings_from_env(monkeypatch):
    http_port = 5000
    monkeypatch.setenv("mlserver_http_port", str(http_port))

    settings = Settings()

    assert settings.http_port == http_port


def test_settings_debug_default_is_disabled(monkeypatch):
    monkeypatch.delenv("MLSERVER_DEBUG", raising=False)
    monkeypatch.delenv("mlserver_debug", raising=False)
    settings = Settings(_env_file=None)
    assert settings.debug is False


def test_settings_log_level_default_is_info(monkeypatch):
    monkeypatch.delenv("MLSERVER_LOG_LEVEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_settings_log_level_from_env(monkeypatch, level):
    monkeypatch.setenv("MLSERVER_LOG_LEVEL", level)
    settings = Settings(_env_file=None)
    assert settings.log_level == level


@pytest.mark.parametrize(
    "env_val, expected", [("warning", "WARNING"), ("error", "ERROR")]
)
def test_settings_log_level_from_env_case_insensitive(monkeypatch, env_val, expected):
    monkeypatch.setenv("MLSERVER_LOG_LEVEL", env_val)
    settings = Settings(_env_file=None)
    assert settings.log_level == expected


def test_settings_log_level_invalid_from_env_defaults_to_info(monkeypatch):
    monkeypatch.setenv("MLSERVER_LOG_LEVEL", "BOGUS")
    with pytest.warns(UserWarning, match="Unrecognised log_level"):
        settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"


@pytest.mark.parametrize(
    "env_val, expected",
    [(" DEBUG ", "DEBUG"), ("info\n", "INFO")],
)
def test_settings_log_level_strips_whitespace(monkeypatch, env_val, expected):
    monkeypatch.setenv("MLSERVER_LOG_LEVEL", env_val)
    settings = Settings(_env_file=None)
    assert settings.log_level == expected


def test_settings_access_log_default_is_enabled(monkeypatch):
    monkeypatch.delenv("MLSERVER_ACCESS_LOG", raising=False)
    settings = Settings(_env_file=None)
    assert settings.access_log is True


@pytest.mark.parametrize("val", ["false", "False", "0", "no"])
def test_settings_access_log_from_env(monkeypatch, val):
    monkeypatch.setenv("MLSERVER_ACCESS_LOG", val)
    settings = Settings(_env_file=None)
    assert settings.access_log is False


def test_settings_access_log_independent_of_debug(monkeypatch):
    monkeypatch.setenv("MLSERVER_ACCESS_LOG", "false")
    monkeypatch.setenv("MLSERVER_DEBUG", "true")
    settings = Settings(_env_file=None)
    assert settings.access_log is False
    assert settings.debug is True


@pytest.mark.parametrize(
    "access_log, debug, expect_access_log_in_config",
    [
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (False, True, False),
    ],
)
def test_rest_server_config_uses_access_log_setting(
    access_log: bool, debug: bool, expect_access_log_in_config: bool
):
    settings = Settings(access_log=access_log, debug=debug)

    from mlserver.rest.server import RESTServer
    from unittest.mock import MagicMock

    server = RESTServer.__new__(RESTServer)
    server._settings = settings
    server._app = MagicMock()

    config = server._get_config()
    assert config.access_log is expect_access_log_in_config


@pytest.mark.parametrize(
    "access_log, debug, expect_health_filter",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
@pytest.mark.asyncio
async def test_rest_health_log_filter_condition(
    access_log: bool, debug: bool, expect_health_filter: bool
):
    from mlserver.rest.server import RESTServer
    from unittest.mock import MagicMock, AsyncMock

    settings = Settings(access_log=access_log, debug=debug)
    server = RESTServer.__new__(RESTServer)
    server._settings = settings
    server._app = MagicMock()

    with patch(
        "mlserver.rest.server.disable_health_access_logs"
    ) as mock_disable, patch.object(
        server, "_get_config", return_value=MagicMock()
    ), patch(
        "mlserver.rest.server._NoSignalServer"
    ) as mock_server_cls:
        mock_instance = MagicMock()
        mock_instance.serve = AsyncMock()
        mock_server_cls.return_value = mock_instance
        await server.start()

    if expect_health_filter:
        mock_disable.assert_called_once()
    else:
        mock_disable.assert_not_called()


@pytest.mark.asyncio
async def test_grpc_interceptor_default_skips_health_methods(caplog):
    from mlserver.grpc.interceptors import LoggingInterceptor
    from unittest.mock import MagicMock, AsyncMock

    interceptor = LoggingInterceptor()
    mock_handler = MagicMock()
    mock_continuation = AsyncMock(return_value=mock_handler)

    health_details = MagicMock()
    health_details.method = "/inference.GRPCInferenceService/ServerLive"

    with caplog.at_level(logging.INFO, logger="mlserver.grpc"):
        await interceptor.intercept_service(mock_continuation, health_details)

    assert "/inference.GRPCInferenceService/ServerLive" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health_method",
    [
        "/inference.GRPCInferenceService/ServerLive",
        "/inference.GRPCInferenceService/ServerReady",
        "/inference.GRPCInferenceService/ModelReady",
    ],
)
async def test_grpc_interceptor_filters_health_methods(caplog, health_method):
    from mlserver.grpc.interceptors import LoggingInterceptor
    from unittest.mock import MagicMock, AsyncMock

    interceptor = LoggingInterceptor(skip_health=True)
    mock_handler = MagicMock()
    mock_continuation = AsyncMock(return_value=mock_handler)

    health_details = MagicMock()
    health_details.method = health_method

    with caplog.at_level(logging.INFO, logger="mlserver.grpc"):
        await interceptor.intercept_service(mock_continuation, health_details)

    assert health_method not in caplog.text


@pytest.mark.asyncio
async def test_grpc_interceptor_logs_non_health_methods(caplog):
    from mlserver.grpc.interceptors import LoggingInterceptor
    from unittest.mock import MagicMock, AsyncMock

    interceptor = LoggingInterceptor(skip_health=True)
    mock_handler = MagicMock()
    mock_continuation = AsyncMock(return_value=mock_handler)

    infer_details = MagicMock()
    infer_details.method = "/inference.GRPCInferenceService/ModelInfer"

    with caplog.at_level(logging.INFO, logger="mlserver.grpc"):
        await interceptor.intercept_service(mock_continuation, infer_details)

    assert "/inference.GRPCInferenceService/ModelInfer" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "health_method",
    [
        "/inference.GRPCInferenceService/ServerLive",
        "/inference.GRPCInferenceService/ServerReady",
        "/inference.GRPCInferenceService/ModelReady",
    ],
)
async def test_grpc_interceptor_logs_health_when_skip_false(caplog, health_method):
    from mlserver.grpc.interceptors import LoggingInterceptor
    from unittest.mock import MagicMock, AsyncMock

    interceptor = LoggingInterceptor(skip_health=False)
    mock_handler = MagicMock()
    mock_continuation = AsyncMock(return_value=mock_handler)

    health_details = MagicMock()
    health_details.method = health_method

    with caplog.at_level(logging.INFO, logger="mlserver.grpc"):
        await interceptor.intercept_service(mock_continuation, health_details)

    assert health_method in caplog.text


# ---------------------------------------------------------------------------
# REST HealthEndpointFilter unit tests
# ---------------------------------------------------------------------------


class TestHealthEndpointFilter:
    @staticmethod
    def _make_record(*args):
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, "", 0, "msg", (), None
        )
        record.args = args
        return record

    def test_suppresses_health_live(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record("127.0.0.1:8080", "GET", "/v2/health/live")
        assert f.filter(record) is False

    def test_suppresses_health_ready(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record("127.0.0.1:8080", "GET", "/v2/health/ready")
        assert f.filter(record) is False

    def test_suppresses_model_ready(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record("127.0.0.1:8080", "GET", "/v2/models/my-model/ready")
        assert f.filter(record) is False

    def test_suppresses_successful_model_ready_with_status(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(
            "127.0.0.1:8080", "GET", "/v2/models/my-model/ready", "1.1", 200
        )
        assert f.filter(record) is False

    def test_allows_failed_model_ready_with_status(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(
            "127.0.0.1:8080", "GET", "/v2/models/my-model/ready", "1.1", 503
        )
        assert f.filter(record) is True

    def test_allows_failed_health_ready_with_status(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(
            "127.0.0.1:8080", "GET", "/v2/health/ready", "1.1", 503
        )
        assert f.filter(record) is True

    def test_allows_infer_endpoint(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(
            "127.0.0.1:8080", "POST", "/v2/models/my-model/infer"
        )
        assert f.filter(record) is True

    def test_suppresses_versioned_model_ready(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(
            "127.0.0.1:8080", "GET", "/v2/models/my-model/versions/1/ready"
        )
        assert f.filter(record) is False

    def test_allows_post_to_health_path(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record("127.0.0.1:8080", "POST", "/v2/health/live")
        assert f.filter(record) is True

    def test_suppresses_bytes_health_path(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(b"127.0.0.1:8080", b"GET", b"/v2/health/live")
        assert f.filter(record) is False

    def test_suppresses_bytes_model_ready_path(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = self._make_record(
            b"127.0.0.1:8080", b"GET", b"/v2/models/my-model/ready"
        )
        assert f.filter(record) is False

    def test_allows_non_tuple_args(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, "", 0, "msg", (), None
        )
        record.args = "not a tuple"
        assert f.filter(record) is True

    def test_allows_short_tuple_args(self):
        from mlserver.rest.logging import HealthEndpointFilter

        f = HealthEndpointFilter()
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, "", 0, "msg", (), None
        )
        record.args = ("127.0.0.1:8080",)
        assert f.filter(record) is True


@pytest.fixture(autouse=False)
def _clean_uvicorn_filters():
    from mlserver.rest.logging import clear_uvicorn_access_log_filters

    clear_uvicorn_access_log_filters()
    yield
    clear_uvicorn_access_log_filters()


def test_disable_health_access_logs_is_idempotent(_clean_uvicorn_filters):
    from mlserver.rest.logging import (
        disable_health_access_logs,
        HealthEndpointFilter,
    )

    uvicorn_logger = logging.getLogger("uvicorn.access")
    disable_health_access_logs()
    disable_health_access_logs()
    count = sum(
        1 for f in uvicorn_logger.filters if isinstance(f, HealthEndpointFilter)
    )
    assert count == 1


def test_clear_uvicorn_access_log_filters(_clean_uvicorn_filters):
    from mlserver.rest.logging import (
        disable_health_access_logs,
        clear_uvicorn_access_log_filters,
        HealthEndpointFilter,
    )

    uvicorn_logger = logging.getLogger("uvicorn.access")
    disable_health_access_logs()
    assert any(isinstance(f, HealthEndpointFilter) for f in uvicorn_logger.filters)

    clear_uvicorn_access_log_filters()
    assert not any(isinstance(f, HealthEndpointFilter) for f in uvicorn_logger.filters)


@pytest.mark.asyncio
async def test_rest_server_stop_removes_health_filter(_clean_uvicorn_filters):
    from mlserver.rest.server import RESTServer
    from mlserver.rest.logging import (
        disable_health_access_logs,
        HealthEndpointFilter,
    )
    from unittest.mock import MagicMock

    uvicorn_logger = logging.getLogger("uvicorn.access")

    settings = Settings(access_log=True, debug=False)
    server = RESTServer.__new__(RESTServer)
    server._settings = settings
    server._server = MagicMock()

    disable_health_access_logs()
    assert any(isinstance(f, HealthEndpointFilter) for f in uvicorn_logger.filters)

    await server.stop()
    server._server.handle_exit.assert_called_once_with(sig=signal.SIGINT, frame=None)
    assert not any(isinstance(f, HealthEndpointFilter) for f in uvicorn_logger.filters)


# ---------------------------------------------------------------------------
# gRPC server init integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "access_log, debug, expect_interceptor, expect_skip_health",
    [
        (True, False, True, True),
        (True, True, True, False),
        (False, False, False, None),
        (False, True, False, None),
    ],
)
def test_grpc_create_server_interceptor_config(
    access_log, debug, expect_interceptor, expect_skip_health
):
    from mlserver.grpc.server import GRPCServer
    from mlserver.grpc.interceptors import LoggingInterceptor
    from unittest.mock import MagicMock

    settings = Settings(access_log=access_log, debug=debug, metrics_endpoint=None)
    data_plane = MagicMock()
    model_repo = MagicMock()

    server = GRPCServer(settings, data_plane, model_repo)

    with patch("mlserver.grpc.server.InferenceServicer"), patch(
        "mlserver.grpc.server.ModelRepositoryServicer"
    ), patch("mlserver.grpc.server.aio.server", return_value=MagicMock()), patch(
        "mlserver.grpc.server.add_GRPCInferenceServiceServicer_to_server"
    ), patch(
        "mlserver.grpc.server.add_ModelRepositoryServiceServicer_to_server"
    ):
        server._create_server()

    logging_interceptors = [
        i for i in server._interceptors if isinstance(i, LoggingInterceptor)
    ]

    if expect_interceptor:
        assert len(logging_interceptors) == 1
        assert logging_interceptors[0]._skip_health is expect_skip_health
    else:
        assert len(logging_interceptors) == 0


def test_settings_from_env_file(monkeypatch):
    env_file = f"{TESTDATA_PATH}/.test.env"

    settings = Settings(_env_file=env_file)
    cors_settings = CORSSettings(_env_file=env_file)
    model_settings = ModelSettings(_env_file=env_file)
    model_settings.parameters = ModelParameters(_env_file=env_file)

    assert settings.http_port == 9999
    assert settings.debug is True

    assert cors_settings.allow_origins == ["https://app.example.com"]
    assert cors_settings.max_age == 999

    assert model_settings.name == "dummy-name"
    assert model_settings.parameters.uri == "dummy-uri"


def test_model_settings_from_env(monkeypatch):
    model_name = "foo-model"
    model_version = "v0.1.0"
    model_uri = "/mnt/models/my-model"

    monkeypatch.setenv("mlserver_model_name", model_name)
    monkeypatch.setenv("mlserver_model_version", model_version)
    monkeypatch.setenv("mlserver_model_uri", model_uri)
    monkeypatch.setenv("mlserver_model_implementation", "tests.fixtures.SumModel")

    model_settings = ModelSettings()
    model_settings.parameters = ModelParameters()

    assert model_settings.name == model_name
    assert model_settings.parameters.version == model_version
    assert model_settings.parameters.uri == model_uri


@pytest.mark.parametrize(
    "obj",
    [
        ({"name": "foo", "implementation": "tests.fixtures.SumModel"}),
        (
            {
                "_source": os.path.join(TESTS_PATH, DEFAULT_MODEL_SETTINGS_FILENAME),
                "name": "foo",
                "implementation": "fixtures.SumModel",
            }
        ),
    ],
)
def test_model_settings_model_validate(obj: dict):
    pre_sys_path = sys.path[:]
    model_settings = ModelSettings.model_validate(obj)
    post_sys_path = sys.path[:]

    assert pre_sys_path == post_sys_path
    assert model_settings.implementation.__name__ == "SumModel"


def _build_model_settings(implementation=None) -> ModelSettings:
    payload = {
        "_source": os.path.join(TESTS_PATH, DEFAULT_MODEL_SETTINGS_FILENAME),
        "name": "foo",
    }
    if implementation is not None:
        payload["implementation"] = implementation
    source = payload.pop("_source")
    model_settings = ModelSettings(**payload)
    model_settings._source = source
    return model_settings


def _assert_implementation_resolves_to_mocked_runtime(
    model_settings: ModelSettings, expected_import_path: str, mocked_runtime_name: str
) -> None:
    mocked_runtime = type(mocked_runtime_name, (), {})
    with patch(
        "mlserver.settings.import_string", return_value=mocked_runtime
    ) as mock_import, patch("mlserver.settings._reload_module"):
        implementation = model_settings.implementation

    assert implementation is mocked_runtime
    mock_import.assert_called_once_with(expected_import_path)


def _clear_internal_test_runtime_overrides(_monkeypatch):
    _monkeypatch.setattr(
        mlserver_settings,
        "_get_trusted_runtimes_artifact_path",
        lambda: mlserver_settings.TRUSTED_RUNTIMES_ARTIFACT_PATH,
    )
    mlserver_settings.clear_trusted_runtime_caches()


def test_model_settings_allowlisted_implementation():
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_model_settings_builtin_runtime_class_is_canonicalized(request, mode):
    """Test builtin runtime class canonicalization.
    Should work for both PRODUCTION and DEVELOPMENT modes.

    Canonicalization of builtin aliases should work in BOTH modes.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    built_in_runtime = type(
        "SKLearnModel", (), {"__module__": "mlserver_sklearn.sklearn"}
    )
    model_settings = ModelSettings(name="foo", implementation=built_in_runtime)

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_model_settings_builtin_runtime_setter_is_canonicalized(request, mode):
    """Test builtin runtime setter canonicalization.
    Should work in both PRODUCTION and DEVELOPMENT modes.

    Canonicalization via setter should work in BOTH modes.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    built_in_runtime = type(
        "SKLearnModel", (), {"__module__": "mlserver_sklearn.sklearn"}
    )
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )

    model_settings.implementation = built_in_runtime

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_model_settings_builtin_submodule_import_path_is_canonicalized(request, mode):
    """Test builtin submodule import paths are canonicalized.
    Should work in both PRODUCTION and DEVELOPMENT modes.

    Canonicalization of submodule import paths should work in BOTH modes.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.sklearn.SKLearnModel"
    )

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_model_settings_access_time_canonicalizes_legacy_builtin_alias(request, mode):
    """Test access-time canonicalization of legacy aliases.
    Should work in both PRODUCTION and DEVELOPMENT modes.

    Legacy builtin alias canonicalization should work in BOTH modes.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    # Simulate direct attribute mutation after validation.
    model_settings.implementation_ = "mlserver_sklearn.sklearn.SKLearnModel"

    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )
    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


def test_model_settings_untrusted_implementation_rejected():
    with pytest.raises(ValueError, match="trusted runtimes allowlist"):
        _build_model_settings(implementation="malicious.CustomModel")


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_model_settings_invalid_import_path_rejected(request, mode):
    """Test invalid import paths are rejected.
    Should work in both PRODUCTION and DEVELOPMENT modes.

    Format validation (hyphens, lowercase class names, etc.) should work in BOTH modes.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    with pytest.raises(ValueError, match="invalid import path"):
        _build_model_settings(implementation="custom.Runtime-Model")

    with pytest.raises(ValueError, match="invalid import path"):
        _build_model_settings(implementation="custom.runtime")


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
@pytest.mark.parametrize("invalid_mutation", [[], 123])
def test_model_settings_access_time_invalid_mutation_rejected(
    request, mode, invalid_mutation
):
    """Test access-time mutation detection.
    Should work in both PRODUCTION and DEVELOPMENT modes.

    Defense-in-depth validation should work in BOTH modes.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    # Simulate direct mutation after validation (defense-in-depth check).
    model_settings.implementation_ = invalid_mutation  # type: ignore[assignment]

    with pytest.raises(ValueError, match="invalid import path"):
        _ = model_settings.implementation


def test_model_settings_untrusted_env_implementation_rejected(monkeypatch):
    monkeypatch.setenv("mlserver_model_name", "foo")
    monkeypatch.setenv("mlserver_model_implementation", "malicious.CustomModel")
    with pytest.raises(ValueError, match="trusted runtimes allowlist"):
        ModelSettings()


def test_model_settings_file_implementation_overrides_untrusted_env(monkeypatch):
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "malicious.CustomModel")
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_missing_file_implementation_falls_back_to_env_rejected(
    monkeypatch,
):
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "malicious.CustomModel")
    with pytest.raises(ValueError, match="trusted runtimes allowlist"):
        _build_model_settings()


def test_model_settings_missing_file_implementation_falls_back_to_allowlisted_env(
    monkeypatch,
):
    monkeypatch.setenv("MLSERVER_MODEL_IMPLEMENTATION", "mlserver_sklearn.SKLearnModel")
    model_settings = _build_model_settings()
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_empty_allowlist_rejected(empty_allowlist_mode):
    with pytest.raises(ValueError, match="trusted runtimes allowlist"):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_malformed_allowlist_entry_rejected(monkeypatch, tmp_path):
    # Whitespace-padded entries are treated as malformed and fail closed.
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('[" mlserver_sklearn.SKLearnModel "]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {" mlserver_sklearn.SKLearnModel "},
    )

    with pytest.raises(ValueError, match="invalid runtime import path"):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_image_baked_custom_runtime_allowed(monkeypatch, tmp_path):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.RuntimeModel"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {"mlserver_sklearn.SKLearnModel"},
    )

    model_settings = _build_model_settings(implementation="custom.RuntimeModel")
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="custom.RuntimeModel",
        mocked_runtime_name="MockedCustomRuntime",
    )


def test_model_settings_image_baked_builtin_alias_is_canonicalized(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text(
        '["mlserver_sklearn.sklearn.SKLearnModel"]', encoding="utf-8"
    )
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(mlserver_settings, "ALLOWED_MODEL_IMPLEMENTATIONS", set())
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )
    _assert_implementation_resolves_to_mocked_runtime(
        model_settings,
        expected_import_path="mlserver_sklearn.SKLearnModel",
        mocked_runtime_name="MockedSKLearnRuntime",
    )


def test_model_settings_invalid_trusted_runtime_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('{"runtime": "custom.RuntimeModel"}', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError, match="Trusted runtimes artifact must be a JSON list"
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_unparseable_trusted_runtime_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text("{invalid-json", encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError, match="Trusted runtimes artifact .* could not be loaded"
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_unreadable_trusted_runtime_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.RuntimeModel"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    real_open = open

    def _failing_open(path, *args, **kwargs):
        if path == str(artifact_path):
            raise OSError("permission denied")
        return real_open(path, *args, **kwargs)

    with patch("mlserver.settings.open", side_effect=_failing_open):
        with pytest.raises(
            ValueError, match="Trusted runtimes artifact .* could not be loaded"
        ):
            _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_invalid_runtime_import_path_in_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom-runtime"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError,
        match="Trusted runtimes artifact contains an invalid runtime import path",
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


@pytest.mark.parametrize(
    "invalid_runtime_path",
    [
        "RuntimeOnly",
        "custom.runtime",
        "_private.RuntimeModel",
        "custom._RuntimeModel",
        "custöm.RuntimeModel",
        "custom.Runtime-Model",
        "custom.runtime$Model",
    ],
)
def test_model_settings_unicode_or_special_runtime_in_artifact_rejected(
    monkeypatch, tmp_path, invalid_runtime_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text(
        json.dumps([invalid_runtime_path]),
        encoding="utf-8",
    )
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )

    with pytest.raises(
        ValueError,
        match="Trusted runtimes artifact contains an invalid runtime import path",
    ):
        _build_model_settings(implementation="mlserver_sklearn.SKLearnModel")


def test_model_settings_custom_runtime_not_in_image_artifact_rejected(
    monkeypatch, tmp_path
):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.AllowedRuntime"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {"mlserver_sklearn.SKLearnModel"},
    )

    with pytest.raises(ValueError, match="trusted runtimes allowlist"):
        _build_model_settings(implementation="custom.NotAllowedRuntime")


def test_model_settings_trusted_runtime_artifact_is_cached(monkeypatch, tmp_path):
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["custom.RuntimeModel"]', encoding="utf-8")
    _clear_internal_test_runtime_overrides(monkeypatch)
    monkeypatch.setattr(
        mlserver_settings, "TRUSTED_RUNTIMES_ARTIFACT_PATH", str(artifact_path)
    )
    monkeypatch.setattr(
        mlserver_settings,
        "ALLOWED_MODEL_IMPLEMENTATIONS",
        {"mlserver_sklearn.SKLearnModel"},
    )

    mocked_runtime = type("MockedCustomRuntime", (), {})
    with patch("mlserver.settings.open", wraps=open) as mock_open:
        model_settings = _build_model_settings(implementation="custom.RuntimeModel")
        with patch("mlserver.settings.import_string", return_value=mocked_runtime):
            # Access twice to confirm the trusted-runtimes artifact is read once.
            _ = model_settings.implementation
            _ = model_settings.implementation

    read_calls = [
        call for call in mock_open.call_args_list if call.args[0] == str(artifact_path)
    ]
    assert len(read_calls) == 1


def test_model_settings_serialisation():
    # Module may have been reloaded in a diff test, so let's re-import it
    from .fixtures import SumModel

    expected = "tests.fixtures.SumModel"
    model_settings = ModelSettings(name="foo", implementation=SumModel)

    assert model_settings.implementation == SumModel
    assert model_settings.implementation_ == expected

    # Dump `by_alias` to ensure that our alias overrides [1] are used
    # [2][3].
    #
    # > Whether to serialize using field aliases. [2][3]
    #
    # [1] https://github.com/jesse-c/MLServer/blob/4ac2da1d0dd7aa4b3796c047013b841fffa60e58/mlserver/settings.py#L373-L376  # noqa: E501
    # [2]  https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump  # noqa: E501
    # [3]  https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump_json  # noqa: E501

    as_dict = model_settings.model_dump(by_alias=True)
    as_dict["implementation"] == expected

    as_json = model_settings.model_dump_json(by_alias=True)
    as_dict = json.loads(as_json)
    as_dict["implementation"] == expected


@pytest.mark.parametrize(
    ("version", "expected_version"),
    [(None, None), ("", None), ("v1", "v1")],
)
def test_model_parameters_normalizes_empty_version(version, expected_version):
    parameters = ModelParameters(version=version)

    assert parameters.version == expected_version


def test_model_parameters_environment_tarball_blocked_in_production_mode():
    """Test that environment_tarball is rejected in PRODUCTION mode."""
    with pytest.raises(ValueError, match="environment_tarball is not allowed"):
        ModelParameters(environment_tarball="/path/to/env.tar.gz")


def test_model_parameters_environment_path_blocked_in_production_mode():
    """Test that environment_path is rejected in PRODUCTION mode."""
    with pytest.raises(ValueError, match="environment_path is not allowed"):
        ModelParameters(environment_path="/path/to/env")


def test_model_parameters_both_environments_blocked_in_production_mode():
    """Test that both environment parameters are rejected in PRODUCTION mode."""
    # Should fail on first parameter checked (environment_tarball)
    with pytest.raises(ValueError, match="environment_tarball is not allowed"):
        ModelParameters(
            environment_tarball="/path/to/env.tar.gz",
            environment_path="/path/to/env",
        )


def test_model_parameters_environment_tarball_allowed_in_development_mode(
    development_mode,
):
    """Test that environment_tarball is allowed in DEVELOPMENT mode."""
    # Should not raise
    params = ModelParameters(environment_tarball="/path/to/env.tar.gz")
    assert params.environment_tarball == "/path/to/env.tar.gz"


def test_model_parameters_environment_path_allowed_in_development_mode(
    development_mode,
):
    """Test that environment_path is allowed in DEVELOPMENT mode."""
    # Should not raise
    params = ModelParameters(environment_path="/path/to/env")
    assert params.environment_path == "/path/to/env"


def test_model_parameters_both_environments_allowed_in_development_mode(
    development_mode,
):
    """Test that both environment parameters are allowed in DEVELOPMENT mode."""
    # Should not raise
    params = ModelParameters(
        environment_tarball="/path/to/env.tar.gz",
        environment_path="/path/to/env",
    )
    assert params.environment_tarball == "/path/to/env.tar.gz"
    assert params.environment_path == "/path/to/env"


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_wildcard_cors_origins_blocked(request, mode):
    """Test that wildcard CORS origins are rejected."""
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    with pytest.raises(ValueError, match="Wildcard CORS origins"):
        Settings(cors_settings=CORSSettings(allow_origins=["*"]))


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_wildcard_cors_origins_in_list_blocked(request, mode):
    """Test that wildcard CORS origins are rejected.

    This should work even when mixed with explicit origins.
    """
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    with pytest.raises(ValueError, match="Wildcard CORS origins"):
        Settings(
            cors_settings=CORSSettings(allow_origins=["https://app.example.com", "*"])
        )


@pytest.mark.parametrize("mode", ["production_mode", "development_mode"])
def test_explicit_cors_origins_allowed(request, mode):
    """Test that explicit CORS origins are allowed."""
    if mode == "development_mode":
        request.getfixturevalue("development_mode")

    # Should not raise
    settings = Settings(
        cors_settings=CORSSettings(
            allow_origins=["https://app.example.com", "https://dashboard.example.com"]
        )
    )
    assert settings.cors_settings.allow_origins == [
        "https://app.example.com",
        "https://dashboard.example.com",
    ]


def test_model_settings_custom_runtime_allowed_in_development_mode(development_mode):
    """Test that custom runtimes are allowed in DEVELOPMENT mode.

    Custom runtimes (not in allowlist) should only be allowed in DEVELOPMENT mode.
    """
    # Custom runtime not in any allowlist
    model_settings = _build_model_settings(implementation="custom_pkg.CustomRuntime")

    # Should succeed in DEVELOPMENT mode
    assert model_settings.implementation_ == "custom_pkg.CustomRuntime"


def test_model_settings_builtin_runtime_allowed_in_development_mode(development_mode):
    """Test that builtin runtimes work in DEVELOPMENT mode.

    Builtin runtimes should work in BOTH modes.
    This test covers explicit DEVELOPMENT mode coverage.
    """
    model_settings = _build_model_settings(
        implementation="mlserver_sklearn.SKLearnModel"
    )

    assert model_settings.implementation_ == "mlserver_sklearn.SKLearnModel"


def test_dynamic_loading_actually_imports_from_model_folder(development_mode, tmp_path):
    """Test that dynamic loading actually imports runtime from model folder.

    This tests the _extra_sys_path() mechanism with real imports, not mocks.
    Requires loading from file (_source must be set) to trigger dynamic loading.
    """
    # Create model folder with custom runtime
    model_folder = tmp_path / "my_model"
    model_folder.mkdir()

    (model_folder / "custom_runtime.py").write_text(
        "from mlserver import MLModel\nclass MyRuntime(MLModel): pass\n",
        encoding="utf-8",
    )

    (model_folder / "model-settings.json").write_text(
        '{"name": "test-model", "implementation": "custom_runtime.MyRuntime"}',
        encoding="utf-8",
    )

    # Load from file (sets _source, required for dynamic loading)
    model_settings = ModelSettings.parse_file(str(model_folder / "model-settings.json"))

    # Trigger actual dynamic loading via .implementation property
    runtime_class = model_settings.implementation
    assert runtime_class.__name__ == "MyRuntime"
    assert runtime_class.__module__ == "custom_runtime"


def test_is_valid_runtime_import_path_valid():
    """Test is_valid_runtime_import_path() with valid import paths."""
    from mlserver.settings import is_valid_runtime_import_path

    # Valid cases
    assert is_valid_runtime_import_path("custom.MyRuntime")
    assert is_valid_runtime_import_path("acme.deep.nested.Runtime")
    assert is_valid_runtime_import_path("pkg.Runtime123")
    assert is_valid_runtime_import_path("mlserver_sklearn.SKLearnModel")


def test_is_valid_runtime_import_path_invalid():
    """Test is_valid_runtime_import_path() with invalid import paths."""
    from mlserver.settings import is_valid_runtime_import_path

    # Invalid cases
    assert not is_valid_runtime_import_path("RuntimeOnly")  # No module
    assert not is_valid_runtime_import_path("custom.runtime")  # Lowercase class
    assert not is_valid_runtime_import_path("_private.Runtime")  # Leading underscore
    assert not is_valid_runtime_import_path("custom.Runtime-")  # Special char
    assert not is_valid_runtime_import_path("custom.-Runtime")  # Special char
    assert not is_valid_runtime_import_path("custom..Runtime")  # Double dot
    assert not is_valid_runtime_import_path(".custom.Runtime")  # Leading dot
    assert not is_valid_runtime_import_path("custom.Runtime.")  # Trailing dot
    assert not is_valid_runtime_import_path("")  # Empty string


def test_is_valid_runtime_import_path_non_string():
    """Test is_valid_runtime_import_path() with non-string inputs."""
    from mlserver.settings import is_valid_runtime_import_path

    # Non-string inputs should return False
    assert not is_valid_runtime_import_path(None)
    assert not is_valid_runtime_import_path(123)
    assert not is_valid_runtime_import_path([])
    assert not is_valid_runtime_import_path({})
    assert not is_valid_runtime_import_path(("module", "Class"))


def test_canonicalize_runtime_import_path_builtins():
    """Test canonicalize_runtime_import_path() with builtin aliases."""
    from mlserver.settings import canonicalize_runtime_import_path

    # All builtin aliases should be canonicalized
    assert (
        canonicalize_runtime_import_path("mlserver_sklearn.sklearn.SKLearnModel")
        == "mlserver_sklearn.SKLearnModel"
    )
    assert (
        canonicalize_runtime_import_path("mlserver_xgboost.xgboost.XGBoostModel")
        == "mlserver_xgboost.XGBoostModel"
    )
    assert (
        canonicalize_runtime_import_path("mlserver_lightgbm.lightgbm.LightGBMModel")
        == "mlserver_lightgbm.LightGBMModel"
    )
    assert (
        canonicalize_runtime_import_path("mlserver_onnx.onnx.OnnxModel")
        == "mlserver_onnx.OnnxModel"
    )


def test_canonicalize_runtime_import_path_non_builtin():
    """Test canonicalize_runtime_import_path() with non-builtin paths."""
    from mlserver.settings import canonicalize_runtime_import_path

    # Non-builtin paths should pass through unchanged
    assert canonicalize_runtime_import_path("custom.Runtime") == "custom.Runtime"
    assert (
        canonicalize_runtime_import_path("acme.deep.nested.Runtime")
        == "acme.deep.nested.Runtime"
    )
    assert (
        canonicalize_runtime_import_path("mlserver_sklearn.CustomModel")
        == "mlserver_sklearn.CustomModel"
    )


def test_clear_trusted_runtime_caches_clears(monkeypatch, tmp_path):
    """Test that clear_trusted_runtime_caches() actually clears the cache."""
    from mlserver.settings import (
        clear_trusted_runtime_caches,
        _load_image_baked_allowed_model_implementations,
    )

    # Create initial artifact
    artifact_path = tmp_path / "trusted-runtimes.json"
    artifact_path.write_text('["runtime_v1.Model"]', encoding="utf-8")
    monkeypatch.setattr(
        mlserver_settings,
        "_get_trusted_runtimes_artifact_path",
        lambda: str(artifact_path),
    )

    # Load once to populate cache
    result1 = _load_image_baked_allowed_model_implementations(str(artifact_path))
    assert result1 == frozenset(["runtime_v1.Model"])

    # Modify artifact
    artifact_path.write_text('["runtime_v2.Model"]', encoding="utf-8")

    # Without clearing cache, should still get old value (cached)
    result2 = _load_image_baked_allowed_model_implementations(str(artifact_path))
    assert result2 == frozenset(["runtime_v1.Model"])  # Still cached

    # Clear cache
    clear_trusted_runtime_caches()

    # Now should get new value
    result3 = _load_image_baked_allowed_model_implementations(str(artifact_path))
    assert result3 == frozenset(["runtime_v2.Model"])  # Cache cleared!


def test_clear_trusted_runtime_caches_multiple_calls():
    """Test that clear_trusted_runtime_caches() can be called multiple times safely."""
    from mlserver.settings import clear_trusted_runtime_caches

    # Should not raise
    clear_trusted_runtime_caches()
    clear_trusted_runtime_caches()
    clear_trusted_runtime_caches()
