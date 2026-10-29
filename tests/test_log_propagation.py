"""
Unit tests for log-level propagation via _configure_framework_logger() hook.

Each test class is independently skipped when its runtime package is not
installed, so the suite works in core-only CI as well as full-runtimes CI.

Tests cover:
  - Base class hook contract (no-op, called in __init__, polymorphic dispatch)
  - Level-mapping dicts (XGBoost, ONNX)
  - _configure_framework_logger() side-effects for each runtime (sklearn,
    LightGBM, XGBoost, ONNX)
  - Deferred application in load() where required (ONNX)
  - Level captured once as ``self._mlserver_log_level`` on ``MLModel.__init__``
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlserver.model import MLModel
from mlserver.settings import ModelSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _set_mlserver_level(level: int) -> None:
    """Force the mlserver logger to a specific level for testing."""
    logging.getLogger("mlserver").setLevel(level)


def _make_settings(name: str = "test-model", **kwargs) -> ModelSettings:
    # Use a test-fixture implementation that passes the production-mode
    # trusted-runtimes allowlist (TEST_ONLY_EXTRA_IMPLEMENTATIONS).
    kwargs.setdefault("implementation", "tests.fixtures.SimpleModel")
    return ModelSettings(name=name, **kwargs)


@pytest.fixture(autouse=True)
def _restore_mlserver_level():
    """Reset the mlserver logger level after each test."""
    original = logging.getLogger("mlserver").level
    yield
    logging.getLogger("mlserver").setLevel(original)


# ---------------------------------------------------------------------------
# mlserver.logging.get_log_level
# ---------------------------------------------------------------------------


class TestGetLogLevel:
    def test_returns_int(self):
        from mlserver.logging import get_log_level

        assert isinstance(get_log_level(), int)

    @pytest.mark.parametrize(
        "level",
        [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
    )
    def test_reflects_logger_level(self, level: int):
        from mlserver.logging import get_log_level

        _set_mlserver_level(level)
        assert get_log_level() == level


# ---------------------------------------------------------------------------
# Base class hook contract
# ---------------------------------------------------------------------------


class TestConfigureFrameworkLogger:
    def test_base_is_noop(self):
        settings = _make_settings()
        model = MLModel(settings)
        assert model._configure_framework_logger() is None

    def test_hook_called_during_init(self):
        settings = _make_settings()
        with patch.object(MLModel, "_configure_framework_logger") as mock_hook:
            MLModel(settings)
            mock_hook.assert_called_once()

    def test_subclass_override_invoked(self):
        called = []

        class FakeRuntime(MLModel):
            def _configure_framework_logger(self) -> None:
                called.append(True)

        settings = _make_settings()
        FakeRuntime(settings)
        assert called == [True]

    def test_init_sets_log_level_from_mlserver(self):
        _set_mlserver_level(logging.ERROR)
        model = MLModel(_make_settings())
        assert model._mlserver_log_level == logging.ERROR


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_lightgbm"),
    reason="mlserver-lightgbm not installed",
)
class TestLGBMConfigureFrameworkLogger:
    def test_registers_mlserver_lightgbm_logger(self):
        from mlserver_lightgbm.lightgbm import LightGBMModel

        settings = _make_settings()
        with patch("mlserver_lightgbm.lightgbm.lgb") as mock_lgb:
            LightGBMModel(settings)
            mock_lgb.register_logger.assert_called_once()
            registered = mock_lgb.register_logger.call_args[0][0]
            assert registered.name == "mlserver.lightgbm"

    def test_registered_logger_inherits_mlserver_level(self):
        from mlserver_lightgbm.lightgbm import LightGBMModel

        _set_mlserver_level(logging.WARNING)
        with patch("mlserver_lightgbm.lightgbm.lgb") as mock_lgb:
            LightGBMModel(settings=_make_settings())
            lgb_logger = mock_lgb.register_logger.call_args[0][0]
        assert lgb_logger.level == logging.WARNING
        assert lgb_logger.getEffectiveLevel() == logging.WARNING

    def test_registered_logger_has_info_and_warning(self):
        from mlserver_lightgbm.lightgbm import LightGBMModel

        with patch("mlserver_lightgbm.lightgbm.lgb") as mock_lgb:
            LightGBMModel(settings=_make_settings())
            registered = mock_lgb.register_logger.call_args[0][0]
        for method in ("info", "warning"):
            assert callable(getattr(registered, method, None))


# ---------------------------------------------------------------------------
# XGBoost verbosity mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_xgboost"),
    reason="mlserver-xgboost not installed",
)
class TestXGBoostVerbosityMapping:
    @pytest.fixture(autouse=True)
    def _mapping(self):
        from mlserver_xgboost.xgboost import _XGB_VERBOSITY

        self.mapping = _XGB_VERBOSITY

    @pytest.fixture(autouse=True)
    def _reset_xgboost_logger(self):
        yield
        logging.getLogger("xgboost").setLevel(logging.NOTSET)

    @pytest.mark.parametrize(
        "level,expected",
        [
            (logging.DEBUG, 3),
            (logging.INFO, 2),
            (logging.WARNING, 1),
            (logging.ERROR, 0),
            (logging.CRITICAL, 0),
        ],
    )
    def test_mapping_values(self, level: int, expected: int):
        assert self.mapping[level] == expected

    def test_configure_calls_set_config(self):
        from mlserver_xgboost.xgboost import XGBoostModel

        with patch("mlserver_xgboost.xgboost.xgb") as mock_xgb:
            _set_mlserver_level(logging.WARNING)
            XGBoostModel(settings=_make_settings())
            mock_xgb.set_config.assert_called_once_with(verbosity=1)

    def test_configure_sets_python_xgboost_logger(self):
        from mlserver_xgboost.xgboost import XGBoostModel

        with patch("mlserver_xgboost.xgboost.xgb"):
            _set_mlserver_level(logging.ERROR)
            XGBoostModel(settings=_make_settings())
            assert logging.getLogger("xgboost").level == logging.ERROR


# ---------------------------------------------------------------------------
# ONNX Runtime severity mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxLogSeverityMapping:
    @pytest.fixture(autouse=True)
    def _mapping(self):
        from mlserver_onnx.utils import _ORT_LOG_SEVERITY

        self.mapping = _ORT_LOG_SEVERITY

    @pytest.mark.parametrize(
        "level,expected",
        [
            (logging.DEBUG, 0),
            (logging.INFO, 1),
            (logging.WARNING, 2),
            (logging.ERROR, 3),
            (logging.CRITICAL, 4),
        ],
    )
    def test_mapping_values(self, level: int, expected: int):
        assert self.mapping[level] == expected

    def test_unknown_level_falls_back_to_info(self):
        from mlserver_onnx.utils import _ORT_LOG_SEVERITY

        assert _ORT_LOG_SEVERITY.get(99, 1) == 1


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxBuildSessionOptions:
    """Tests for explicit per-model severity and user-override semantics."""

    def test_returns_session_options_when_settings_none(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options
        import onnxruntime as ort

        settings = OnnxSettings(session_options=None)
        result = _build_session_options(settings)
        assert isinstance(result, ort.SessionOptions)
        assert result.log_severity_level == ort.SessionOptions().log_severity_level

    def test_applies_explicit_debug_severity(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        settings = OnnxSettings(session_options=None)
        result = _build_session_options(settings, log_severity=0)
        assert result.log_severity_level == 0  # DEBUG → 0

    def test_applies_explicit_warning_severity(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        settings = OnnxSettings(session_options=None)
        result = _build_session_options(settings, log_severity=2)
        assert result.log_severity_level == 2  # WARNING → 2

    def test_respects_user_supplied_log_severity(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        settings = OnnxSettings(session_options={"log_severity_level": 4})
        result = _build_session_options(settings, log_severity=0)
        assert result.log_severity_level == 4  # user override wins

    def test_applies_severity_with_other_session_options(self):
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _build_session_options

        settings = OnnxSettings(session_options={"inter_op_num_threads": 2})
        result = _build_session_options(settings, log_severity=3)
        assert result.log_severity_level == 3  # ERROR → 3
        assert result.inter_op_num_threads == 2


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxApplySessionConfigEntries:
    """Tests for _apply_session_config_entries."""

    def test_noop_when_entries_none(self):
        import onnxruntime as ort
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _apply_session_config_entries

        options = ort.SessionOptions()
        settings = OnnxSettings(session_config_entries=None)
        result = _apply_session_config_entries(options, settings)
        assert result is options

    def test_applies_config_entries(self):
        import onnxruntime as ort
        from mlserver_onnx.settings import OnnxSettings
        from mlserver_onnx.utils import _apply_session_config_entries

        options = ort.SessionOptions()
        settings = OnnxSettings(
            session_config_entries={
                "session.use_env_allocators": "1",
            }
        )
        result = _apply_session_config_entries(options, settings)
        assert result is options
        assert result.get_session_config_entry("session.use_env_allocators") == "1"


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxConfigureFrameworkLogger:
    """Verify OnnxModel._configure_framework_logger() stores per-session severity."""

    def test_does_not_call_global_api(self):
        from mlserver_onnx.onnx import OnnxModel

        with patch("mlserver_onnx.onnx.ort") as mock_ort:
            settings = _make_settings()
            OnnxModel(settings)
            assert not mock_ort.set_default_logger_severity.called

    @pytest.mark.parametrize("override", [None, 4])
    async def test_load_uses_each_models_captured_severity(self, mocker, override):
        from mlserver.settings import ModelParameters
        from mlserver_onnx.onnx import OnnxModel

        mocker.patch(
            "mlserver_onnx.onnx.get_model_uri", AsyncMock(return_value="model.onnx")
        )
        mocker.patch("mlserver_onnx.onnx.onnx.load")
        mocker.patch(
            "mlserver_onnx.onnx._extract_metadata",
            return_value={"inputs": [], "outputs": []},
        )
        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.get_inputs.return_value = [SimpleNamespace(name="input")]
        session.get_outputs.return_value = [SimpleNamespace(name="output")]
        constructor = mocker.patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=session
        )
        extra = (
            {"session_options": {"log_severity_level": override}}
            if override is not None
            else {}
        )
        _set_mlserver_level(logging.DEBUG)
        first = OnnxModel(
            _make_settings("first", parameters=ModelParameters(extra=extra))
        )
        _set_mlserver_level(logging.ERROR)
        second = OnnxModel(
            _make_settings("second", parameters=ModelParameters(extra=extra))
        )
        # Changing the logger after construction must not change either model's
        # session severity when loading is deferred until later.
        _set_mlserver_level(logging.WARNING)
        assert await first.load()
        assert await second.load()
        severities = [
            call.kwargs["sess_options"].log_severity_level
            for call in constructor.call_args_list
        ]
        assert severities == ([override, override] if override is not None else [0, 3])


@pytest.mark.skipif(
    not _can_import("mlserver_onnx"),
    reason="mlserver-onnx not installed",
)
class TestOnnxProviderFallbackLogging:
    """Tests for provider mismatch warning in OnnxModel.load()."""

    @staticmethod
    def _make_mock_session(active_providers, input_names=None):
        if input_names is None:
            input_names = ["input_0"]
        mock_session = MagicMock()
        mock_session.get_providers.return_value = active_providers
        mock_inputs = []
        for name in input_names:
            inp = MagicMock()
            inp.name = name
            mock_inputs.append(inp)
        mock_session.get_inputs.return_value = mock_inputs
        out = MagicMock()
        out.name = "output_0"
        mock_session.get_outputs.return_value = [out]
        return mock_session

    @pytest.fixture()
    def _patch_onnx_load(self):
        """Patch all externals so OnnxModel.load() runs without a model file."""
        patches = [
            patch(
                "mlserver_onnx.onnx.get_model_uri",
                new_callable=AsyncMock,
                return_value="/fake/model.onnx",
            ),
            patch("mlserver_onnx.onnx.onnx.load", return_value=MagicMock()),
            patch(
                "mlserver_onnx.onnx._extract_metadata",
                return_value={"inputs": [], "outputs": []},
            ),
        ]
        mocks = [p.start() for p in patches]
        yield mocks
        for p in patches:
            p.stop()

    async def test_logs_warning_on_provider_mismatch(self, _patch_onnx_load, caplog):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings, ModelParameters

        mock_session = self._make_mock_session(
            active_providers=["CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
            parameters=ModelParameters(
                extra={"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
            ),
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.WARNING, logger="mlserver"):
                await model.load()

        assert "requested providers" in caplog.text
        assert "CUDAExecutionProvider" in caplog.text

    async def test_no_warning_when_providers_match(self, _patch_onnx_load, caplog):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings, ModelParameters

        mock_session = self._make_mock_session(
            active_providers=["CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
            parameters=ModelParameters(extra={"providers": ["CPUExecutionProvider"]}),
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.WARNING, logger="mlserver"):
                await model.load()

        assert "requested providers" not in caplog.text

    async def test_no_warning_when_ort_adds_cpu_fallback(
        self, _patch_onnx_load, caplog
    ):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings, ModelParameters

        mock_session = self._make_mock_session(
            active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
            parameters=ModelParameters(extra={"providers": ["CUDAExecutionProvider"]}),
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.WARNING, logger="mlserver"):
                await model.load()

        assert "requested providers" not in caplog.text

    async def test_logs_info_with_active_providers(self, _patch_onnx_load, caplog):
        from mlserver_onnx.onnx import OnnxModel
        from mlserver.settings import ModelSettings

        mock_session = self._make_mock_session(
            active_providers=["CPUExecutionProvider"]
        )
        settings = ModelSettings(
            name="test-model",
            implementation=OnnxModel,
        )
        model = OnnxModel(settings)

        with patch(
            "mlserver_onnx.onnx.ort.InferenceSession", return_value=mock_session
        ):
            with caplog.at_level(logging.INFO, logger="mlserver"):
                await model.load()

        assert "loaded with execution providers" in caplog.text
        assert "CPUExecutionProvider" in caplog.text


# ---------------------------------------------------------------------------
# sklearn
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _can_import("mlserver_sklearn"),
    reason="mlserver-sklearn not installed",
)
class TestSklearnConfigureFrameworkLogger:
    @pytest.fixture(autouse=True)
    def _reset_sklearn_loggers(self):
        yield
        logging.getLogger("sklearn").setLevel(logging.NOTSET)
        logging.getLogger("joblib").setLevel(logging.NOTSET)

    def test_sets_sklearn_and_joblib_levels(self):
        from mlserver_sklearn.sklearn import SKLearnModel

        _set_mlserver_level(logging.WARNING)
        SKLearnModel(settings=_make_settings())
        assert logging.getLogger("sklearn").level == logging.WARNING
        assert logging.getLogger("joblib").level == logging.WARNING
