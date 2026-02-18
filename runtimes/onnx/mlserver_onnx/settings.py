from typing import Any, Dict, List, Optional, Union

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core import ValidationError

from mlserver.errors import ModelValidationError
from mlserver.settings import ModelSettings

ENV_PREFIX_ONNX_SETTINGS = "MLSERVER_MODEL_ONNX_"


class OnnxSettings(BaseSettings):
    """Settings for the ONNX runtime (parameters.extra / env)."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX_ONNX_SETTINGS,
        extra="ignore",
        protected_namespaces=(),
    )

    providers: Optional[List[str]] = None
    """Ordered list of execution providers; tried in this order."""

    provider_options: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    """Provider-specific options, aligned with the providers list."""

    session_options: Optional[Dict[str, Any]] = None
    """ONNX Runtime SessionOptions attributes applied at load."""

    session_config_entries: Optional[Dict[str, Any]] = None
    """Key/value entries applied via SessionOptions.add_session_config_entry."""

    run_options: Optional[Dict[str, Any]] = None
    """ONNX Runtime RunOptions attributes applied per inference."""


def _get_env_settings() -> Dict[str, Any]:
    """Read ONNX settings from the environment (MLSERVER_MODEL_ONNX_*)."""
    env_settings = OnnxSettings()
    return env_settings.model_dump(exclude_defaults=True, exclude_none=True)


def _merge_onnx_settings_extra(model_settings: ModelSettings) -> Dict[str, Any]:
    """
    Merge parameters.extra with env; file overrides env for conflicting keys.
    """
    extra_params = (
        model_settings.parameters.extra
        if model_settings.parameters is not None
        else None
    )
    if extra_params is None:
        extra_params = {}

    if not isinstance(extra_params, dict):
        raise ModelValidationError("OnnxModel parameters.extra must be a dict")

    env_params = _get_env_settings()
    settings_params = dict(env_params)
    settings_params.update(extra_params)
    return settings_params


def get_onnx_settings(model_settings: ModelSettings) -> OnnxSettings:
    """Parse ONNX settings from model parameters and environment."""
    extra = _merge_onnx_settings_extra(model_settings)
    try:
        return OnnxSettings(**extra)
    except ValidationError as exc:
        raise ModelValidationError(
            f"OnnxModel parameters.extra is invalid: {exc}"
        ) from exc
