"""Tests for ONNX runtime configuration (providers, session options, run options)."""

import pytest

from mlserver_onnx import OnnxModel
from mlserver.errors import ModelValidationError
from mlserver.settings import ModelSettings
from mlserver.types import InferenceRequest, RequestInput, Parameters, Datatype

# ========== Provider Configuration Tests ==========


async def test_load_with_providers(model_settings: ModelSettings):
    """Test loading model with custom execution providers."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider"]
    }
    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready
    assert "CPUExecutionProvider" in model._model.get_providers()


async def test_invalid_providers_setting(model_settings: ModelSettings):
    """Test that non-list provider setting raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": "CPUExecutionProvider"
    }

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_empty_providers_setting(model_settings: ModelSettings):
    """Test that empty provider list raises error."""
    model_settings.parameters.extra = {"providers": []}  # type: ignore

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_non_string_providers_setting(model_settings: ModelSettings):
    """Test that non-string provider in list raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider", 1]
    }

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


# ========== Provider Options Tests ==========


async def test_provider_options_dict_single_provider(model_settings: ModelSettings):
    """Provider options as dict for single provider (normalized to list for ORT)."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider"],
        "provider_options": {},
    }
    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready


async def test_provider_options_list_single_provider(model_settings: ModelSettings):
    """Test provider options as list of one dict for single provider (ORT API shape)."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider"],
        "provider_options": [{}],
    }
    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready


async def test_provider_options_dict_multiple_providers_error(
    model_settings: ModelSettings,
):
    """Test that dict provider options with multiple providers raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider", "CPUExecutionProvider"],
        "provider_options": {},
    }
    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_provider_options_list_invalid_length(model_settings: ModelSettings):
    """Test that mismatched provider options list length raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider", "CPUExecutionProvider"],
        "provider_options": [{}],
    }
    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_provider_options_list_invalid_type(model_settings: ModelSettings):
    """Test that invalid provider options type raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider"],
        "provider_options": "invalid",
    }
    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_provider_options_list_non_dict(model_settings: ModelSettings):
    """Test that non-dict elements in provider options list raise error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider"],
        "provider_options": ["invalid"],
    }
    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_provider_options_list_empty(model_settings: ModelSettings):
    """Test that empty provider options list raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "providers": ["CPUExecutionProvider"],
        "provider_options": [],
    }
    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


# ========== Session Options Tests ==========


async def test_invalid_session_options(model_settings: ModelSettings):
    """Test that invalid session option raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "session_options": {"unknown_option": 1}
    }

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_invalid_session_options_type(model_settings: ModelSettings):
    """Test that non-dict session options raise error."""
    model_settings.parameters.extra = {"session_options": "invalid"}  # type: ignore

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_valid_session_options(model_settings: ModelSettings):
    """Test loading with valid session options."""
    model_settings.parameters.extra = {  # type: ignore
        "session_options": {"intra_op_num_threads": 1}
    }

    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready


# ========== Session Config Entries Tests ==========


async def test_session_config_entries_invalid_type(model_settings: ModelSettings):
    """Test that non-dict session config entries raise error."""
    model_settings.parameters.extra = {  # type: ignore
        "session_config_entries": "invalid"
    }

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_session_config_entries_invalid_key(model_settings: ModelSettings):
    """Test that non-string keys in session config entries raise error."""
    model_settings.parameters.extra = {  # type: ignore
        "session_config_entries": {1: "value"}
    }

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_session_config_entries_valid(model_settings: ModelSettings):
    """Test loading with valid session config entries."""
    model_settings.parameters.extra = {  # type: ignore
        "session_config_entries": {"session.set_denormal_as_zero": "1"}
    }

    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready


# ========== Run Options Tests ==========


async def test_run_options_invalid_type(model_settings: ModelSettings):
    """Test that non-dict run options raise error."""
    model_settings.parameters.extra = {"run_options": "invalid"}  # type: ignore

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_run_options_invalid_option(model_settings: ModelSettings):
    """Test that invalid run option raises error."""
    model_settings.parameters.extra = {  # type: ignore
        "run_options": {"unknown_option": 1}
    }

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


async def test_run_options_valid(model_settings: ModelSettings):
    """Test loading with valid run options."""
    model_settings.parameters.extra = {  # type: ignore
        "run_options": {"log_severity_level": 3}
    }

    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready


# ========== Extra Parameter Tests ==========


async def test_extra_not_dict(model_settings: ModelSettings):
    """Test that non-dict extra parameter raises error."""
    model_settings.parameters.extra = "not-a-dict"  # type: ignore

    with pytest.raises(ModelValidationError):
        model = OnnxModel(model_settings)
        await model.load()


# ========== Content-type / MLModel.decode codec selection ==========


async def test_decode_uses_content_type_when_provided(model: OnnxModel):
    """self.decode uses content-type codec when input has parameters.content_type."""
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input",
                datatype=Datatype.FP32,
                shape=[2, 4],
                data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                parameters=Parameters(content_type="np"),
            )
        ]
    )
    response = await model.predict(request)
    assert len(response.outputs) == 1
    assert response.outputs[0].data is not None


async def test_decode_falls_back_to_default_codec_without_content_type(
    model: OnnxModel,
):
    """self.decode falls back to default_codec (NumpyCodec) when no content_type."""
    request = InferenceRequest(
        inputs=[
            RequestInput(
                name="input",
                datatype=Datatype.FP32,
                shape=[2, 4],
                data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
        ]
    )
    response = await model.predict(request)
    assert len(response.outputs) == 1
    assert response.outputs[0].data is not None
