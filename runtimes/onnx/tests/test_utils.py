import onnx
from onnx import TensorProto, helper
import pytest

from mlserver.errors import ModelValidationError
from mlserver.settings import ModelParameters, ModelSettings
from mlserver.types import Datatype
from mlserver_onnx.settings import OnnxSettings, get_onnx_settings
from mlserver_onnx.utils import (
    DEFAULT_EXECUTION_PROVIDERS,
    _extract_metadata,
    _get_providers,
    _onnx_elem_type_to_datatype,
    _onnx_shape_to_list,
    _value_info_to_metadata,
)


def test_get_onnx_settings_valid_dict():
    """get_onnx_settings returns parsed settings."""
    settings = ModelSettings(
        name="onnx-model",
        implementation="mlserver_onnx.OnnxModel",
        parameters=ModelParameters(extra={"providers": ["CPUExecutionProvider"]}),
    )

    onnx_settings = get_onnx_settings(settings)
    assert onnx_settings.providers == ["CPUExecutionProvider"]


def test_get_onnx_settings_invalid_type():
    """Invalid providers type raises an error."""
    settings = ModelSettings(
        name="onnx-model",
        implementation="mlserver_onnx.OnnxModel",
        parameters=ModelParameters(extra={"providers": "invalid"}),  # type: ignore
    )

    with pytest.raises(ModelValidationError):
        get_onnx_settings(settings)


def test_get_onnx_settings_model_settings_precedence(monkeypatch):
    """Model settings (parameters.extra) take precedence over environment."""
    monkeypatch.setenv("MLSERVER_MODEL_ONNX_PROVIDERS", '["CPUExecutionProvider"]')
    settings = ModelSettings(
        name="onnx-model",
        implementation="mlserver_onnx.OnnxModel",
        parameters=ModelParameters(extra={"providers": ["other"]}),
    )

    onnx_settings = get_onnx_settings(settings)
    assert onnx_settings.providers == ["other"]


def test_get_onnx_settings_parameters_none(monkeypatch):
    """When parameters is None, only env is used (extra is empty)."""
    monkeypatch.setenv("MLSERVER_MODEL_ONNX_PROVIDERS", '["CPUExecutionProvider"]')
    settings = ModelSettings(
        name="onnx-model",
        implementation="mlserver_onnx.OnnxModel",
        parameters=None,
    )

    onnx_settings = get_onnx_settings(settings)
    assert onnx_settings.providers == ["CPUExecutionProvider"]


def test_get_onnx_settings_extra_none_uses_env(monkeypatch):
    """When parameters.extra is None, env is merged (empty extra)."""
    monkeypatch.setenv("MLSERVER_MODEL_ONNX_PROVIDERS", '["CPUExecutionProvider"]')
    settings = ModelSettings(
        name="onnx-model",
        implementation="mlserver_onnx.OnnxModel",
        parameters=ModelParameters(extra=None),
    )

    onnx_settings = get_onnx_settings(settings)
    assert onnx_settings.providers == ["CPUExecutionProvider"]


def test_get_providers_default():
    """Default providers are returned when none set."""
    assert _get_providers(OnnxSettings()) == DEFAULT_EXECUTION_PROVIDERS


def test_onnx_elem_type_to_datatype_float():
    """Float tensor type maps to FP32."""
    assert _onnx_elem_type_to_datatype(TensorProto.FLOAT) == Datatype.FP32


def test_onnx_elem_type_to_datatype_unsupported():
    """Unsupported element type raises error."""
    with pytest.raises(ModelValidationError):
        _onnx_elem_type_to_datatype(9999)


def test_onnx_shape_to_list_dynamic_dim():
    """Dynamic dimensions convert to -1."""
    value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])

    assert _onnx_shape_to_list(value_info) == [-1, 4]


def test_value_info_to_metadata_missing_type():
    """ValueInfoProto with missing elem_type raises ModelValidationError."""
    value_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    value_info.type.tensor_type.elem_type = 0  # undefined

    with pytest.raises(ModelValidationError) as exc_info:
        _value_info_to_metadata(value_info)

    assert "missing type" in str(exc_info.value).lower()


def test_extract_metadata_ignores_initializers(tmp_path):
    """Initializers are excluded from input metadata."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    initializer = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [1, 4],
        [1.0, 1.0, 1.0, 1.0],
    )
    node = helper.make_node("Add", inputs=["input", "weights"], outputs=["output"])
    graph = helper.make_graph(
        [node], "init_model", [input_tensor], [output_tensor], [initializer]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=9
    )
    model_uri = tmp_path / "init-model.onnx"
    onnx.save(model, str(model_uri))

    metadata = _extract_metadata(str(model_uri))
    assert [tensor.name for tensor in metadata["inputs"]] == ["input"]
