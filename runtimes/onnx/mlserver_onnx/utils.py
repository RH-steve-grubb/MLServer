from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import mapping
import onnxruntime as ort

from mlserver.codecs.numpy import to_datatype
from mlserver.errors import ModelValidationError
from mlserver.types import Datatype, MetadataTensor

from .settings import OnnxSettings

PREDICT_OUTPUT = "predict"
VALID_OUTPUTS = [PREDICT_OUTPUT]

WELLKNOWN_MODEL_FILENAMES = ["model.onnx"]
DEFAULT_EXECUTION_PROVIDERS = ["CPUExecutionProvider"]
PROVIDERS_KEY = "providers"
PROVIDER_OPTIONS_KEY = "provider_options"
SESSION_OPTIONS_KEY = "session_options"
SESSION_CONFIG_ENTRIES_KEY = "session_config_entries"
RUN_OPTIONS_KEY = "run_options"


def _build_session_options(
    settings: OnnxSettings,
) -> Optional[ort.SessionOptions]:
    """
    Build SessionOptions from settings.

    Args:
        settings: Parsed ONNX settings.

    Returns:
        SessionOptions or None if not configured.

    Raises:
        ModelValidationError: If session_options is invalid or unsupported.
    """
    session_options = settings.session_options
    if session_options is None:
        return None

    if not isinstance(session_options, dict):
        raise ModelValidationError(
            "OnnxModel session_options must be a dict of SessionOptions fields"
        )

    options = ort.SessionOptions()
    for key, value in session_options.items():
        if not hasattr(options, key):
            raise ModelValidationError(
                f"OnnxModel session option '{key}' is not supported"
            )
        setattr(options, key, value)

    return options


def _apply_session_config_entries(
    options: Optional[ort.SessionOptions], settings: OnnxSettings
) -> Optional[ort.SessionOptions]:
    """
    Apply session_config_entries to SessionOptions.

    Args:
        options: Existing SessionOptions or None.
        settings: Parsed ONNX settings.

    Returns:
        SessionOptions with config entries applied.

    Raises:
        ModelValidationError: If session_config_entries is invalid.
    """
    entries = settings.session_config_entries
    if entries is None:
        return options

    if not isinstance(entries, dict):
        raise ModelValidationError(
            "OnnxModel session_config_entries must be a dict of string keys"
        )

    if options is None:
        options = ort.SessionOptions()

    for key, value in entries.items():
        if not isinstance(key, str):
            raise ModelValidationError(
                "OnnxModel session_config_entries keys must be strings"
            )
        options.add_session_config_entry(key, str(value))

    return options


def _build_run_options(settings: OnnxSettings) -> Optional[ort.RunOptions]:
    """
    Build RunOptions from settings.

    Args:
        settings: Parsed ONNX settings.

    Returns:
        RunOptions or None if not configured.

    Raises:
        ModelValidationError: If run_options is invalid or unsupported.
    """
    run_options = settings.run_options
    if run_options is None:
        return None

    if not isinstance(run_options, dict):
        raise ModelValidationError(
            "OnnxModel run_options must be a dict of RunOptions fields"
        )

    options = ort.RunOptions()
    for key, value in run_options.items():
        if not hasattr(options, key):
            raise ModelValidationError(f"OnnxModel run option '{key}' is not supported")
        setattr(options, key, value)

    return options


def _get_providers(settings: OnnxSettings) -> List[str]:
    """
    Resolve execution providers from settings.

    Args:
        settings: Parsed ONNX settings.

    Returns:
        Ordered list of provider names.

    Raises:
        ModelValidationError: If providers is invalid.
    """
    providers = settings.providers
    if providers is None:
        return DEFAULT_EXECUTION_PROVIDERS
    if not isinstance(providers, list) or not providers:
        raise ModelValidationError(
            "OnnxModel providers must be a non-empty list of strings"
        )
    if not all(isinstance(provider, str) for provider in providers):
        raise ModelValidationError(
            "OnnxModel providers must be a non-empty list of strings"
        )

    return providers


def _get_provider_options(
    settings: OnnxSettings, providers: Sequence[str]
) -> Optional[List[Dict[str, Any]]]:
    """
    Resolve provider_options aligned with the providers list.

    Args:
        settings: Parsed ONNX settings.
        providers: Ordered list of provider names.

    Returns:
        List of dicts (one per provider) or None.

    Raises:
        ModelValidationError: If provider_options is invalid or length mismatches.
    """
    provider_options = settings.provider_options
    if provider_options is None:
        return None

    if isinstance(provider_options, dict):
        if len(providers) != 1:
            raise ModelValidationError(
                "OnnxModel provider_options dict requires a single provider"
            )
        return [provider_options]

    if not isinstance(provider_options, list):
        raise ModelValidationError(
            "OnnxModel provider_options must be a dict or list of dicts"
        )

    if not provider_options or not all(
        isinstance(option, dict) for option in provider_options
    ):
        raise ModelValidationError("OnnxModel provider_options must be a list of dicts")

    if len(provider_options) != len(providers):
        raise ModelValidationError(
            "OnnxModel provider_options must match providers length"
        )

    return provider_options


def _onnx_elem_type_to_datatype(elem_type: int) -> Datatype:
    """
    Map ONNX tensor element type to MLServer Datatype.

    Args:
        elem_type: ONNX tensor element type id.

    Returns:
        The MLServer datatype.

    Raises:
        ModelValidationError: If the element type is unsupported.
    """
    np_type = mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type)
    if np_type is None:
        raise ModelValidationError(f"Unsupported ONNX tensor element type: {elem_type}")

    return to_datatype(np.dtype(np_type))


def _onnx_shape_to_list(value_info: onnx.ValueInfoProto) -> List[int]:
    """
    Convert ONNX tensor shape to a list of ints; dynamic dims become -1.

    Args:
        value_info: ONNX ValueInfoProto.

    Returns:
        Shape as list of sizes (-1 for dynamic).
    """
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value > 0 else -1)
    return dims


def _value_info_to_metadata(value_info: onnx.ValueInfoProto) -> MetadataTensor:
    """
    Convert ONNX ValueInfoProto to MetadataTensor.

    Args:
        value_info: ONNX ValueInfoProto.

    Returns:
        MetadataTensor with name, datatype, and shape.

    Raises:
        ModelValidationError: If type information is missing.
    """
    tensor_type = value_info.type.tensor_type
    if tensor_type is None or tensor_type.elem_type == 0:
        raise ModelValidationError(
            f"ONNX model tensor '{value_info.name}' missing type information"
        )

    return MetadataTensor(
        name=value_info.name,
        datatype=_onnx_elem_type_to_datatype(tensor_type.elem_type),
        shape=_onnx_shape_to_list(value_info),
    )


def _extract_metadata(model_uri: str) -> Dict[str, List[MetadataTensor]]:
    """
    Extract input and output metadata from the ONNX model file.

    Graph initializers (weights/constants) are excluded from inputs.

    Args:
        model_uri: Path to the ONNX model file.

    Returns:
        Dict with 'inputs' and 'outputs' lists of MetadataTensor.
    """
    model = onnx.load(model_uri)
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = [
        _value_info_to_metadata(value_info)
        for value_info in graph.input
        if value_info.name not in initializer_names
    ]
    outputs = [_value_info_to_metadata(value_info) for value_info in graph.output]

    return {"inputs": inputs, "outputs": outputs}
