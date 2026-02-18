from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from mlserver.codecs import CodecError, NumpyCodec
from mlserver.errors import InferenceError, ModelLoadError
from mlserver.logging import logger
from mlserver.model import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
    RequestOutput,
)
from mlserver.settings import ModelSettings
from mlserver.utils import get_model_uri
from .settings import OnnxSettings, get_onnx_settings
from .utils import (
    PREDICT_OUTPUT,
    VALID_OUTPUTS,
    WELLKNOWN_MODEL_FILENAMES,
    _apply_session_config_entries,
    _build_run_options,
    _build_session_options,
    _extract_metadata,
    _get_provider_options,
    _get_providers,
)


class OnnxModel(MLModel):
    """
    MLModel implementation to load and serve ONNX models via ONNX Runtime.

    Supports single and multiple inputs; request tensors are mapped to model
    inputs by name.
    """

    _onnx_settings: Optional[OnnxSettings] = None

    def __init__(self, settings: ModelSettings):
        super().__init__(settings)
        self._onnx_settings = get_onnx_settings(self.settings)

    async def load(self) -> bool:
        """
        Load the ONNX model and initialize runtime metadata.

        Returns:
            True when the model is loaded and ready.

        Raises:
            ModelLoadError: If the model cannot be loaded or has no inputs.
        """
        if self._onnx_settings is None:
            raise ModelLoadError("ONNX settings not initialized")
        model_uri = await get_model_uri(
            self._settings, wellknown_filenames=WELLKNOWN_MODEL_FILENAMES
        )

        logger.debug(
            "ONNX settings for model '%s': source=%s, uri=%s, extra=%s, resolved=%s",
            self.name,
            getattr(self.settings, "_source", None),
            model_uri,
            getattr(self.settings.parameters, "extra", None),
            self._onnx_settings.model_dump(exclude_none=True),
        )

        providers = _get_providers(self._onnx_settings)
        provider_options = _get_provider_options(self._onnx_settings, providers)
        session_options = _build_session_options(self._onnx_settings)
        session_options = _apply_session_config_entries(
            session_options, self._onnx_settings
        )
        self._run_options = _build_run_options(self._onnx_settings)
        logger.debug(
            "ONNX runtime options for model '%s': providers=%s, provider_options=%s, "
            "session_options=%s, run_options=%s",
            self.name,
            providers,
            provider_options,
            session_options,
            self._run_options,
        )

        try:
            self._model = ort.InferenceSession(
                model_uri,
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load ONNX model from '{model_uri}': {exc}"
            ) from exc

        inputs = self._model.get_inputs()
        if not inputs:
            raise ModelLoadError("ONNX model has no inputs")

        self._input_names = [input_info.name for input_info in inputs]
        self._output_names = [output.name for output in self._model.get_outputs()]

        metadata = _extract_metadata(model_uri)
        self.inputs = metadata["inputs"]
        self.outputs = metadata["outputs"]

        return True

    def _check_request(self, payload: InferenceRequest) -> InferenceRequest:
        """
        Validate requested outputs and apply defaults.

        Args:
            payload: The inference request.

        Returns:
            The request with default output set if none was provided.

        Raises:
            InferenceError: If an unsupported output name is requested.
        """
        if not payload.outputs:
            payload.outputs = [RequestOutput(name=PREDICT_OUTPUT)]
        else:
            for request_output in payload.outputs:
                if (
                    request_output.name not in VALID_OUTPUTS
                    and request_output.name not in self._output_names
                ):
                    raise InferenceError(
                        f"OnnxModel only supports '{PREDICT_OUTPUT}' or "
                        f"model outputs ({request_output.name} was received)"
                    )

        return payload

    def _prepare_inputs(self, payload: InferenceRequest) -> Dict[str, np.ndarray]:
        """
        Decode request inputs and map them to ONNX model input names.

        Args:
            payload: The inference request.

        Returns:
            Dict mapping ONNX input names to numpy arrays.

        Raises:
            InferenceError: If inputs do not match model expectations.
            CodecError: If input decoding fails.
        """
        request_inputs_by_name = {inp.name: inp for inp in payload.inputs}
        missing = [n for n in self._input_names if n not in request_inputs_by_name]
        if missing:
            raise InferenceError(
                f"Model expects input(s) named {self._input_names} but received "
                f"{list(request_inputs_by_name)}. Missing: {missing}"
            )

        input_dict = {}
        for model_input_name in self._input_names:
            request_input = request_inputs_by_name[model_input_name]
            try:
                decoded = self.decode(request_input, default_codec=NumpyCodec)
                input_dict[model_input_name] = decoded
            except Exception as exc:
                raise CodecError(
                    f"Failed to decode input '{request_input.name}': {exc}"
                ) from exc

        return input_dict

    def _get_model_outputs(self, payload: InferenceRequest) -> List[ResponseOutput]:
        """
        Run ONNX inference and return encoded response outputs.

        Args:
            payload: The inference request.

        Returns:
            List of response outputs.

        Raises:
            InferenceError: If the model has no outputs or inference fails.
        """
        if not self._output_names:
            raise InferenceError("ONNX model has no outputs")

        input_dict = self._prepare_inputs(payload)
        requested_output_names: List[str] = []
        output_name_map = {}

        for request_output in payload.outputs:  # type: ignore
            if request_output.name == PREDICT_OUTPUT:
                output_name = self._output_names[0]
            else:
                output_name = request_output.name

            output_name_map[request_output.name] = output_name
            if output_name not in requested_output_names:
                requested_output_names.append(output_name)

        try:
            predictions = self._model.run(
                requested_output_names,
                input_dict,
                run_options=self._run_options,
            )
        except Exception as exc:
            raise InferenceError(f"ONNX inference failed: {exc}") from exc

        predictions_by_name = dict(zip(requested_output_names, predictions))
        outputs = []
        for request_output in payload.outputs:  # type: ignore
            y = predictions_by_name[output_name_map[request_output.name]]
            output = self.encode(y, request_output, default_codec=NumpyCodec)
            outputs.append(output)

        return outputs

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        Run inference and return the response.

        Args:
            payload: The inference request.

        Returns:
            The inference response with model outputs.
        """
        payload = self._check_request(payload)
        outputs = self._get_model_outputs(payload)

        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=outputs,
        )

    async def unload(self) -> bool:
        """Release the loaded model and clear runtime state."""
        if getattr(self, "_model", None):
            del self._model
        self._model = None
        self._onnx_settings = None
        self._input_names = []
        self._output_names = []
        self._run_options = None
        self.inputs = []
        self.outputs = []
        self.ready = False
        return True
