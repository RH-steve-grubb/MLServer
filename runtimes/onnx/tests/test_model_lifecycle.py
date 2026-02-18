"""Tests for ONNX model lifecycle (load, unload, metadata)."""

import os
import pytest

from mlserver_onnx import OnnxModel
from mlserver_onnx.onnx import WELLKNOWN_MODEL_FILENAMES
from mlserver.settings import ModelSettings


def test_load(model: OnnxModel):
    """Test that model loads successfully."""
    assert model.ready
    assert model._model is not None
    assert model._input_names == ["input"]
    assert model._output_names == ["output"]


async def test_unload(model: OnnxModel):
    """Test that model unloads properly and cleans up resources."""
    assert model.ready
    assert await model.unload()
    assert model._model is None
    assert model._output_names == []
    assert model._input_names == []


@pytest.mark.parametrize("fname", WELLKNOWN_MODEL_FILENAMES)
async def test_load_folder(fname, model_uri: str, model_settings: ModelSettings):
    """Test loading model from folder with wellknown filenames."""
    model_folder = os.path.dirname(model_uri)
    model_path = os.path.join(model_folder, fname)
    os.rename(model_uri, model_path)

    model_settings.parameters.uri = model_folder  # type: ignore

    model = OnnxModel(model_settings)
    model.ready = await model.load()

    assert model.ready
    assert model._model is not None


async def test_multi_output_model(multi_output_model: OnnxModel):
    """Test that multi-output model loads correctly."""
    assert multi_output_model.ready
    assert len(multi_output_model._output_names) == 2
    assert "output1" in multi_output_model._output_names
    assert "output2" in multi_output_model._output_names
