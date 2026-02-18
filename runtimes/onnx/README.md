# ONNX runtime for MLServer

This package provides a MLServer runtime compatible with ONNX models using ONNX Runtime.

## Usage

You can install the runtime, alongside `mlserver`, as:

```bash
pip install mlserver mlserver-onnx
```

For further information on how to use MLServer with ONNX, you can check out
this [worked out example](../../docs/examples/onnx/README.md).

## Model Settings

To serve an ONNX model, configure MLServer with a `model-settings.json` that
points to the runtime and the model artifact. The runtime supports standard
`ModelSettings` fields, plus ONNX-specific configuration under
`parameters.extra`.

```{code-block} json
---
emphasize-lines: 3-8
---
{
  "name": "my-onnx-model",
  "implementation": "mlserver_onnx.OnnxModel",
  "parameters": {
    "uri": "./model.onnx",
    "version": "v1.0.0"
  }
}
```

## ONNX Model Format

The ONNX inference runtime will expect that your model is serialised as an ONNX
model file:

| Extension  | Docs                                                                                      | Example                   |
| ---------- | ----------------------------------------------------------------------------------------- | ------------------------- |
| `*.onnx`   | [ONNX Format](https://onnx.ai/onnx/intro/converters.html)                                | `model.save("model.onnx")`|

### Wellknown Model Filenames

By default, the runtime will look for a file called `model.onnx` in the model
directory. If your model uses a different filename, you can specify it using the
`parameters.uri` field.

```{note}
You can point `parameters.uri` to either:
- A specific model file: `"./my-model.onnx"`
- A directory containing `model.onnx`: `"./my-model-directory/"`

See the section on [Model Settings](../../docs/reference/model-settings.md) for more
details.
```


```{code-block} json
---
emphasize-lines: 3-5
---
{
  "name": "my-onnx-model",
  "parameters": {
    "uri": "./my-custom-model.onnx"
  }
}
```

## Content Types

If no [content type](../../docs/user-guide/content-type) is present on the
request or metadata, the ONNX runtime will try to decode the payload as a
[NumPy Array](../../docs/user-guide/content-type).
To avoid this, either send a different content type explicitly, or define the
correct one as part of your [model's
metadata](../../docs/reference/model-settings).

## Inputs

The ONNX runtime supports both single and multiple input tensors. Input tensors
from the inference request are mapped to the corresponding ONNX model inputs
**by name**. Each request input's `name` must match one of the model's graph
input names (as defined when the model was exported). Request input order does
not matter. You must provide all inputs that the model expects; missing inputs
will result in an error.

This matches the [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html): `InferenceSession.run()` expects an `input_feed` dictionary of `{input_name: input_value}`; the runtime builds this from your request using the model's graph input names from `session.get_inputs()`.

### Single Input Models

For models with a single input, use the same name in the request as in the ONNX
model (e.g. the name given in `input_names` at export time):

```{code-block} json
{
  "inputs": [
    {
      "name": "input",
      "datatype": "FP32",
      "shape": [1, 4],
      "data": [1.0, 2.0, 3.0, 4.0]
    }
  ]
}
```

### Multiple Input Models

For models with multiple inputs, provide all inputs in the request. Use the
exact input names from the ONNX model; order in the request does not matter:

```{code-block} json
{
  "inputs": [
    {
      "name": "input_ids",
      "datatype": "INT64",
      "shape": [1, 128],
      "data": [101, 2023, 2003, ...]
    },
    {
      "name": "attention_mask",
      "datatype": "INT64",
      "shape": [1, 128],
      "data": [1, 1, 1, ...]
    }
  ]
}
```

## Model Outputs

The ONNX inference runtime exposes model outputs based on the ONNX model's
defined output names.
These outputs can be accessed through the standard prediction interface.

| Output                 | Returned By Default | Availability                                                |
| ---------------------- | ------------------- | ----------------------------------------------------------- |
| `predict`              | ✅                  | Available on all ONNX models (returns first output).        |
| `<output_name>`        | ❌                  | Available when requested and matches a model output name.   |

By default, the runtime will only return the first output of the model using the
name `predict`. However, you are able to control which outputs you want back
through the `outputs` field of your {class}`InferenceRequest
<mlserver.types.InferenceRequest>` payload.

For example, to request a specific model output by name, you could define a
payload such as:

```{code-block} json
---
emphasize-lines: 10-12
---
{
  "inputs": [
    {
      "name": "input",
      "datatype": "FP32",
      "shape": [1, 3, 224, 224],
      "data": [...]
    }
  ],
  "outputs": [
    { "name": "output_layer_name" }
  ]
}
```

### Multi-Output Example

For models with multiple outputs, you can request specific outputs or all outputs:

```{code-block} json
{
  "inputs": [...],
  "outputs": [
    { "name": "logits" },
    { "name": "embeddings" }
  ]
}
```

If no outputs are specified, the runtime returns only the first output with the
name `predict`.

## Execution Providers

By default, the ONNX runtime uses the `CPUExecutionProvider`. You can override
this by providing execution provider settings under `parameters.extra` in your
model settings.

```{code-block} json
---
emphasize-lines: 6-12
---
{
  "name": "my-onnx-model",
  "implementation": "mlserver_onnx.OnnxModel",
  "parameters": {
    "uri": "./model.onnx",
    "extra": {
      "providers": ["CPUExecutionProvider"],
      "provider_options": [{}],
      "session_options": {
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1
      },
      "session_config_entries": {
        "session.set_denormal_as_zero": "1"
      },
      "run_options": {
        "log_severity_level": 3
      }
    }
  }
}
```

`providers` must be a non-empty list of provider names and `provider_options`
must be a dict (single provider) or a list of dicts aligned with the provider
list (passed to `InferenceSession` as a sequence of option dicts per the
[ONNX Runtime API](https://onnxruntime.ai/docs/api/python/api_summary.html)).
`session_options` maps directly to attributes on
`onnxruntime.SessionOptions`. `session_config_entries` provides additional
runtime keys via `SessionOptions.add_session_config_entry`. `run_options` maps
to `onnxruntime.RunOptions` and is applied on every `session.run()` call.

### Execution provider fields

- `providers`: ordered list of execution providers; the runtime attempts them in the order they appear in the list.
- `provider_options`: provider-specific settings aligned with `providers` (either
  a dict for a single provider, or a list of dicts).
- `session_options`: low-level runtime options such as
  `intra_op_num_threads`, `inter_op_num_threads`, and graph optimization flags.
- `session_config_entries`: string key/value entries added to the session config.
- `run_options`: per-run settings such as `log_severity_level` and `tag`.

## Performance Tuning

The ONNX runtime provides several configuration options to optimize inference
performance for your specific use case.

### Thread Configuration

Control the number of threads used for intra-operation and inter-operation
parallelism:

```{code-block} json
{
  "name": "my-onnx-model",
  "implementation": "mlserver_onnx.OnnxModel",
  "parameters": {
    "uri": "./model.onnx",
    "extra": {
      "session_options": {
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 2
      }
    }
  }
}
```

- `intra_op_num_threads`: Number of threads to parallelize operations within a
  single node (e.g., matrix multiplication). Higher values can improve
  performance for compute-intensive operations.
- `inter_op_num_threads`: Number of threads to parallelize execution across
  independent nodes in the graph. Useful for models with parallel branches.

### Graph Optimization

Enable graph optimizations to improve inference speed:

```{code-block} json
{
  "extra": {
    "session_options": {
      "graph_optimization_level": 99
    }
  }
}
```

Optimization levels:
- `0` (ORT_DISABLE_ALL): Disable all optimizations
- `1` (ORT_ENABLE_BASIC): Enable basic optimizations (constant folding, etc.)
- `2` (ORT_ENABLE_EXTENDED): Enable extended optimizations (default)
- `99` (ORT_ENABLE_ALL): Enable all possible optimizations

### Memory Optimization

For large models or memory-constrained environments:

```{code-block} json
{
  "extra": {
    "session_options": {
      "enable_mem_pattern": 0,
      "enable_cpu_mem_arena": 0
    }
  }
}
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

**Error**: `Failed to load ONNX model from '<path>': ...`

**Solutions**:
- Verify the model file exists and path is correct
- Ensure the ONNX model is valid: `python -m onnx.checker model.onnx`
- Check ONNX Runtime version compatibility with your model's opset version

#### Shape Mismatch Errors

**Error**: Codec errors related to tensor shapes

**Solutions**:
- Ensure input shapes in request match model's expected input shapes
- Check for dynamic dimensions (marked as -1 in model metadata)
- Verify data length matches shape: `product(shape) == len(data)`

#### Memory Issues

**Error**: Out of memory during inference

**Solutions**:
- Reduce batch size in input shapes
- Disable memory optimizations (see Memory Optimization section)
- Consider model quantization to reduce memory footprint

### Debugging

Enable detailed logging for troubleshooting:

```{code-block} json
{
  "extra": {
    "run_options": {
      "log_severity_level": 0
    },
    "session_options": {
      "log_severity_level": 0
    }
  }
}
```

Log severity levels:
- `0`: VERBOSE
- `1`: INFO
- `2`: WARNING
- `3`: ERROR
- `4`: FATAL

### Getting Help

If you encounter issues:
1. Check the [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
2. Verify your model with `onnx.checker`
3. Review MLServer logs for detailed error messages
4. Open an issue on [GitHub](https://github.com/opendatahub-io/MLServer)

