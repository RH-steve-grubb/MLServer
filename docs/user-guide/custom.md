# Custom Inference Runtimes

There may be cases where the [inference runtimes](./index) offered
out-of-the-box by MLServer may not be enough, or where you may need **extra
custom functionality** which is not included in MLServer (e.g. custom codecs).
To cover these cases, MLServer lets you create custom runtimes very easily.

This page covers some of the bigger points that need to be taken into account
when extending MLServer.
You can also see this [end-to-end example](../examples/custom/README) which
walks through the process of writing a custom runtime.

## Writing a custom inference runtime

MLServer is designed as an easy-to-extend framework, encouraging users to write
their own custom runtimes easily.
The starting point for this is the {class}`MLModel <mlserver.MLModel>`
abstract class, whose main methods are:

- {func}`load() <mlserver.MLModel.load>`:
  Responsible for loading any artifacts related to a model (e.g. model
  weights, pickle files, etc.).
- {func}`unload() <mlserver.MLModel.unload>`:
  Responsible for unloading the model, freeing any resources (e.g. GPU memory,
  etc.).
- {func}`predict() <mlserver.MLModel.predict>`:
  Responsible for using a model to perform inference on an incoming data point.

Therefore, the _"one-line version"_ of how to write a custom runtime is to
write a custom class extending from {class}`MLModel <mlserver.MLModel>`,
and then overriding those methods with your custom logic.

```{code-block} python
---
emphasize-lines: 7-8, 12-13
---
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse

class MyCustomRuntime(MLModel):

  async def load(self) -> bool:
    # TODO: Replace for custom logic to load a model artifact
    self._model = load_my_custom_model()
    return True

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    # TODO: Replace for custom logic to run inference
    return self._model.predict(payload)
```

### Simplified interface

MLServer exposes an alternative _"simplified" interface_ which can be used to
write custom runtimes.
This interface can be enabled by decorating your `predict()` method with the
`mlserver.codecs.decode_args` decorator.
This will let you specify in the method signature both how you want your
request payload to be decoded and how to encode the response back.

Based on the information provided in the method signature, MLServer will
automatically decode the request payload into the different inputs specified as
keyword arguments.
Under the hood, this is implemented through [MLServer's codecs and content types
system](./content-type.md).

```{note}
MLServer's _"simplified" interface_ aims to cover use cases where encoding /
decoding can be done through one of the codecs built-in into the MLServer
package.
However, there are instances where this may not be enough (e.g. variable number
of inputs, variable content types, etc.).
For these types of cases, please use MLServer's [_"advanced"
interface_](#writing-a-custom-inference-runtime), where you will have full
control over the full encoding / decoding process.
```

As an example of the above, let's assume a model which

- Takes two lists of strings as inputs:
  - `questions`, containing multiple questions to ask our model.
  - `context`, containing multiple contexts for each of the
    questions.
- Returns a Numpy array with some predictions as the output.

Leveraging MLServer's simplified notation, we can represent the above as the
following custom runtime:

```{code-block} python
---
emphasize-lines: 2-3, 12-13
---
from mlserver import MLModel
from mlserver.codecs import decode_args
from typing import List

class MyCustomRuntime(MLModel):

  async def load(self) -> bool:
    # TODO: Replace for custom logic to load a model artifact
    self._model = load_my_custom_model()
    return True

  @decode_args
  async def predict(self, questions: list[str], context: list[str]) -> np.ndarray:
    # TODO: Replace for custom logic to run inference
    return self._model.predict(questions, context)
```

Note that, the method signature of our `predict` method now specifies:

- The input names that we should be looking for in the request
  payload (i.e. `questions` and `context`).
- The expected content type for each of the request inputs (i.e. `list[str]` on
  both cases).
- The expected content type of the response outputs (i.e. `np.ndarray`).

### Read and write headers

```{note}
The `headers` field within the `parameters` section of the request / response
is managed by MLServer.
Therefore, incoming payloads where this field has been explicitly modified will
be overriden.
```

There are occasions where custom logic must be made conditional to extra
information sent by the client outside of the payload.
To allow for these use cases, MLServer will map all incoming HTTP headers (in
the case of REST) or metadata (in the case of gRPC) into the `headers` field of
the `parameters` object within the `InferenceRequest` instance.

```{code-block} python
---
emphasize-lines: 9-11
---
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse

class CustomHeadersRuntime(MLModel):

  ...

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    if payload.parameters and payload.parametes.headers:
      # These are all the incoming HTTP headers / gRPC metadata
      print(payload.parameters.headers)
    ...
```

Similarly, to return any HTTP headers (in the case of REST) or metadata (in the
case of gRPC), you can append any values to the `headers` field within the
`parameters` object of the returned `InferenceResponse` instance.

```{code-block} python
---
emphasize-lines: 13
---
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse

class CustomHeadersRuntime(MLModel):

  ...

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:
    ...
    return InferenceResponse(
      # Include any actual outputs from inference
      outputs=[],
      parameters=Parameters(headers={"foo": "bar"})
    )
```

## Loading a custom MLServer runtime

MLServer supports two modes for loading custom runtimes, depending on the
runtime security configuration:

### PRODUCTION Mode (Production)

When a trusted runtimes allowlist file exists (typically in production images),
MLServer operates in PRODUCTION mode. In this mode, custom runtimes must be
explicitly allowlisted and properly packaged into the image.

```{warning}
For production deployments (PRODUCTION mode), custom runtimes must be present in the
trusted allowlist artifact used by the running image. Runtime import paths not
in that allowlist are rejected at load time.
```

### DEVELOPMENT Mode (Development)

When no trusted runtimes allowlist file exists, MLServer operates in DEVELOPMENT
mode. This mode is designed for local development and provides maximum convenience
by supporting dynamic runtime loading directly from model folders.

```{note}
In DEVELOPMENT mode, you can simply place your custom runtime code (e.g.,
`my_runtime.py`) next to your `model-settings.json` file, and MLServer will
automatically discover and load it. No packaging or installation required!
```

```{warning}
DEVELOPMENT mode is intended for development and testing only. It allows
arbitrary code execution and should NEVER be used in production environments.
Always use PRODUCTION mode (production-mode images or Dockerfiles) for production deployments.
```

**Example for DEVELOPMENT mode:**

```bash
# Simple development workflow - no Docker build needed
models/
  └── my-model/
      ├── model-settings.json  # {"implementation": "my_runtime.MyModel"}
      ├── my_runtime.py        # Your custom runtime code
      └── model.pkl

# Start MLServer in development mode (no allowlist file)
mlserver start models/
# Automatically loads my_runtime.py from model folder!
```

For production deployments in PRODUCTION mode, continue reading below for the proper
packaging workflow.

### Verifying runtime allowlist configuration

You can verify which runtimes are allowed in your running MLServer instance by
querying the `/v2/runtimes` endpoint (REST) or `RuntimeSecurity` RPC (gRPC).
This is useful for debugging allowlist issues or confirming your custom runtime
was properly registered during image build.

```bash
curl http://localhost:8080/v2/runtimes
```

The response shows the security mode and, when in `PRODUCTION` mode, lists all
allowed model implementations. For more details and examples, see the
[model-settings reference](../reference/model-settings.md#querying-runtime-security-configuration).

### Packaging custom runtimes

The recommended workflow is to package your runtime code in the image and
declare each custom runtime with both `--allow-runtime` and a matching
`--runtime-path` during image build.

When running `mlserver build`, use an isolated and trusted build workspace.
Runtime-path validation catches misconfigurations early, but does not prevent
concurrent modifications. Ensure files are not modified during the build by
enforcing isolation via filesystem permissions or process controls.

For example, if we assume a flat model repository where each folder represents
a model, you would end up with a folder structure like the one below:

```bash
.
└── models
    └── sum-model
        ├── model-settings.json
        ├── models.py
```

Note that, from the example above, we are assuming that:

- Your custom runtime code lives in the `models.py` file and is packaged into
  the serving image.
- The `implementation` field of your `model-settings.json` configuration file
  contains the import path of your custom runtime (e.g.
  `models.MyCustomRuntime`).

  ```{code-block} json
  ---
  emphasize-lines: 3
  ---
  {
    "model": "sum-model",
    "implementation": "models.MyCustomRuntime"
  }
  ```

### Loading a custom Python environment

More often that not, your custom runtimes will depend on external 3rd party
dependencies which are not included within the main MLServer package.
In these cases, to load your custom runtime, MLServer will need access to these
dependencies.

It is possible to load this custom set of dependencies by providing them
through an environment tarball (`environment_tarball`) or by giving a
path to an already existing python environment (`environment_path`). Both paths can be
specified within your `model-settings.json` file.

```{warning}
**PRODUCTION Mode Restriction:**
The `environment_tarball` and `environment_path` parameters are **ONLY available 
in DEVELOPMENT mode**. In PRODUCTION mode (when a trusted runtimes allowlist exists), 
these parameters will be rejected with an error. 

For production deployments, all dependencies must be pre-installed in the container 
image during the build process using a [Conda environment file or requirements.txt](#custom-environment).
```

```{warning}
To load a custom environment, [parallel inference](./parallel-inference)
**must** be enabled.
```

```{warning}
The main MLServer process communicates with workers in custom environments via
[`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
using pickled objects. Custom environments therefore **must** use the same
version of MLServer and a compatible version of Python with the same [default
pickle protocol](https://docs.python.org/3/library/pickle.html#pickle.DEFAULT_PROTOCOL)
as the main process. Consult the tables below for environment compatibility.
```

| Status | Description  |
| ------ | ------------ |
| 🔴     | Unsupported  |
| 🟢     | Supported    |
| 🔵     | Untested     |

| Worker Python \ Server Python | 3.11 | 3.12 |
| ----------------------------- | ---- | ---- |
| 3.11                          | 🟢   | 🟢   |
| 3.12                          | 🟢   | 🟢   |

If we take the [previous example](#loading-a-custom-mlserver-runtime) above as
a reference, we could extend it to include our custom environment as:

```bash
.
└── models
    └── sum-model
        ├── environment.tar.gz
        ├── model-settings.json
        ├── models.py
```

Note that, in the folder layout above, we are assuming that:

- The `environment.tar.gz` tarball contains a pre-packaged version of your
  custom environment.
- The `environment_tarball` field of your `model-settings.json` configuration file
  points to your pre-packaged custom environment (i.e.
  `./environment.tar.gz`).

  ```{code-block} json
  ---
  emphasize-lines: 5
  ---
  {
    "model": "sum-model",
    "implementation": "models.MyCustomRuntime",
    "parameters": {
      "environment_tarball": "./environment.tar.gz"
    }
  }
  ```

If you want to use an already exisiting python environment, you can use the parameter `environment_path` of your `model-settings.json`:

```
---
emphasize-lines: 5
---
{
  "model": "sum-model",
  "implementation": "models.MyCustomRuntime",
  "parameters": {
    "environment_path": "~/micromambda/envs/my-conda-environment"
  }
}
```

## Building a custom MLServer image

```{note}
The `mlserver build` command expects that a Docker runtime is available and
running in the background.
```

MLServer offers built-in utilities to help you build a custom MLServer image.
This image can contain any custom code (including custom inference runtimes),
as well as any custom environment, provided either through a [Conda environment
file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
or a `requirements.txt` file.

### Build Modes: PRODUCTION vs DEVELOPMENT

You can build images in two different security modes:

**PRODUCTION Mode (Recommended for Production):**

Build an image with a trusted runtimes allowlist. Only explicitly declared
runtimes can be loaded at runtime.

```bash
# Build with specific custom runtimes allowlisted
mlserver build . -t my-custom-server \
  --allow-runtime models.MyCustomRuntime \
  --runtime-path models.py
```

- Custom runtimes are baked into the image at build time
- Runtime import paths are validated and allowlisted
- Provides strong security guarantees for production
- Default mode when using `--allow-runtime` / `--runtime-path`

**DEVELOPMENT Mode (Development/Testing Only):**

Build an image that allows any runtime to be loaded dynamically at runtime.

```bash
# Build a development image (no allowlist)
mlserver build . -t my-dev-server --dev
```

- No trusted runtimes allowlist is created
- Custom runtimes can be loaded from model folders at runtime
- Convenient for development and testing
- **WARNING:** Should NEVER be used in production environments

```{warning}
The `--dev` flag is mutually exclusive with `--allow-runtime` and
`--runtime-path`. You must choose either PRODUCTION or DEVELOPMENT mode, not both.
```

**Quick Reference:**

| Build Command | Mode | Custom Runtime Loading | Production Use |
|--------------|------|----------------------|----------------|
| `mlserver build . -t image` | PRODUCTION | Built-in runtimes only | ✅ Yes |
| `mlserver build . -t image --allow-runtime X --runtime-path x.py` | PRODUCTION | Built-in + allowlisted custom | ✅ Yes |
| `mlserver build . -t image --dev` | DEVELOPMENT | Any runtime (dynamic) | ❌ No |

### Building with Custom Runtimes (PRODUCTION Mode)

```{note}
When using custom runtimes, pass the exact dotted Python
import path for each runtime through `--allow-runtime` (for example
`--allow-runtime models.MyCustomRuntime`).
```

```{note}
In PRODUCTION mode, Python modules placed only in the model folder are not
auto-imported. Package custom runtime code into the built image and allowlist
it through `--allow-runtime`.

For each custom runtime module, also pass a matching `--runtime-path` value so
the source is copied into the image import path (for example,
`--runtime-path models.py` or `--runtime-path models/`).
When using a directory path, it must be an importable Python package containing
`__init__.py`.
```

```bash
mlserver build . -t my-custom-server \
  --allow-runtime models.MyCustomRuntime \
  --runtime-path models.py
```

```{note}
Migration tip: if you already have custom runtime images, make sure every
runtime in use is declared through `--allow-runtime module.ClassName` during
build. Runtime import paths not present in the trusted allowlist will be
rejected at load time.
```

The output will be a Docker image named `my-custom-server`, ready to be used.

### Custom Environment

The [`mlserver build`](../reference/cli) subcommand will search for any Conda
environment file (i.e. named either as `environment.yaml` or `conda.yaml`) and
/ or any `requirements.txt` present in your root folder.
These can be used to tell MLServer what Python environment is required in the
final Docker image.

```{note}
The environment built by the `mlserver build` will be global to the whole
MLServer image (i.e. every loaded model will, by default, use that custom
environment).
For Multi-Model Serving scenarios, it may be better to use [per-model custom
environments](#loading-a-custom-python-environment) instead - which will allow
you to run multiple custom environments at the same time.
```

### Default Settings

The `mlserver build` subcommand will treat any
[`settings.json`](../reference/settings) or
[`model-settings.json`](../reference/model-settings) files present on your root
folder as the default settings that must be set in your final image.
Therefore, these files can be used to configure things like the default
inference runtime to be used, or to even include **embedded models** that will
always be present within your custom image.

```{note}
Default setting values can still be overriden by external environment variables
or model-specific `model-settings.json`.
```

### Custom Dockerfile

Out-of-the-box, the `mlserver build` subcommand leverages a default
`Dockerfile` which takes into account a number of requirements, like

- Supporting arbitrary user IDs.
- Building your [base custom environment](#custom-environment) on the fly.
- Configure a set of [default setting values](#default-settings).

However, there may be occasions where you need to customise your `Dockerfile`
even further.
This may be the case, for example, when you need to provide extra environment
variables or when you need to customise your Docker build process (e.g. by
using other _"Docker-less"_ tools, like
[Kaniko](https://github.com/GoogleContainerTools/kaniko) or
[Buildah](https://buildah.io/)).

To account for these cases, MLServer also includes a [`mlserver
dockerfile`](../reference/cli) subcommand which will just generate a
`Dockerfile` (and optionally a `.dockerignore` file) exactly like the one used
by the `mlserver build` command.
This `Dockerfile` can then be customised according to your needs.

```{note}
The `mlserver dockerfile` command supports the same build modes as `mlserver build`:

- **PRODUCTION mode:** Use `--allow-runtime` and `--runtime-path` to generate a
  Dockerfile with a trusted runtimes allowlist
- **DEVELOPMENT mode:** Use `--dev` to generate a Dockerfile
  without an allowlist (development only)

For custom runtimes in PRODUCTION mode, pass `--allow-runtime module.ClassName`
and matching `--runtime-path` so the generated Dockerfile includes the
allowlist and corresponding `COPY` / `PYTHONPATH` entries.
```

````{note}
The base `Dockerfile` requires [Docker's
Buildkit](https://docs.docker.com/build/buildkit/) to be enabled.
To ensure BuildKit is used, you can use the `DOCKER_BUILDKIT=1` environment
variable, e.g.

```bash
DOCKER_BUILDKIT=1 docker build . -t my-custom-runtime:0.1.0
```
````
