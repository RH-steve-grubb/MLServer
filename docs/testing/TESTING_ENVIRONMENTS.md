# Testing with Venv and Conda Environments

This guide provides an overview of how MLServer tests support both venv and conda environments, how to configure tests for each scenario, and the underlying mechanisms.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variable: USE_CONDA](#environment-variable-use_conda)
  - [Tox Environments](#tox-environments)
- [How It Works](#how-it-works)
  - [With Venv](#with-venv)
  - [With Conda](#with-conda)
- [Python Version Testing](#python-version-testing)

---

## Overview

MLServer's test suite supports testing custom Python environments in **two modes**:

1. **Venv mode** (default): Uses Python `venv` and `pip` to create environment tarballs
2. **Conda mode**: Uses `conda` and `conda-pack` to create environment tarballs

Both approaches create compatible tarballs that work with MLServer's `Environment` class, which is **environment-manager agnostic**.

### Key Features

- **Flexible environment creation**: Venv or conda, your choice
- **Multiple Python versions**: Test across Python 3.10-3.12
- **Efficient caching**: Tarballs cached in `tests/testdata/.cache/`
- **Compatible outputs**: Both methods produce identical results
- **No code changes needed**: Same tests work in both modes

---

## Quick Start

### With Venv (default)

```bash
# Core tests
poetry run tox -e mlserver-venv

# All runtimes
poetry run tox -e all-runtimes-venv
```

### With ZeroMQ

The optional transport has dedicated environments that install the `zmq` extra
inside tox and fail immediately if it cannot be imported:

```bash
poetry run tox -e mlserver-venv-zmq
poetry run tox -e all-runtimes-venv-zmq
```

These run the same suites, with the same parallel/serial ordering, as their
default counterparts. PR CI runs both venv variants across Python 3.10–3.12;
combined-runtime conda tests remain push-only. The final **Test results** job
fails if any required job fails, is cancelled, or is skipped.

### With Conda

```bash
# Core tests (requires conda installed)
poetry run tox -e mlserver-conda

# All runtimes (requires conda installed)
poetry run tox -e all-runtimes-conda
```

### Manual Invocation

When running pytest directly (without tox), set the required environment variables and activate the virtual environment first:

```bash
# Required by tests/cli (environment.yml template rendering)
export GITHUB_SERVER_URL="${GITHUB_SERVER_URL:-https://github.com}"
export GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-opendatahub-io/MLServer}"
export GITHUB_REF="${GITHUB_REF:-refs/heads/master}"

```

Then run the tests in two steps. Some test directories (`metrics`, `parallel`, `grpc`, `env`, `cli`) are flaky when running in parallel, so they run sequentially in a separate step.

**Core tests:**

```bash
# Install dependencies and the project itself, then activate
poetry sync
source .venv/bin/activate

# Step 1: Run most tests in parallel
USE_CONDA=false python -m pytest -n auto tests/ \
    --ignore=tests/metrics --ignore=tests/parallel \
    --ignore=tests/grpc --ignore=tests/env --ignore=tests/cli

# Step 2: Run flaky-in-parallel tests sequentially
USE_CONDA=false python -m pytest \
    tests/metrics tests/parallel tests/grpc tests/env tests/cli
```

**All runtimes:**

```bash
# Install dependencies (including all runtimes) and the project itself, then activate
poetry sync --with all-runtimes --with all-runtimes-dev
source .venv/bin/activate

# Step 1: Run most tests in parallel
USE_CONDA=false python -m pytest -n auto tests/ \
    runtimes/sklearn/ runtimes/xgboost/ runtimes/lightgbm/ \
    runtimes/onnx/ \
    --ignore=tests/metrics --ignore=tests/parallel \
    --ignore=tests/grpc --ignore=tests/env --ignore=tests/cli

# Step 2: Run flaky-in-parallel tests sequentially
USE_CONDA=false python -m pytest \
    tests/metrics tests/parallel tests/grpc tests/env tests/cli
```

> **Note:** Replace `USE_CONDA=false` with `USE_CONDA=true` to use conda mode instead of venv.

---

## Configuration

### Environment Variable: USE_CONDA

The `USE_CONDA` environment variable controls which environment creation method to use:

| Value | Venv Mode | Conda Mode | Notes |
|-------|-----------|-----------|-------|
| (unset) | ✅ | ❌ | Default: venv mode |
| `false`, `0`, `no` | ✅ | ❌ | Tests current Python version only |
| `true`, `1`, `yes` | ❌ | ✅ | Tests all Python versions (3.10-3.12) |

**Implementation**: See [tests/conftest.py:47-52](../../tests/conftest.py#L47-L52)

```python
def get_python_versions() -> list[tuple[int, int]]:
    use_conda = os.environ.get("USE_CONDA", "").lower() in ("1", "true", "yes")
    if use_conda:
        return PYTHON_VERSIONS  # [(3,10), (3,11), (3,12)]

    return [(sys.version_info.major, sys.version_info.minor)]  # Current only
```

### Tox Environments

The four base environments select core/combined suites and venv/conda behavior:

| Feature | `mlserver-venv` | `mlserver-conda` | `all-runtimes-venv` | `all-runtimes-conda` |
|---------|-----------------|-------------------|---------------------|----------------------|
| Conda usage | Disabled | Enabled | Disabled | Enabled |
| Python versions | Current only | All (3.10-3.12) | Current only | All (3.10-3.12) |
| Runtimes tested | Core only | Core only | All | All |
| Requires | venv, pip (built-in) | conda, conda-pack | venv, pip (built-in) | conda, conda-pack |
| Speed | Faster (pip install) | Slower (conda install) | Faster (pip install) | Slower (conda install) |
| Tarball source | environment.txt | environment.yml | environment.txt | environment.yml |
| Environment variable | `USE_CONDA=false` | `USE_CONDA=true` | `USE_CONDA=false` | `USE_CONDA=true` |
| Tox command | `poetry run tox -e mlserver-venv` | `poetry run tox -e mlserver-conda` | `poetry run tox -e all-runtimes-venv` | `poetry run tox -e all-runtimes-conda` |

### Dependency Groups

The project defines separate Poetry dependency groups for runtime packages:

| Group | Runtimes | Purpose |
|-------|----------|---------|
| `odh-runtimes` | sklearn, xgboost, lightgbm, onnx | ODH-shipped runtimes used for production builds and constraints |
| `odh-runtimes-cuda` | onnx (cuda extra) | CUDA-accelerated ONNX runtime. Requires GPU hardware for CUDA tests. Run via `make test-cuda` or `tox -c ./runtimes/onnx -e cuda`. CPU tests run automatically in CI via `tox -c ./runtimes/onnx`. |
| `odh-runtimes-cuda-dev` | nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12, etc. | NVIDIA CUDA shared libraries installed via pip for dev/test on bare metal (no system CUDA toolkit required). Included automatically by `make install-dev-odh-cuda`. |
| `all-runtimes` | All of the above (sklearn, xgboost, lightgbm, onnx) | Full set used for testing |

---

## How It Works

### With Venv

When `USE_CONDA=false` (venv mode, default):

1. **Test fixture reads**: [tests/testdata/environment.txt](../../tests/testdata/environment.txt)
   ```
   scikit-learn==1.6.1
   ../../.
   ```

   Note: `../../.` installs the local MLServer package

2. **Environment creation**: For each Python version:
   ```bash
   python3.10 -m venv --copies /tmp/mlserver-<uuid>
   ```

3. **Dependency installation**:
   ```bash
   /tmp/mlserver-<uuid>/bin/pip install --upgrade pip
   /tmp/mlserver-<uuid>/bin/pip install -r tests/testdata/environment.txt
   ```

4. **Tarball packaging**: Creates `.tar.gz` with identical structure to conda-pack:
   ```python
   import tarfile
   with tarfile.open(tarball_path, "w:gz") as tar:
       tar.add(venv_path, arcname=".")
   ```

5. **Test execution**: Same tests, same assertions, same results

6. **Single-version testing**: Only tests against current Python interpreter:
   - If running on Python 3.11: `test_from_tarball[py311]`
   - Avoids requiring multiple Python versions installed

### With Conda

When `USE_CONDA=true` (or `1`, `yes`):

1. **Test fixture reads**: [tests/testdata/environment.yml](../../tests/testdata/environment.yml)
   ```yaml
   name: custom-runtime-environment
   channels:
     - conda-forge
   dependencies:
     - python == 3.12
     - scikit-learn == 1.6.1
     - pip:
         - mlserver @ git+${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}.git@${GITHUB_REF}
   ```

2. **Python version injection**: The test infrastructure dynamically modifies the environment.yml to test multiple Python versions:
   - Reads the template environment.yml
   - Replaces `python == 3.12` with requested version (e.g., `python == 3.10`)
   - Saves as `environment-py310.yml`, `environment-py311.yml`, etc.

3. **Environment creation**: For each Python version:
   ```bash
   conda env create -n mlserver-<uuid> -f environment-py310.yml
   ```

4. **Tarball packaging**:
   ```bash
   conda-pack --ignore-missing-files --exclude lib/python3.1 \
     -n mlserver-<uuid> -o tests/testdata/.cache/environment-py310.tar.gz
   ```

5. **Cleanup**: Conda environment removed after packaging

6. **Test execution**: All 17+ environment tests run against these tarballs

7. **Multi-version testing**: Tests parameterized across Python versions:
   - Python 3.10: `test_from_tarball[py310]`
   - Python 3.11: `test_from_tarball[py311]`
   - Python 3.12: `test_from_tarball[py312]`

---

## Python Version Testing

### Venv Mode (Single-version)

With venv, only the current Python version is tested.

**Example output** (running on Python 3.11):
```text
tests/env/test_env.py::test_from_tarball[py311] PASSED
```

### Conda Mode (Multi-version)

When conda is enabled, tests run against all supported Python versions:

```python
MIN_PYTHON_VERSION = (3, 10)
MAX_PYTHON_VERSION = (3, 12)
PYTHON_VERSIONS = [
    (major, minor)
    for major in range(MIN_PYTHON_VERSION[0], MAX_PYTHON_VERSION[0] + 1)
    for minor in range(MIN_PYTHON_VERSION[1], MAX_PYTHON_VERSION[1] + 1)
]
# Result: [(3, 10), (3, 11), (3, 12)]
```

Tests are parameterized using pytest:

```python
@pytest.fixture(
    params=get_python_versions(),
    ids=[f"py{major}{minor}" for (major, minor) in get_python_versions()],
)
def env_python_version(request: pytest.FixtureRequest) -> tuple[int, int]:
    return request.param
```

**Example output**:
```
tests/env/test_env.py::test_from_tarball[py310] PASSED
tests/env/test_env.py::test_from_tarball[py311] PASSED
tests/env/test_env.py::test_from_tarball[py312] PASSED
```

---

## Summary

Choose the mode that fits your development environment. Both venv and conda produce identical test results and compatible MLServer environments. See the [Tox Environments](#tox-environments) table for a full comparison.
