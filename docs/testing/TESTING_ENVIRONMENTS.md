# Testing with Conda and Non-Conda Environments

This guide provides an overview of how MLServer tests support both conda and non-conda environments, how to configure tests for each scenario, and the underlying mechanisms.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [With Conda](#with-conda)
  - [Without Conda](#without-conda)
- [Configuration](#configuration)
  - [Environment Variable: USE_CONDA](#environment-variable-use_conda)
  - [Tox Environments](#tox-environments)
  - [Python Version Testing](#python-version-testing)

---

## Overview

MLServer's test suite supports testing custom Python environments in **two modes**:

1. **Conda mode** (default): Uses `conda` and `conda-pack` to create environment tarballs
2. **Non-conda mode** (ODH): Uses Python `venv` and `pip` to create environment tarballs

Both approaches create compatible tarballs that work with MLServer's `Environment` class, which is **environment-manager agnostic**.

### Key Features

- **Flexible environment creation**: Conda or venv, your choice
- **Multiple Python versions**: Test across Python 3.9-3.12
- **Efficient caching**: Tarballs cached in `tests/testdata/.cache/`
- **Compatible outputs**: Both methods produce identical results
- **No code changes needed**: Same tests work in both modes

---

## Quick Start

### Run tests WITHOUT conda (venv mode):

```bash
# Default for odh-mlserver - uses venv/pip
tox -e odh-mlserver

# Or set explicitly
USE_CONDA=false pytest tests/
```

### Run tests WITH conda:

```bash
# Default for mlserver - uses conda-pack
tox -e mlserver

# Or set explicitly
USE_CONDA=true pytest tests/
```

---

## How It Works

### With Conda

When `USE_CONDA=true` (or `1`, `yes`):

1. **Test fixture reads**: [tests/testdata/environment.yml](tests/testdata/environment.yml)
   ```yaml
   name: custom-runtime-environment
   channels:
     - conda-forge
   dependencies:
     - python == 3.12
     - scikit-learn == 1.3.1
     - pip:
         - mlserver @ git+${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}.git@${GITHUB_REF}
   ```

2. **Python version injection**: The test infrastructure dynamically modifies the environment.yml to test multiple Python versions:
   - Reads the template environment.yml
   - Replaces `python == 3.12` with requested version (e.g., `python == 3.9`)
   - Saves as `environment-py39.yml`, `environment-py310.yml`, etc.

3. **Environment creation**: For each Python version:
   ```bash
   conda env create -n mlserver-<uuid> -f environment-py39.yml
   ```

4. **Tarball packaging**:
   ```bash
   conda-pack --ignore-missing-files --exclude lib/python3.1 \
     -n mlserver-<uuid> -o tests/testdata/.cache/environment-py39.tar.gz
   ```

5. **Cleanup**: Conda environment removed after packaging

6. **Test execution**: All 17+ environment tests run against these tarballs

7. **Multi-version testing**: Tests parameterized across Python versions:
   - Python 3.9: `test_from_tarball[py39]`
   - Python 3.10: `test_from_tarball[py310]`
   - Python 3.11: `test_from_tarball[py311]`
   - Python 3.12: `test_from_tarball[py312]`

### Without Conda

When `USE_CONDA=false` (default for ODH):

1. **Test fixture reads**: [tests/testdata/environment.txt](tests/testdata/environment.txt)
   ```
   scikit-learn==1.3.1
   ../../.
   ```

   Note: `../../.` installs the local MLServer package

2. **Environment creation**: For each Python version:
   ```bash
   python3.9 -m venv --copies /tmp/mlserver-<uuid>
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

---

## Configuration

### Environment Variable: USE_CONDA

The `USE_CONDA` environment variable controls which environment creation method to use:

| Value | Conda Mode | Venv Mode | Notes |
|-------|-----------|-----------|-------|
| `true`, `1`, `yes` | ✅ | ❌ | Tests all Python versions (3.9-3.12) |
| `false`, `0`, `no` | ❌ | ✅ | Tests current Python version only |
| (unset) | ❌ | ✅ | Default: venv mode |

**Implementation**: See [tests/conftest.py:46-50](tests/conftest.py#L46-L50)

```python
def get_python_versions() -> list[tuple[int, int]]:
    use_conda = os.environ.get("USE_CONDA", "").lower() in ("1", "true", "yes")
    if use_conda:
        return PYTHON_VERSIONS  # [(3,9), (3,10), (3,11), (3,12)]

    return [(sys.version_info.major, sys.version_info.minor)]  # Current only
```

### Tox Environments

MLServer provides multiple tox environments with different conda/venv configurations:

#### Standard Environments (Conda-enabled)

```ini
[testenv:mlserver]
# mlserver default - tests against all Python versions
commands = python -m pytest tests/ ...
set_env =
    USE_CONDA = {env:USE_CONDA:true}  # Explicitly enabled conda
```

```ini
[testenv:all-runtimes]
# Tests all ML runtimes with conda
commands = python -m pytest tests/ runtimes/ ...
set_env =
    USE_CONDA = {env:USE_CONDA:true}  # Explicitly enabled conda
```

#### ODH Environments (Venv-only)

```ini
[testenv:odh-mlserver]
# OpenDataHub variant - uses venv, current Python only
commands = python -m pytest tests/ ...
    --ignore={toxinidir}/tests/cli/test_build.py  # Skips conda-dependent build tests
set_env =
    USE_CONDA = {env:USE_CONDA:false}  # Explicitly disable conda
```

```ini
[testenv:odh-all-runtimes]
# ODH with limited runtime testing (sklearn, xgboost, lightgbm only)
commands = python -m pytest tests/ runtimes/sklearn/ runtimes/xgboost/ runtimes/lightgbm/ ...
    --ignore={toxinidir}/tests/cli/test_build.py
set_env =
    USE_CONDA = {env:USE_CONDA:false}
```

**Key differences**:

| Feature | Standard (`mlserver`) | ODH (`odh-mlserver`) |
|---------|---------------------|---------------------|
| Conda usage | Auto-detect | Disabled by default |
| Python versions | All (3.9-3.12) | Current only |
| Build tests | Included | Excluded |
| Runtimes tested | All | Subset (sklearn, xgboost, lightgbm) |

### Python Version Testing

#### Conda Mode (Multi-version)

When conda is enabled, tests run against all supported Python versions:

```python
# From tests/conftest.py
MIN_PYTHON_VERSION = (3, 9)
MAX_PYTHON_VERSION = (3, 12)
PYTHON_VERSIONS = [
    (major, minor)
    for major in range(MIN_PYTHON_VERSION[0], MAX_PYTHON_VERSION[0] + 1)
    for minor in range(MIN_PYTHON_VERSION[1], MAX_PYTHON_VERSION[1] + 1)
]
# Result: [(3, 9), (3, 10), (3, 11), (3, 12)]
```

Tests are parameterized using pytest:

```python
@pytest.fixture(
    params=get_python_versions(),
    ids=[f"py{major}{minor}" for (major, minor) in get_python_versions()],
)
def env_python_version(request: pytest.FixtureRequest) -> Tuple[int, int]:
    return request.param
```

**Example output**:
```
tests/env/test_env.py::test_from_tarball[py39] PASSED
tests/env/test_env.py::test_from_tarball[py310] PASSED
tests/env/test_env.py::test_from_tarball[py311] PASSED
tests/env/test_env.py::test_from_tarball[py312] PASSED
```

#### Venv Mode (Single-version)

Without conda, only the current Python version is tested:

```python
def get_python_versions() -> list[tuple[int, int]]:
    use_conda = os.environ.get("USE_CONDA", "").lower() in ("1", "true", "yes")
    if use_conda:
        return PYTHON_VERSIONS

    # Return only current Python version
    return [(sys.version_info.major, sys.version_info.minor)]
```

**Example output** (running on Python 3.11):
```
tests/env/test_env.py::test_from_tarball[py311] PASSED
```

---

## Summary

MLServer's test infrastructure provides **flexible environment management**:

| Aspect | Conda Mode | Venv Mode |
|--------|-----------|-----------|
| **Default for** | mlserver | odh-mlserver |
| **Requires** | conda, conda-pack | venv, pip (built-in) |
| **Python versions** | All (3.9-3.12) | Current only |
| **Speed** | Slower (conda install) | Faster (pip install) |
| **Tarball source** | environment.yml | environment.txt |
| **Test coverage** | Identical | Identical |
| **Environment activation** | Same (manager-agnostic) | Same (manager-agnostic) |
| **Tox command** | `tox -e mlserver` | `tox -e odh-mlserver` |
| **Environment variable** | `USE_CONDA=true` | `USE_CONDA=false` |

**Bottom line**: Choose the mode that fits your development environment. Both produce identical test results and compatible MLServer environments.
