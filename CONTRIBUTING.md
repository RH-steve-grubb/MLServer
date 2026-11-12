# Contributing to MLServer

Thank you for your interest in contributing to MLServer. This guide covers the
development setup, coding conventions, and PR process.

## Development Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11–3.12 | Runtime |
| Poetry | 2.5.1 | Dependency management |
| Make | any | Task runner |
| Docker | 20+ | Container builds and integration tests |

### Clone and Install

```bash
git clone https://github.com/opendatahub-io/MLServer.git
cd MLServer
poetry install
```

To work on a specific runtime plugin:

```bash
cd runtimes/sklearn
poetry install
```

### Running Locally

```bash
# Start the server (looks for models in current directory)
poetry run mlserver start .
```

## Code Style

- **Python** — follow PEP 8. The codebase uses type annotations throughout.
- **Docstrings** — use triple-quoted docstrings on all public classes,
  methods, and functions. Keep them concise — explain *why*, not *what*,
  unless the behaviour is non-obvious.
- **Imports** — group into stdlib, third-party, and local. Use relative
  imports within the `mlserver` package.
- **Settings** — all configuration uses Pydantic Settings classes in
  `mlserver/settings.py`. New settings should include a docstring and
  support environment variable override via the `MLSERVER_` prefix.

## Commit Message Convention

Format: `type(scope): description`

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build, CI, dependency updates |

Examples:
```
feat(batching): add configurable batch timeout per model
fix(parallel): handle worker crash during model reload
docs: update architecture diagram with gRPC flow
```

## Running Tests

```bash
# Full test suite
make test

# Core tests only
poetry run pytest tests/

# Single file
poetry run pytest tests/batch_processing/test_rest.py

# Runtime-specific tests
cd runtimes/sklearn && poetry run pytest tests/
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make changes and ensure tests pass.
3. Add docstrings to any new public functions or classes.
4. Choose a PR template from [`/docs/PULL_REQUEST_TEMPLATE/`](./docs/PULL_REQUEST_TEMPLATE/).
5. Submit the PR targeting `master`.

### Before Opening a PR

- Is the change important and ready enough to ask the community to spend
  time reviewing?
- Have you searched for existing, related issues and pull requests?
- Is the change being proposed clearly explained and motivated?

## Adding a New Runtime

1. Create a new directory under `runtimes/` with a Poetry project.
2. Subclass `mlserver.MLModel` and implement `load()` and `predict()`.
3. Add tests in the runtime's `tests/` directory.
4. Add documentation in `docs/runtimes/`.

### Shipped vs Community Runtimes

The ODH midstream production images include only **four shipped runtimes**:
sklearn, xgboost, lightgbm, and onnx. These are controlled by the
`ARG RUNTIMES` in the `Dockerfile` and the `odh-runtimes` dependency group
in `pyproject.toml`.

If a new runtime should be **shipped in production images**:

5. Add the runtime to the `RUNTIMES` build arg in `Dockerfile`.
6. Add the canonical import path to `ALLOWED_MODEL_IMPLEMENTATIONS` in
   `mlserver/settings.py`.
7. Add tests validating allowlist behaviour.

If the runtime is a **community runtime** (source available but not shipped):

5. It will remain installable via `pip` but will not be included in
   production images.
6. Users can bake it into custom images with `mlserver build`.

## Engineering Documentation

For architectural context, design decisions, and the security model, see:

- [Architecture](./docs/engineering/architecture.md)
- [ADRs](./docs/engineering/adr/)
- [Security Model](./docs/engineering/security.md)
- [Developer Onboarding](./docs/engineering/onboarding.md)

## License

When you contribute code, you affirm that the contribution is your original
work and that you license the work to the project under the project's open
source license.
