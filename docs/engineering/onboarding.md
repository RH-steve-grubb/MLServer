# Developer Onboarding Guide

> Everything a new contributor needs to get MLServer running locally and
> start making changes.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11–3.12 | Runtime |
| Poetry | 2.5.1 | Dependency management and virtualenvs |
| Docker | 20+ | Container builds and integration tests |
| Make | any | Task runner (optional, wraps common commands) |
| protoc | 3.x | gRPC protobuf compilation (only if modifying `.proto` files) |

## Repository Structure

```
MLServer/
├── mlserver/                # Core server package
│   ├── server.py            # Top-level MLServer orchestrator
│   ├── model.py             # MLModel base class
│   ├── registry.py          # Multi-model registry with versioning
│   ├── settings.py          # Pydantic Settings (server + model)
│   ├── handlers/            # DataPlane + ModelRepositoryHandlers
│   ├── rest/                # FastAPI REST transport
│   ├── grpc/                # gRPC transport + protobuf converters
│   ├── metrics/             # Prometheus metrics server
│   ├── parallel/            # Multiprocessing worker pool
│   ├── batching/            # Adaptive request batching
│   ├── codecs/              # V2 ↔ Python type conversion
│   ├── types/               # V2 Inference Protocol type definitions
│   ├── middleware.py         # Inference middleware chain
│   ├── cloudevents.py        # CloudEvents middleware
│   └── cli/                 # CLI entry points (mlserver start/build)
├── runtimes/                # Runtime plugins (one per framework)
│   ├── sklearn/             # ← Shipped in production images
│   ├── xgboost/             # ← Shipped in production images
│   ├── lightgbm/            # ← Shipped in production images
│   └── onnx/                # ← Shipped in production images
├── tests/                   # Test suite (pytest)
├── docs/                    # Sphinx documentation + engineering docs
├── hack/                    # Build scripts and utilities
└── pyproject.toml           # Poetry project definition
```

## Local Setup

### 1. Clone and set up the core package

```bash
git clone https://github.com/opendatahub-io/MLServer.git
cd MLServer
poetry install
```

### 2. Install a runtime plugin (optional)

Each runtime is a separate Poetry project. To work on a specific runtime:

```bash
cd runtimes/sklearn
poetry install
cd ../..
```

### 3. Run the server

```bash
# Start with default settings (looks for models in current directory)
poetry run mlserver start .

# Or with a specific settings file
poetry run mlserver start /path/to/model/directory
```

### 4. Run tests

```bash
# Core tests
poetry run pytest tests/

# Runtime-specific tests
cd runtimes/sklearn
poetry run pytest tests/
```

## Key Concepts for New Contributors

### The MLModel Contract

Every runtime plugin subclasses `mlserver.MLModel` and overrides:

- `load()` — load model artifacts, return `True` when ready.
- `predict(payload)` — run inference, return `InferenceResponse`.
- `unload()` — release resources (optional).

The model receives an `InferenceRequest` conforming to the V2 protocol and
must return an `InferenceResponse` with the same structure.

### The DataPlane

The `DataPlane` class is the single implementation of inference logic, shared
by REST and gRPC transports. Transport servers are thin adapters
that convert wire formats and delegate to the DataPlane.

### Registry Hooks

Model lifecycle is managed through hooks on the `MultiModelRegistry`:

- `on_model_load` — called sequentially after model initialisation (parallel
  pool loading, custom handler registration, batching setup).
- `on_model_reload` — called sequentially during model reload.
- `on_model_unload` — called in parallel during model unload.

### Configuration

MLServer uses Pydantic Settings with environment variable support:

- `MLSERVER_*` — server-level settings.
- `MLSERVER_MODEL_*` — model-level settings.
- `settings.json` — server settings file.
- `model-settings.json` — per-model settings file.

See `mlserver/settings.py` for all available settings and their defaults.

## Development Workflow

1. Create a feature branch from `main`.
2. Make changes, ensuring tests pass (`poetry run pytest`).
3. Add docstrings to any new public functions or classes.
4. Submit a PR targeting `main`.

## Further Reading

- [Architecture Document](architecture.md) — system design and Mermaid
  diagrams.
- [ADRs](adr/) — key design decisions and their rationale.
- [Security Document](security.md) — runtime security model.
