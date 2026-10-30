# MLServer Architecture

> Internal engineering reference for the MLServer inference platform.
> This document describes the system architecture, component interactions,
> data flows, and key design patterns used throughout the codebase.

---

## Table of Contents

- [System Overview](#system-overview)
- [C4 Context Diagram](#c4-context-diagram)
- [Core Domain Model](#core-domain-model)
- [Inference Request Flow](#inference-request-flow)
- [Model Lifecycle](#model-lifecycle)
- [Parallel Inference](#parallel-inference)
- [Adaptive Batching](#adaptive-batching)
- [Codec System](#codec-system)
- [Runtime Security](#runtime-security)
- [Server Component Layout](#server-component-layout)
- [Configuration Architecture](#configuration-architecture)
- [Runtime Plugins](#runtime-plugins)

---

## System Overview

MLServer is a Python-based inference server implementing the
[V2 Inference Protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol)
(also known as the Open Inference Protocol). It serves ML models over REST
(FastAPI/Uvicorn) and gRPC simultaneously and a dedicated Prometheus metrics endpoint.

Key architectural properties:

- **Multi-model serving** — a single server instance hosts multiple models,
  each with independent versioning, lifecycle, and readiness state.
- **Parallel inference** — bypasses the Python GIL via multiprocessing worker
  pools, enabling true CPU-parallel prediction across models.
- **Adaptive batching** — transparently groups incoming requests into batches
  based on configurable size and time thresholds.
- **Pluggable runtimes** — model-serving logic is decoupled into runtime
  plugins (sklearn, xgboost, lightgbm, onnx) that extend the `MLModel`
  base class. The ODH midstream images ship these four runtimes.
- **Codec pipeline** — a type-conversion layer that encodes/decodes between
  high-level Python objects (NumPy arrays, Pandas DataFrames) and the
  V2 wire format.
- **Runtime security** — a dual-mode (DEVELOPMENT / PRODUCTION) security
  model that controls which model implementations can be loaded.

---

## C4 Context Diagram

The system boundary of MLServer and its external interactions.

![Software Architecture](../diagrams/architecture_animated.gif)

```mermaid
graph TB
    subgraph "External Actors"
        Client["Client Application"]
        Prometheus["Prometheus"]
        ModelStore["Model Artifact Store<br/>(filesystem / S3 / PVC)"]
    end

    subgraph "MLServer Boundary"
        direction TB
        REST["REST Server<br/>(FastAPI + Uvicorn)<br/>:8080"]
        GRPC["gRPC Server<br/>(grpcio)<br/>:8081"]
        Metrics["Metrics Server<br/>(FastAPI + Uvicorn)<br/>:8082"]
        DataPlane["DataPlane<br/>(inference orchestration)"]
        Registry["MultiModelRegistry<br/>(model lifecycle)"]
        Pool["InferencePoolRegistry<br/>(parallel workers)"]
        Batcher["AdaptiveBatcher<br/>(request batching)"]
    end

    Client -->|"HTTP/REST"| REST
    Client -->|"gRPC"| GRPC
    Prometheus -->|"scrape /metrics"| Metrics

    REST --> DataPlane
    GRPC --> DataPlane
    DataPlane --> Registry
    Registry --> Pool
    Registry --> Batcher
    Pool --> ModelStore
    Registry --> ModelStore

    classDef server fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef external fill:#F5A623,stroke:#C47D1A,color:#fff
    classDef core fill:#7ED321,stroke:#5A9A18,color:#fff

    class REST,GRPC,Metrics server
    class Client,Prometheus,ModelStore external
    class DataPlane,Registry,Pool,Batcher core
```

---

## Core Domain Model

The primary classes and their relationships.

```mermaid
classDiagram
    direction LR

    class MLServer {
        -Settings _settings
        -RESTServer _rest_server
        -GRPCServer _grpc_server
        -MetricsServer _metrics_server
        -MultiModelRegistry _model_registry
        -InferencePoolRegistry _inference_pool_registry
        -DataPlane _data_plane
        +start(models_settings) void
        +stop(sig) void
        +add_custom_handlers(model) MLModel
        +reload_custom_handlers(old, new) MLModel
        +remove_custom_handlers(model) MLModel
    }

    class Settings {
        +bool debug
        +int parallel_workers
        +str host
        +int http_port
        +int grpc_port
        +int metrics_port
        +str metrics_endpoint
        +bool strict_readiness
        +bool empty_registry_readiness
        +str log_level
        +bool use_structured_logging
        +CORSSettings cors_settings
    }

    class ModelSettings {
        +str name
        +str implementation_
        +str platform
        +int max_batch_size
        +float max_batch_time
        +ModelParameters parameters
        +implementation() type~MLModel~
    }

    class MLModel {
        -ModelSettings _settings
        +bool ready
        +load() bool
        +predict(payload) InferenceResponse
        +predict_stream(payloads) AsyncIterator
        +unload() bool
        +decode(input, codec) Any
        +decode_request(request, codec) Any
        +encode_response(payload, codec) InferenceResponse
        +encode(payload, output, codec) ResponseOutput
        +metadata() MetadataModelResponse
        +name str
        +version str
        +settings ModelSettings
    }

    class MultiModelRegistry {
        -dict~str,SingleModelRegistry~ _models
        +load(model_settings) MLModel
        +unload(name) void
        +unload_version(name, version) void
        +get_model(name, version) MLModel
        +get_models(name) list~MLModel~
    }

    class SingleModelRegistry {
        -dict~str,MLModel~ _versions
        -MLModel _default
        +load(model_settings) MLModel
        +unload() void
        +unload_version(version) void
        +get_model(version) MLModel
        +get_models() list~MLModel~
    }

    class DataPlane {
        -Settings _settings
        -MultiModelRegistry _model_registry
        -bool _startup_complete
        -InferenceMiddlewares _inference_middleware
        +live() bool
        +ready() bool
        +startup_complete() void
        +is_startup_complete bool
        +model_ready(name, version) bool
        +metadata() MetadataServerResponse
        +runtimes() RuntimeSecurityResponse
        +infer(payload, name, version) InferenceResponse
        +infer_stream(payloads, name, version) AsyncIterator
    }

    class InferencePoolRegistry {
        -Settings _settings
        -InferencePool _default_pool
        -dict~str,InferencePool~ _pools
        +load_model(model) MLModel
        +reload_model(old, new) MLModel
        +unload_model(model) MLModel
        +model_initialiser(settings) MLModel
        +close() void
    }

    class InferencePool {
        -dict~int,Worker~ _workers
        -WorkerRegistry _worker_registry
        -Dispatcher _dispatcher
        +load_model(model) MLModel
        +reload_model(old, new) MLModel
        +unload_model(model) MLModel
        +on_worker_stop(pid, exit_code) void
        +close() void
    }

    class AdaptiveBatcher {
        -MLModel _model
        -int _max_batch_size
        -float _max_batch_time
        +predict(req) InferenceResponse
    }

    MLServer --> Settings
    MLServer --> MultiModelRegistry
    MLServer --> DataPlane
    MLServer --> InferencePoolRegistry
    MultiModelRegistry *-- SingleModelRegistry : contains 1..*
    SingleModelRegistry o-- MLModel : manages versions
    MLModel --> ModelSettings
    DataPlane --> MultiModelRegistry
    InferencePoolRegistry *-- InferencePool : manages pools
    InferencePool *-- Worker : spawns N
    AdaptiveBatcher --> MLModel : wraps predict
    ModelSettings --> Settings : nested within server
```

---

## Inference Request Flow

End-to-end sequence from client request to response, covering both REST and
gRPC paths converging at the DataPlane.

```mermaid
sequenceDiagram
    participant Client
    participant REST as RESTServer<br/>(Endpoints)
    participant GRPC as gRPC<br/>(InferenceServicer)
    participant DP as DataPlane
    participant MW as InferenceMiddlewares<br/>(CloudEvents)
    participant Reg as MultiModelRegistry
    participant Model as MLModel<br/>(runtime plugin)

    alt REST path
        Client->>REST: POST /v2/models/{name}/infer
        REST->>REST: Extract headers, parse InferenceRequest
        REST->>DP: infer(payload, name, version)
    else gRPC path
        Client->>GRPC: ModelInfer(ModelInferRequest)
        GRPC->>GRPC: Convert protobuf to InferenceRequest
        GRPC->>DP: infer(payload, name, version)
    end

    DP->>DP: Generate request UUID
    DP->>DP: Start Prometheus timer + error counter

    DP->>Reg: get_model(name, version)
    Reg-->>DP: model instance

    alt Model not ready
        DP-->>Client: 400 ModelNotReady
    end

    DP->>MW: request_middleware(payload, settings)
    MW-->>DP: processed payload

    DP->>Model: predict(payload)
    Model-->>DP: InferenceResponse

    DP->>MW: response_middleware(response, settings)
    MW-->>DP: processed response
    DP->>DP: Increment success counter

    alt REST path
        DP-->>REST: InferenceResponse
        REST->>REST: Set response headers
        REST-->>Client: 200 JSON
    else gRPC path
        DP-->>GRPC: InferenceResponse
        GRPC->>GRPC: Convert to ModelInferResponse protobuf
        GRPC-->>Client: ModelInferResponse
    end
```

---

## Model Lifecycle

The load, reload, and unload sequence managed by the registry, including
hook execution order.

```mermaid
sequenceDiagram
    participant Server as MLServer
    participant MMR as MultiModelRegistry
    participant SMR as SingleModelRegistry
    participant Hooks as Registry Hooks
    participant IPR as InferencePoolRegistry
    participant Batch as load_batching
    participant Model as MLModel

    Note over Server,Model: Model Load (first time)
    Server->>MMR: load(model_settings)
    MMR->>MMR: Create SingleModelRegistry if new name
    MMR->>SMR: load(model_settings)
    SMR->>SMR: model_initialiser(settings)
    SMR->>SMR: _register(model) — add to versions dict
    
    loop on_model_load hooks (sequential)
        SMR->>IPR: load_model(model) → ParallelModel
        SMR->>Server: add_custom_handlers(model)
        SMR->>Batch: load_batching(model) → wraps predict
    end
    
    SMR->>SMR: _register(model) — save hook-modified version
    SMR->>Model: load()
    Model-->>SMR: ready = True
    SMR->>SMR: _refresh_default(model)

    Note over Server,Model: Model Reload (version exists)
    Server->>MMR: load(model_settings)
    MMR->>SMR: load(model_settings)
    SMR->>SMR: Find previous model for same version
    
    loop on_model_reload hooks (sequential)
        SMR->>IPR: reload_model(old, new)
        SMR->>Server: reload_custom_handlers(old, new)
    end
    
    SMR->>Model: new_model.load()
    Model-->>SMR: ready = True
    SMR->>SMR: _register(new_model)
    SMR->>Model: old_model.unload()

    Note over Server,Model: Model Unload
    Server->>MMR: unload(name)
    MMR->>SMR: unload()
    
    par on_model_unload hooks (parallel)
        SMR->>IPR: unload_model(model)
        SMR->>Server: remove_custom_handlers(model)
    end
    
    SMR->>Model: unload()
    Model-->>SMR: unloaded = True
    SMR->>SMR: Clear versions dict + default
```

---

## Parallel Inference

How MLServer bypasses the Python GIL using multiprocessing workers,
dispatchers, and message queues.

![Worker Pool Architecture](../diagrams/worker_pool.png)

```mermaid
flowchart TB
    subgraph "Main Process"
        direction TB
        DP["DataPlane.infer()"]
        PM["ParallelModel"]
        DISP["Dispatcher"]
        REQ_Q["Request Queues<br/>(per worker)"]
        RESP_Q["Response Queue<br/>(shared)"]
    end

    subgraph "Worker Process 1"
        direction TB
        W1["Worker (Process)"]
        W1_REG["MultiModelRegistry"]
        W1_MODEL["MLModel instance"]
        W1_SEL["select() loop"]
    end

    subgraph "Worker Process N"
        direction TB
        WN["Worker (Process)"]
        WN_REG["MultiModelRegistry"]
        WN_MODEL["MLModel instance"]
        WN_SEL["select() loop"]
    end

    DP -->|"predict(payload)"| PM
    PM -->|"ModelRequestMessage"| DISP
    DISP -->|"Round-robin"| REQ_Q
    REQ_Q -->|"Queue.put()"| W1_SEL
    REQ_Q -->|"Queue.put()"| WN_SEL

    W1_SEL -->|"get request"| W1_REG
    W1_REG --> W1_MODEL
    W1_MODEL -->|"ModelResponseMessage"| RESP_Q

    WN_SEL -->|"get request"| WN_REG
    WN_REG --> WN_MODEL
    WN_MODEL -->|"ModelResponseMessage"| RESP_Q

    RESP_Q -->|"dispatch to Future"| DISP
    DISP -->|"resolve Future"| PM
    PM -->|"InferenceResponse"| DP

    classDef mainProc fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef workerProc fill:#7ED321,stroke:#5A9A18,color:#fff

    class DP,PM,DISP,REQ_Q,RESP_Q mainProc
    class W1,W1_REG,W1_MODEL,W1_SEL,WN,WN_REG,WN_MODEL,WN_SEL workerProc
```

### Worker Lifecycle

Each `Worker` is a `multiprocessing.Process` that:

1. Ignores SIGINT/SIGTERM/SIGQUIT (the main process manages shutdown).
2. Installs uvloop, configures logging and metrics.
3. Initialises its own `MultiModelRegistry` (models are loaded independently
   in each worker).
4. Runs a `select()` loop on two queues: `_requests` (inference) and
   `_model_updates` (load/unload).
5. On unexpected exit, the `InferencePoolRegistry` detects SIGCHLD, spawns a
   replacement worker, and reloads all models from the `WorkerRegistry`.

### Dispatcher

The `Dispatcher` runs in the main process and:

- Routes `ModelRequestMessage` to workers via round-robin.
- Listens on the shared `_responses` queue for `ModelResponseMessage`.
- Resolves the corresponding `asyncio.Future` to return the result to the
  caller.
- Handles `ModelUpdateMessage` (load/unload) by broadcasting to all workers.

### Environment Isolation

The `InferencePoolRegistry` manages multiple `InferencePool` instances, each
associated with a Python environment (extracted from a tarball or an existing
path). Models sharing the same environment share the same pool. The default
pool uses the server's own Python environment.

---

## Adaptive Batching

How individual inference requests are grouped into batches transparently.

```mermaid
flowchart TB
    subgraph "Incoming Requests"
        R1["Request 1"]
        R2["Request 2"]
        R3["Request 3"]
        RN["Request N"]
    end

    subgraph "AdaptiveBatcher"
        direction TB
        QUEUE["AsyncIO Queue<br/>(maxsize = max_batch_size)"]
        TIMER["Batch Timer<br/>(max_batch_time seconds)"]
        BATCHER["_batcher() coroutine"]
        MERGE["BatchedRequests.merge()"]
    end

    subgraph "Prediction"
        PREDICT["model.predict(merged_request)"]
        SPLIT["BatchedRequests.split_response()"]
    end

    subgraph "Responses"
        F1["Future 1 → Response 1"]
        F2["Future 2 → Response 2"]
        F3["Future 3 → Response 3"]
        FN["Future N → Response N"]
    end

    R1 & R2 & R3 & RN -->|"queue_request()"| QUEUE

    QUEUE --> BATCHER
    TIMER --> BATCHER

    BATCHER -->|"Batch ready when:<br/>queue full OR timer expires"| MERGE
    MERGE -->|"Single merged InferenceRequest"| PREDICT
    PREDICT -->|"Single merged InferenceResponse"| SPLIT
    SPLIT --> F1 & F2 & F3 & FN

    classDef req fill:#F5A623,stroke:#C47D1A,color:#fff
    classDef batch fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef pred fill:#7ED321,stroke:#5A9A18,color:#fff
    classDef resp fill:#9B59B6,stroke:#7D3C98,color:#fff

    class R1,R2,R3,RN req
    class QUEUE,TIMER,BATCHER,MERGE batch
    class PREDICT,SPLIT pred
    class F1,F2,F3,FN resp
```

### Batching Mechanics

1. Each `AdaptiveBatcher` wraps a single model's `predict()` method.
2. Incoming requests are placed in an `asyncio.Queue` with
   `maxsize=max_batch_size`.
3. Each caller receives a `Future` that will resolve to its individual
   response.
4. The `_batcher()` coroutine collects requests until either:
   - The queue reaches `max_batch_size`, or
   - `max_batch_time` seconds have elapsed since the first request.
5. Collected requests are merged into a single `InferenceRequest` via
   `BatchedRequests`.
6. The merged request is passed to the model's original `predict()`.
7. The merged response is split back into individual responses, and each
   caller's `Future` is resolved.

Batching is enabled per-model via `ModelSettings.max_batch_size` (> 0) and
`ModelSettings.max_batch_time` (> 0.0). The `load_batching` hook decorates
the model at load time.

---

## Codec System

The type-conversion pipeline between high-level Python objects and V2
Inference Protocol wire format.

```mermaid
classDiagram
    direction TB

    class InputCodec {
        <<abstract>>
        +ContentType str$
        +TypeHint type$
        +can_encode(payload) bool$
        +encode_output(name, payload) ResponseOutput$
        +decode_output(output) Any$
        +encode_input(name, payload) RequestInput$
        +decode_input(input) Any$
    }

    class RequestCodec {
        <<abstract>>
        +ContentType str$
        +TypeHint type$
        +can_encode(payload) bool$
        +encode_response(model_name, payload) InferenceResponse$
        +decode_response(response) Any$
        +encode_request(payload) InferenceRequest$
        +decode_request(request) Any$
    }

    class CodecRegistry {
        -dict~str,InputCodecLike~ _input_codecs
        -dict~str,RequestCodecLike~ _request_codecs
        +register_input_codec(content_type, codec)
        +register_request_codec(content_type, codec)
        +find_input_codec(content_type, payload, type_hint) InputCodecLike
        +find_request_codec(content_type, payload, type_hint) RequestCodecLike
    }

    class NumpyCodec {
        +ContentType = "np"
    }
    class PandasCodec {
        +ContentType = "pd"
    }
    class StringCodec {
        +ContentType = "str"
    }
    class Base64Codec {
        +ContentType = "base64"
    }
    class DatetimeCodec {
        +ContentType = "datetime"
    }

    InputCodec <|-- NumpyCodec
    InputCodec <|-- StringCodec
    InputCodec <|-- Base64Codec
    InputCodec <|-- DatetimeCodec
    RequestCodec <|-- PandasCodec

    CodecRegistry o-- InputCodec : registers
    CodecRegistry o-- RequestCodec : registers

    note for InputCodec "Per-tensor encode/decode\n(individual input/output)"
    note for RequestCodec "Per-request encode/decode\n(e.g. DataFrames spanning\nmultiple tensors)"
```

### Codec Resolution

Codecs are resolved in priority order:

1. **Content-type header** — if `content_type` is set on the request input
   or model metadata, the matching codec is used directly.
2. **Payload type** — `can_encode(payload)` is called on each registered
   codec; the first match wins.
3. **Type hint** — `TypeHint` class variable is compared against the target
   type.

The `@register_input_codec` and `@register_request_codec` decorators
auto-register codec classes with the singleton `_CodecRegistry` at import
time.

### Two Levels of Codecs

| Level | Interface | Scope | Example |
|-------|-----------|-------|---------|
| Input/Output | `InputCodec` | Single tensor (one `RequestInput` / `ResponseOutput`) | NumPy array, string list |
| Request | `RequestCodec` | Entire request (multiple tensors) | Pandas DataFrame (columns = tensors) |

---

## Runtime Security

MLServer operates in one of two security modes, determined by the presence
of a trusted runtimes allowlist file at
`/etc/mlserver/trusted-runtimes.json`.

```mermaid
flowchart TB
    START["Server Startup"]
    CHECK{"Trusted runtimes<br/>allowlist file exists?<br/>/etc/mlserver/trusted-runtimes.json"}

    subgraph "DEVELOPMENT Mode"
        DEV_LOG["Log: DEVELOPMENT mode<br/>All implementations allowed"]
        DEV_LOAD["Load model:<br/>validate import path format<br/>allow sys.path injection<br/>allow dynamic module reload<br/>allow custom environments"]
    end

    subgraph "PRODUCTION Mode"
        PROD_LOG["Log: PRODUCTION mode<br/>N implementations allowed"]
        PROD_PARSE["Parse JSON allowlist"]
        PROD_VALIDATE["Validate each entry:<br/>canonical import path format<br/>uppercase class name"]
        PROD_LOAD["Load model:<br/>reject if not in allowlist<br/>no sys.path injection<br/>no dynamic reload<br/>no custom environments<br/>no environment tarballs"]
    end

    CORS_POLICY["CORS: explicit origins only<br/>origin regex unsupported"]
    PROD_EMPTY{"Allowlist<br/>empty?"}
    PROD_WARN["WARNING: no models<br/>can be loaded"]

    START --> CHECK
    CHECK -->|"No"| DEV_LOG
    DEV_LOG --> DEV_LOAD
    DEV_LOAD --> CORS_POLICY

    CHECK -->|"Yes"| PROD_LOG
    PROD_LOG --> PROD_PARSE
    PROD_PARSE --> PROD_VALIDATE
    PROD_VALIDATE --> PROD_EMPTY
    PROD_EMPTY -->|"Yes"| PROD_WARN
    PROD_EMPTY -->|"No"| PROD_LOAD
    PROD_LOAD --> CORS_POLICY

    classDef dev fill:#7ED321,stroke:#5A9A18,color:#fff
    classDef prod fill:#D0021B,stroke:#A50216,color:#fff
    classDef warn fill:#F5A623,stroke:#C47D1A,color:#fff
    classDef neutral fill:#4A90D9,stroke:#2C5F8A,color:#fff

    class DEV_LOG,DEV_LOAD dev
    class PROD_LOG,PROD_PARSE,PROD_VALIDATE,PROD_LOAD prod
    class PROD_WARN warn
    class START,CHECK,CORS_POLICY neutral
```

### Security Enforcement Points

Runtime security is enforced at multiple layers (defense in depth):

| Layer | When | What |
|-------|------|------|
| Settings parse | `ModelSettings` validation | `validate_trusted_runtime` rejects untrusted import paths |
| Property access | `ModelSettings.implementation` getter | Re-validates on every access in case `implementation_` was mutated |
| Server startup | `MLServer.start()` | `log_runtime_security_mode()` validates allowlist before any endpoints are accessible |
| Model repository | Discovery scan | Invalid model entries are skipped without crashing the server |
| CORS validation | `Settings` model validator | Blocks wildcard CORS in every mode; origin regex setting is unsupported |
| Environment validation | `ModelParameters` model validator | Blocks `environment_tarball` and `environment_path` in PRODUCTION mode |

### Import Path Validation

All runtime import paths must match the canonical pattern:
```
^[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*\.[A-Za-z][A-Za-z0-9_]*$
```

The final segment (class name) must begin with an uppercase letter. Built-in
runtime aliases (e.g., `mlserver_sklearn.sklearn.SKLearnModel` →
`mlserver_sklearn.SKLearnModel`) are canonicalized before validation.

---

## Server Component Layout

How the four server components are initialised and coordinated by `MLServer`.

```mermaid
flowchart LR
    subgraph "MLServer.__init__()"
        direction TB
        S["Settings"]
        SIG["Signal Handlers<br/>(SIGINT, SIGTERM, SIGQUIT)"]

        S --> MS_CHECK{"metrics_endpoint<br/>set?"}
        MS_CHECK -->|"Yes"| METRICS["MetricsServer"]
        MS_CHECK -->|"No"| SKIP_M["Skip metrics"]

        S --> PW_CHECK{"parallel_workers<br/>> 0?"}
        PW_CHECK -->|"Yes"| IPR["InferencePoolRegistry"]
        PW_CHECK -->|"No"| SKIP_P["Skip parallel"]

        IPR --> MMR["MultiModelRegistry<br/>(with parallel hooks)"]
        SKIP_P --> MMR2["MultiModelRegistry<br/>(without parallel hooks)"]

        MMR --> DP["DataPlane"]
        MMR2 --> DP
        DP --> MRH["ModelRepositoryHandlers"]

        DP --> REST_S["RESTServer"]
        DP --> GRPC_S["GRPCServer"]
    end

    subgraph "MLServer.start()"
        direction TB
        SEC["Validate runtime security"]
        SEC --> START_ALL["asyncio.gather:<br/>REST.start()<br/>gRPC.start()<br/>Metrics.start()"]
        START_ALL --> LOAD["Load initial models<br/>from repository"]
        LOAD --> READY["startup_complete()"]
    end
```

### Startup Sequence

1. Validate runtime security configuration (`log_runtime_security_mode()`).
   If the allowlist is invalid, startup is **aborted** before any endpoint
   becomes accessible.
2. Start all transport servers concurrently via `asyncio.gather()`.
3. Load all models from the model repository concurrently.
4. Mark `startup_complete()` on the `DataPlane` — the readiness endpoint stops reporting not ready because of startup. From then on it follows `strict_readiness` and `empty_registry_readiness`.
5. If any model fails to load during startup, the server shuts down
   gracefully.

### Shutdown Sequence

1. Close `InferencePoolRegistry` (stop all worker processes).
2. Stop `GRPCServer`.
3. Stop `RESTServer`.
4. Stop `MetricsServer`.

---

## Configuration Architecture

MLServer uses Pydantic Settings for configuration, supporting environment
variables, `.env` files, and JSON configuration files.

### Configuration Hierarchy

| Scope | Class | Env Prefix | File |
|-------|-------|------------|------|
| Server-wide | `Settings` | `MLSERVER_` | `settings.json` |
| Per-model | `ModelSettings` | `MLSERVER_MODEL_` | `model-settings.json` |
| Model parameters | `ModelParameters` | `MLSERVER_MODEL_` | nested in `model-settings.json` |
| CORS | `CORSSettings` | `MLSERVER_` | nested in `settings.json` |

### Key Settings

| Setting | Default | Effect |
|---------|---------|--------|
| `parallel_workers` | 1 | Number of multiprocessing workers (1 = disabled) |
| `max_batch_size` | 0 | Adaptive batching batch size (0 = disabled) |
| `max_batch_time` | 0.0 | Adaptive batching time window in seconds |
| `strict_readiness` | true | If true, all models in the registry must be ready. If false, model readiness is ignored after startup; an empty registry follows `empty_registry_readiness` |
| `empty_registry_readiness` | true | Report ready when no models are in the registry (both readiness modes) |

---

## Runtime Plugins

MLServer uses a pluggable runtime architecture. Each plugin is a separate
Python package in the `runtimes/` directory that provides an `MLModel`
subclass.

#### Shipped Runtimes (included in ODH midstream production images)

These runtimes are installed in the container images and covered by the
default trusted runtimes allowlist:

| Plugin | Package | MLModel Subclass | Framework |
|--------|---------|-----------------|-----------|
| scikit-learn | `mlserver-sklearn` | `SKLearnModel` | scikit-learn |
| XGBoost | `mlserver-xgboost` | `XGBoostModel` | XGBoost |
| LightGBM | `mlserver-lightgbm` | `LightGBMModel` | LightGBM |
| ONNX | `mlserver-onnx` | `OnnxModel` | ONNX Runtime |

### Plugin Contract

Every runtime plugin must:

1. Subclass `mlserver.MLModel`.
2. Override `load()` to load model artifacts and return readiness status.
3. Override `predict()` to perform inference and return an
   `InferenceResponse`.
4. Optionally override `predict_stream()` for streaming inference.
5. Optionally override `unload()` to release resources.
6. Optionally override `_configure_framework_logger()` to align framework
   log levels with MLServer's configured level.

### Plugin Discovery

In **DEVELOPMENT** mode, the `implementation` field in `model-settings.json`
is a Python import path (e.g., `mlserver_sklearn.SKLearnModel`). The model
folder is temporarily added to `sys.path` to support custom runtimes.

In **PRODUCTION** mode, only import paths present in the
`/etc/mlserver/trusted-runtimes.json` allowlist are permitted. Dynamic
`sys.path` injection and custom environments are disabled.

---

## Appendix: V2 Inference Protocol Endpoints

### REST Endpoints (port 8080)

| Method | Path | Handler |
|--------|------|---------|
| GET | `/v2/health/live` | `DataPlane.live()` |
| GET | `/v2/health/ready` | `DataPlane.ready()` |
| GET | `/v2` | `DataPlane.metadata()` |
| GET | `/v2/runtime-security` | `DataPlane.runtimes()` |
| GET | `/v2/models/{name}/ready` | `DataPlane.model_ready()` |
| GET | `/v2/models/{name}` | `DataPlane.model_metadata()` |
| POST | `/v2/models/{name}/infer` | `DataPlane.infer()` |
| POST | `/v2/models/{name}/infer_stream` | `DataPlane.infer_stream()` |
| POST | `/v2/repository/index` | `ModelRepositoryHandlers.index()` |
| POST | `/v2/repository/models/{name}/load` | `ModelRepositoryHandlers.load()` |
| POST | `/v2/repository/models/{name}/unload` | `ModelRepositoryHandlers.unload()` |

### gRPC Services (port 8081)

| RPC | Service | Handler |
|-----|---------|---------|
| `ServerLive` | `GRPCInferenceService` | `DataPlane.live()` |
| `ServerReady` | `GRPCInferenceService` | `DataPlane.ready()` |
| `ModelReady` | `GRPCInferenceService` | `DataPlane.model_ready()` |
| `ServerMetadata` | `GRPCInferenceService` | `DataPlane.metadata()` |
| `RuntimeSecurity` | `GRPCInferenceService` | `DataPlane.runtimes()` |
| `ModelMetadata` | `GRPCInferenceService` | `DataPlane.model_metadata()` |
| `ModelInfer` | `GRPCInferenceService` | `DataPlane.infer()` |
| `ModelStreamInfer` | `GRPCInferenceService` | `DataPlane.infer_stream()` |
| `RepositoryIndex` | `GRPCInferenceService` | `ModelRepositoryHandlers.index()` |
| `RepositoryModelLoad` | `GRPCInferenceService` | `ModelRepositoryHandlers.load()` |
| `RepositoryModelUnload` | `GRPCInferenceService` | `ModelRepositoryHandlers.unload()` |
