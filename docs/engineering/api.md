# API Reference

MLServer implements the [V2 Inference Protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol) over both REST (HTTP/1.1) and gRPC (HTTP/2) transports.

---

## Transports

| Transport | Default Port | Protocol | Toggle |
|-----------|-------------|----------|--------|
| REST | 8080 | HTTP/1.1 + JSON | Always enabled |
| gRPC | 8081 | HTTP/2 + Protobuf | Always enabled |
| Metrics | 8082 | HTTP/1.1 (Prometheus) | `metrics_endpoint` setting |

---

## REST API

All REST endpoints live under the `/v2` prefix and follow the V2 Inference Protocol specification.

### Health Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/v2/health/live` | Server liveness probe | `200` if alive, `400` otherwise |
| GET | `/v2/health/ready` | Server readiness probe | `200` if ready, `400` otherwise |

### Server Metadata

| Method | Path | Description | Response Body |
|--------|------|-------------|---------------|
| GET | `/v2` | Server metadata | `MetadataServerResponse` |
| GET | `/v2/runtimes` | Runtime security posture | `RuntimeSecurityResponse` |

**`MetadataServerResponse`**

```json
{
  "name": "mlserver",
  "version": "1.7.0",
  "extensions": ["model_repository", "runtime_security"]
}
```

**`RuntimeSecurityResponse`**

```json
{
  "mode": "PRODUCTION",
  "allowed_model_implementations": [
    "mlserver_sklearn.SKLearnModel",
    "mlserver_xgboost.XGBoostModel"
  ]
}
```

### Model Endpoints

| Method | Path | Description | Response Body |
|--------|------|-------------|---------------|
| GET | `/v2/models/{model_name}/ready` | Model readiness | `200` / `400` |
| GET | `/v2/models/{model_name}/versions/{version}/ready` | Versioned model readiness | `200` / `400` |
| GET | `/v2/models/{model_name}` | Model metadata | `MetadataModelResponse` |
| GET | `/v2/models/{model_name}/versions/{version}` | Versioned model metadata | `MetadataModelResponse` |
| POST | `/v2/models/{model_name}/infer` | Run inference | `InferenceResponse` |
| POST | `/v2/models/{model_name}/versions/{version}/infer` | Versioned inference | `InferenceResponse` |
| POST | `/v2/models/{model_name}/generate` | Run inference (alias) | `InferenceResponse` |
| POST | `/v2/models/{model_name}/versions/{version}/generate` | Versioned inference (alias) | `InferenceResponse` |
| POST | `/v2/models/{model_name}/infer_stream` | Streaming inference (SSE) | `StreamingResponse` |
| POST | `/v2/models/{model_name}/versions/{version}/infer_stream` | Versioned streaming inference | `StreamingResponse` |
| POST | `/v2/models/{model_name}/generate_stream` | Streaming inference (alias) | `StreamingResponse` |
| POST | `/v2/models/{model_name}/versions/{version}/generate_stream` | Versioned streaming (alias) | `StreamingResponse` |

### Model Repository Endpoints

| Method | Path | Description | Response Body |
|--------|------|-------------|---------------|
| POST | `/v2/repository/index` | List models in repository | `RepositoryIndexResponse` |
| POST | `/v2/repository/models/{model_name}/load` | Load model | `200` / `400` |
| POST | `/v2/repository/models/{model_name}/unload` | Unload model | `200` / `400` |

#### Repository index request

The repository index request supports the standard `ready` field and the
MLServer-specific `include_registry_models` extension:

```json
{
  "ready": false,
  "include_registry_models": false
}
```

- `include_registry_models` defaults to `false`, so the index lists only
  models discovered in the model repository.
- When `true`, the index also includes currently registered models in the
  model registry, including models that are absent from the repository.
- When `ready` is `true`, only `READY` models from the selected set are
  returned. When `ready` is omitted or `false`, all selected models and their
  current states are returned.

### Documentation Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v2/docs` | Swagger UI (server-level) |
| GET | `/v2/docs/dataplane.json` | OpenAPI schema (server-level) |
| GET | `/v2/models/{model_name}/docs` | Swagger UI (model-scoped) |
| GET | `/v2/models/{model_name}/docs/dataplane.json` | OpenAPI schema (model-scoped) |

---

## Request and Response Schemas

### InferenceRequest

```json
{
  "id": "optional-request-id",
  "parameters": {
    "content_type": "np",
    "headers": {}
  },
  "inputs": [
    {
      "name": "input-0",
      "shape": [1, 4],
      "datatype": "FP32",
      "parameters": { "content_type": "np" },
      "data": [[1.0, 2.0, 3.0, 4.0]]
    }
  ],
  "outputs": [
    {
      "name": "output-0",
      "parameters": {}
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Request identifier; auto-generated (UUID) if omitted |
| `parameters` | object | No | Extensible request parameters (allows arbitrary extra fields) |
| `parameters.content_type` | string | No | Codec hint (e.g. `"np"`, `"pd"`, `"str"`, `"base64"`) |
| `parameters.headers` | object | No | HTTP headers forwarded into the payload |
| `inputs` | array | Yes | One or more input tensors |
| `outputs` | array | No | Requested output tensors; if omitted, all outputs are returned |

### RequestInput

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Tensor name |
| `shape` | int[] | Yes | Tensor dimensions |
| `datatype` | string | Yes | Element type (see Datatypes below) |
| `parameters` | object | No | Per-tensor parameters (e.g. `content_type`) |
| `data` | any | Yes | Tensor payload (nested list or flat array) |

### InferenceResponse

```json
{
  "model_name": "iris-sklearn",
  "model_version": "v1",
  "id": "request-id",
  "parameters": {},
  "outputs": [
    {
      "name": "predict",
      "shape": [1, 1],
      "datatype": "INT64",
      "parameters": { "content_type": "np" },
      "data": [2]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | Name of the model that produced the response |
| `model_version` | string | Version of the model (may be null) |
| `id` | string | Matches the request ID |
| `parameters` | object | Response-level parameters |
| `outputs` | array | One or more output tensors |

### ResponseOutput

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tensor name |
| `shape` | int[] | Tensor dimensions |
| `datatype` | string | Element type |
| `parameters` | object | Per-tensor parameters |
| `data` | any | Tensor payload |

### MetadataModelResponse

```json
{
  "name": "iris-sklearn",
  "versions": ["v1", "v2"],
  "platform": "mlserver",
  "inputs": [
    { "name": "input-0", "datatype": "FP32", "shape": [-1, 4] }
  ],
  "outputs": [
    { "name": "predict", "datatype": "INT64", "shape": [-1, 1] }
  ],
  "parameters": {}
}
```

### RepositoryIndexRequest / Response

**Request:**
```json
{
  "ready": false
}
```

**Response:**
```json
[
  {
    "name": "iris-sklearn",
    "version": "v1",
    "state": "READY",
    "reason": ""
  },
  {
    "name": "mushroom-xgboost",
    "version": null,
    "state": "LOADING",
    "reason": "Model is being loaded"
  }
]
```

Model states: `UNKNOWN`, `READY`, `UNAVAILABLE`, `LOADING`, `UNLOADING`.

---

## Supported Datatypes

| V2 Datatype | Python / NumPy Equivalent | Size |
|------------|--------------------------|------|
| `BOOL` | `bool` / `np.bool_` | 1 byte |
| `UINT8` | `np.uint8` | 1 byte |
| `UINT16` | `np.uint16` | 2 bytes |
| `UINT32` | `np.uint32` | 4 bytes |
| `UINT64` | `np.uint64` | 8 bytes |
| `INT8` | `np.int8` | 1 byte |
| `INT16` | `np.int16` | 2 bytes |
| `INT32` | `np.int32` | 4 bytes |
| `INT64` | `np.int64` | 8 bytes |
| `FP16` | `np.float16` | 2 bytes |
| `FP32` | `np.float32` | 4 bytes |
| `FP64` | `np.float64` | 8 bytes |
| `BYTES` | `bytes` | Variable |

---

## gRPC API

The gRPC service is defined in `proto/dataplane.proto` under the `inference.GRPCInferenceService` package.

### Service Definition

```protobuf
service GRPCInferenceService {
  rpc ServerLive(ServerLiveRequest) returns (ServerLiveResponse);
  rpc ServerReady(ServerReadyRequest) returns (ServerReadyResponse);
  rpc ModelReady(ModelReadyRequest) returns (ModelReadyResponse);
  rpc ServerMetadata(ServerMetadataRequest) returns (ServerMetadataResponse);
  rpc RuntimeSecurity(RuntimeSecurityRequest) returns (RuntimeSecurityResponse);
  rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse);
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse);
  rpc ModelStreamInfer(stream ModelInferRequest) returns (stream ModelInferResponse);
  rpc RepositoryIndex(RepositoryIndexRequest) returns (RepositoryIndexResponse);
  rpc RepositoryModelLoad(RepositoryModelLoadRequest) returns (RepositoryModelLoadResponse);
  rpc RepositoryModelUnload(RepositoryModelUnloadRequest) returns (RepositoryModelUnloadResponse);
}
```

### RPC Reference

| RPC | Request | Response | Description |
|-----|---------|----------|-------------|
| `ServerLive` | `ServerLiveRequest` | `ServerLiveResponse` | Liveness probe |
| `ServerReady` | `ServerReadyRequest` | `ServerReadyResponse` | Readiness probe |
| `ModelReady` | `ModelReadyRequest` | `ModelReadyResponse` | Model readiness check |
| `ServerMetadata` | `ServerMetadataRequest` | `ServerMetadataResponse` | Server name, version, extensions |
| `RuntimeSecurity` | `RuntimeSecurityRequest` | `RuntimeSecurityResponse` | Security mode and allowlist |
| `ModelMetadata` | `ModelMetadataRequest` | `ModelMetadataResponse` | Model inputs, outputs, platform |
| `ModelInfer` | `ModelInferRequest` | `ModelInferResponse` | Unary inference |
| `ModelStreamInfer` | `stream ModelInferRequest` | `stream ModelInferResponse` | Bidirectional streaming inference |
| `RepositoryIndex` | `RepositoryIndexRequest` | `RepositoryIndexResponse` | List repository models |
| `RepositoryModelLoad` | `RepositoryModelLoadRequest` | `RepositoryModelLoadResponse` | Load a model |
| `RepositoryModelUnload` | `RepositoryModelUnloadRequest` | `RepositoryModelUnloadResponse` | Unload a model |

### Raw Tensor Contents

gRPC supports an optimized "raw" byte encoding for tensor data via `raw_input_contents` and `raw_output_contents`.
When these fields are populated, tensor data is sent as flattened, one-dimensional, row-major byte arrays instead of the typed `InferTensorContents` message.
This avoids protobuf serialization overhead for large tensors.

The client signals raw mode by populating `raw_input_contents`; the server detects this via the `_GetReturnRaw` check and uses the same format for the response.

### Header Propagation

HTTP headers are propagated through gRPC metadata:

- **Request path:** gRPC metadata → `to_headers()` → `insert_headers()` → `InferenceRequest.parameters.headers`
- **Response path:** `InferenceResponse.parameters.headers` → `extract_headers()` → `to_metadata()` → gRPC trailing metadata

---

## Streaming Inference (SSE)

The REST streaming endpoints (`/infer_stream`, `/generate_stream`) use Server-Sent Events (SSE) to deliver incremental responses.

Each SSE frame contains a complete `InferenceResponse` JSON object:

```
data: {"model_name":"my-llm","id":"req-1","outputs":[{"name":"output","shape":[1],"datatype":"BYTES","data":["Hello"]}]}

data: {"model_name":"my-llm","id":"req-1","outputs":[{"name":"output","shape":[1],"datatype":"BYTES","data":[" world"]}]}

```

All frames in a stream share the same `id` (set from the first request payload).

---

## Error Handling

### REST Errors

Errors return appropriate HTTP status codes with a JSON body:

```json
{
  "error": "Model 'unknown-model' not found."
}
```

| Error Type | HTTP Status |
|-----------|-------------|
| `ModelNotFound` | 404 |
| `ModelNotReady` | 400 |
| `InferenceError` | 400 |
| `InvalidModelURI` | 400 |
| Server error | 500 |

### gRPC Errors

MLServer errors are mapped to gRPC status codes:

| Error Type | gRPC Status |
|-----------|-------------|
| `ModelNotFound` | `NOT_FOUND` |
| `ModelNotReady` | `FAILED_PRECONDITION` |
| `InferenceError` | `INVALID_ARGUMENT` |
| Other `MLServerError` | `INVALID_ARGUMENT` |
| Unexpected exception | `INTERNAL` |

---

## Metrics

When `metrics_endpoint` is set (default: `/metrics`), a Prometheus-compatible metrics server runs on `metrics_port` (default: `8082`).

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `model_infer_request_success` | Counter | `model`, `version` | Successful inference count |
| `model_infer_request_failure` | Counter | `model`, `version` | Failed inference count |
| `model_infer_request_duration` | Summary | `model`, `version` | Inference latency (seconds) |
| `rest_server_requests_total` | Counter | `method`, `path`, `status_code` | REST request count |
| `rest_server_request_duration_seconds` | Histogram | `method`, `path` | REST request latency |
| `batch_request_queue_size` | Gauge | `model`, `version` | Current adaptive batching queue depth |

---

## Codec System

MLServer uses a two-level codec system to convert between V2 tensor format and native Python types:

### Content Type Hints

Set `content_type` in the request or input `parameters` to select a codec:

| Content Type | Codec Class | Python Type |
|-------------|-------------|-------------|
| `np` | `NumpyCodec` | `numpy.ndarray` |
| `pd` | `PandasCodec` | `pandas.DataFrame` |
| `str` | `StringCodec` | `list[str]` |
| `base64` | `Base64Codec` | `list[bytes]` |
| `datetime` | `DatetimeCodec` | `list[datetime]` |

### Codec Levels

- **`InputCodec`** — encodes/decodes a single `RequestInput` ↔ Python object
- **`RequestCodec`** — encodes/decodes an entire `InferenceRequest` ↔ Python object (e.g. DataFrame from multiple inputs)

### Decorator-based Codecs

Use the `@mlserver.codecs.decode_args` decorator on your `predict` method to automatically decode inputs by name:

```python
from mlserver.codecs import decode_args

class MyModel(MLModel):
    @decode_args
    async def predict(self, x: np.ndarray, name: str) -> np.ndarray:
        return self.model.predict(x)
```

---

## Configuration Reference

### Server Settings (`settings.json`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `debug` | bool | `false` | Enable debug mode |
| `host` | string | `0.0.0.0` | Bind address |
| `http_port` | int | `8080` | REST server port |
| `grpc_port` | int | `8081` | gRPC server port |
| `metrics_port` | int | `8082` | Metrics server port |
| `metrics_endpoint` | string | `/metrics` | Prometheus scrape path (null to disable) |
| `parallel_workers` | int | `1` | Number of parallel inference workers |
| `parallel_workers_timeout` | int | `5` | Worker shutdown grace timeout (seconds) |
| `model_repository_root` | string | `.` | Path to model repository |
| `load_models_at_startup` | bool | `true` | Auto-load models on start |
| `strict_readiness` | bool | `true` | Require ALL models ready for server readiness |
| `empty_registry_readiness` | bool | `true` | Report ready when no models are loaded |
| `gzip_enabled` | bool | `true` | Enable GZip compression |
| `log_level` | string | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `access_log` | bool | `true` | Enable REST/gRPC access logging |
| `use_structured_logging` | bool | `false` | Use JSON structured logs |
| `tracing_server` | string | null | OpenTelemetry collector endpoint |
| `cors_settings` | object | null | CORS configuration (see below) |

### CORS Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `allow_origins` | string[] | `[]` | Explicitly allowed CORS origins; the `*` wildcard origin is rejected |
| `allow_credentials` | bool | `false` | Allow cookies in CORS |
| `allow_methods` | string[] | `["GET"]` | Allowed HTTP methods |
| `allow_headers` | string[] | `[]` | Allowed request headers |
| `max_age` | int | `600` | Browser CORS cache TTL (seconds) |

`allow_origin_regex` is not supported. Replace it with an explicit
`allow_origins` list. Wildcard origins are rejected.

### Model Settings (`model-settings.json`)

| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `name` | string | Yes | Model name |
| `implementation` | string | Yes | Runtime class import path |
| `parameters.version` | string | No | Model version |
| `parameters.uri` | string | No | Path to model artifact |
| `parameters.content_type` | string | No | Default codec for this model |
| `max_batch_size` | int | No | Adaptive batching max batch size |
| `max_batch_time` | float | No | Adaptive batching max wait time (seconds) |
| `parallel_workers` | int | No | Override server-level worker count |

### Environment Variables

All settings can be set via environment variables with the `MLSERVER_` prefix:

```bash
MLSERVER_HTTP_PORT=9090
MLSERVER_GRPC_PORT=9091
MLSERVER_DEBUG=true
MLSERVER_MODEL_PARALLEL_WORKERS=4
```

Model-specific settings use the `MLSERVER_MODEL_` prefix.
