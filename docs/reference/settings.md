# MLServer Settings

MLServer can be configured through a `settings.json` file on the root folder
from where MLServer is started.
Note that these are server-wide settings (e.g. gRPC or HTTP port) which are
separate from the [invidual model settings](./model-settings).
Alternatively, this configuration can also be passed through **environment
variables** prefixed with `MLSERVER_` (e.g. `MLSERVER_GRPC_PORT`).
By default, debug logging is disabled (`"debug": false`) and can be enabled with
`MLSERVER_DEBUG=1` or by setting `"debug": true` in `settings.json`.
The application log level defaults to `INFO` and can be changed with
`MLSERVER_LOG_LEVEL` (e.g. `MLSERVER_LOG_LEVEL=WARNING`).
Valid values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` (case-insensitive).
Unrecognised values are normalised to `INFO` with a Python `UserWarning` at startup.
When `debug` is enabled the log level is always `DEBUG`, FastAPI debug mode is
enabled on the inference and metrics servers, and health-check endpoints are
included in access logs.
Access logging for REST and gRPC requests is enabled by default (`"access_log": true`)
and can be disabled with `MLSERVER_ACCESS_LOG=0` or by setting `"access_log": false`
in `settings.json`. Access logging is independent of `debug`. When enabled,
successful health-check GETs are filtered unless `debug` is also enabled; failed
health-check responses are always logged.

The [Inference Performance Options guide](../user-guide/performance.md) explains
how to enable worker REST JSON, compact raw FP32 gRPC inputs and the optional
ZeroMQ inference transport, including their requirements and defaults.

## Settings

```{eval-rst}

.. autopydantic_settings:: mlserver.settings.Settings
```
