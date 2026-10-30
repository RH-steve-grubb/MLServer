# MLServer Security Model

> This document describes MLServer's runtime security architecture, threat
> model, and the controls that protect production deployments.

---

## Overview

MLServer implements a dual-mode security model that balances development
flexibility with production hardening. The mode is determined by the
presence of a trusted runtimes allowlist file.

| Mode | Trigger | Behaviour |
|------|---------|-----------|
| **DEVELOPMENT** | `/etc/mlserver/trusted-runtimes.json` does **not** exist | All valid import paths accepted; dynamic loading enabled |
| **PRODUCTION** | `/etc/mlserver/trusted-runtimes.json` **exists** | Only allowlisted import paths accepted; dynamic loading disabled |

## Threat Model

### Threats Addressed

| Threat | Vector | Control |
|--------|--------|---------|
| Arbitrary code execution | Malicious `implementation` in `model-settings.json` | Import path validation + allowlist enforcement |
| sys.path injection | `model-settings.json` triggers adding model folder to Python path | Disabled in PRODUCTION mode |
| Dependency poisoning | Custom `environment_tarball` contains malicious packages | `environment_tarball` blocked in PRODUCTION mode |
| Environment path hijacking | `environment_path` points to attacker-controlled directory | `environment_path` blocked in PRODUCTION mode |
| CORS misconfiguration | Wildcard `allow_origins` or regex patterns in CORS settings | Wildcard origins rejected in every mode; support for `allow_origin_regex` is removed |

### Out of Scope

- Network-level security (TLS termination, mTLS) — handled by the
  deployment platform (e.g., Kubernetes Ingress, Istio).
- Authentication and authorisation — not built into MLServer; handled by
  sidecar proxies or API gateways.
- Model artifact integrity — MLServer does not verify checksums or
  signatures of model files.

## Trusted Runtimes Allowlist

### File Format

```json
[
  "mlserver_sklearn.SKLearnModel",
  "mlserver_xgboost.XGBoostModel",
  "mlserver_lightgbm.LightGBMModel",
  "mlserver_onnx.OnnxModel"
]
```

> **Note:** The production images ship with these four runtimes only.

The file is a JSON array of canonical Python import paths. Each entry must:

1. Match the pattern `^[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*\.[A-Za-z][A-Za-z0-9_]*$`
2. Have an uppercase-initial final segment (class name).
3. Be the canonical form (built-in aliases are resolved before comparison).

### File Location

The allowlist file must be at `/etc/mlserver/trusted-runtimes.json`. This
path is a constant (`TRUSTED_RUNTIMES_ARTIFACT_PATH` in `settings.py`) and
is not configurable at runtime — it must be baked into the container image.

### Caching

The allowlist is loaded once and cached via `@lru_cache`. To clear the cache
(e.g., in tests), call `clear_trusted_runtime_caches()`.

## Defense in Depth

Security enforcement occurs at multiple layers to prevent bypasses:

### Layer 1: Settings Validation

`ModelSettings.validate_trusted_runtime()` runs as a Pydantic model
validator during settings parsing. Invalid or untrusted import paths are
rejected before the model object is even created.

### Layer 2: Property Access

The `ModelSettings.implementation` property re-validates the import path on
every access. This prevents bypasses through programmatic mutation of the
`implementation_` attribute after validation.

### Layer 3: Server Startup

`MLServer.start()` calls `log_runtime_security_mode()` before starting any
transport servers. If the allowlist file exists but is malformed, startup is
**aborted** with a `RuntimeError` — no endpoints become accessible.

### Layer 4: Model Repository Discovery

The model repository scanner catches validation errors per-model and logs
them without crashing the server. Invalid models are skipped, not loaded.

### Layer 5: CORS Validation

`Settings.validate_no_wildcard_cors()` always blocks wildcard CORS origins.
`allow_origin_regex` is no longer supported; use explicit origins in
`allow_origins` instead.

### Layer 6: Environment Validation

`ModelParameters.validate_no_custom_environments_in_production_mode()` blocks
`environment_tarball` and `environment_path` in PRODUCTION mode.

## Import Path Canonicalization

Built-in runtime plugins have two valid import paths — a verbose form
(e.g., `mlserver_sklearn.sklearn.SKLearnModel`) and a canonical short form
(e.g., `mlserver_sklearn.SKLearnModel`). The
`canonicalize_runtime_import_path()` function maps verbose forms to their
canonical equivalents before any validation occurs.

The set of recognised aliases is defined in
`_BUILTIN_RUNTIME_IMPORT_PATH_ALIASES` in `settings.py`.

## Production Deployment Checklist

1. Build the container image with all required runtime packages
   pre-installed.
2. Create `/etc/mlserver/trusted-runtimes.json` listing only the runtime
   import paths that should be loadable.
3. Do **not** set `environment_tarball` or `environment_path` in any
   `model-settings.json`.
4. Configure only explicit CORS origins in `allow_origins`. Do **not** use
   wildcard origins (`['*']`); `allow_origin_regex` is no longer supported.
5. Verify at startup that the log shows `Runtime security: PRODUCTION`.

## Logging

| Log Message | Meaning |
|-------------|---------|
| `Runtime security: DEVELOPMENT` | No allowlist file found; all runtimes allowed |
| `Runtime security: PRODUCTION - N model implementations allowed` | Allowlist loaded with N entries |
| `Trusted runtimes allowlist file exists but is empty` | WARNING — no models can be loaded |
| `Rejected untrusted model implementation 'X'` | Model X was blocked by the allowlist |
| `Failed to load trusted runtimes allowlist!` | Allowlist file is malformed; startup aborted |
