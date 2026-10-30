# ADR-0005: Runtime Security — DEVELOPMENT and PRODUCTION Modes

## Status

Accepted

## Context

MLServer supports loading arbitrary Python classes as model runtimes via the
`implementation` field in `model-settings.json`.  In development this
flexibility is valuable, but in production deployments (e.g., inside
container images served by KServe/RHOAI) it presents security risks:

- **Arbitrary code execution** — a malicious `model-settings.json` could
  point `implementation` at any importable Python class.
- **sys.path injection** — development mode temporarily adds the model folder
  to `sys.path`, enabling loading of code from arbitrary filesystem paths.
- **Environment tarballs** — custom environments extracted from tarballs
  could contain malicious dependencies.
- **CORS misconfiguration** — wildcard CORS origins allow any
  website to make cross-origin requests.

## Decision

Implement a dual-mode security model controlled by the presence of a trusted
runtimes allowlist file at `/etc/mlserver/trusted-runtimes.json`:

### DEVELOPMENT Mode (file absent)

- All model implementations with valid import path syntax are allowed.
- `sys.path` injection and dynamic module reload are enabled.
- Custom `environment_tarball` and `environment_path` are allowed.

### PRODUCTION Mode (file present)

- Only model implementations listed in the JSON allowlist are permitted.
- `sys.path` injection and dynamic module reload are disabled.
- `environment_tarball` and `environment_path` are blocked.

### CORS (both modes)

When CORS is configured, `Settings.validate_no_wildcard_cors()` rejects the
`*` wildcard origin in both modes. `allow_origin_regex` is not supported;
configure explicit origins with `allow_origins` instead.

### Defense in Depth

Security is enforced at multiple layers:

1. **Settings parse time** — `ModelSettings.validate_trusted_runtime()`.
2. **Property access time** — `ModelSettings.implementation` getter
   re-validates on every access.
3. **Server startup** — `log_runtime_security_mode()` validates the
   allowlist before any endpoint becomes accessible.
4. **Model repository discovery** — invalid entries are skipped without
   crashing the server.

### Import Path Validation

All runtime import paths are validated against a canonical regex pattern
that requires dotted module segments and an uppercase-initial class name.
Built-in runtime aliases are canonicalized before validation.

## Consequences

**Benefits:**
- Container images can ship a baked-in allowlist, locking down which
  runtimes are loadable — no runtime configuration changes can bypass it.
- Zero-config for development: the absence of the file enables flexibility.
- Defense-in-depth prevents bypasses through programmatic mutation of
  `implementation_`.
- The allowlist is cached with `@lru_cache` for performance.

**Trade-offs:**
- The mode is determined by file existence, not an explicit configuration
  flag — this can be surprising if the file is accidentally created or
  deleted.
- The allowlist cache must be explicitly cleared
  (`clear_trusted_runtime_caches()`) for test isolation.
- Custom runtimes in production require rebuilding the container image with
  an updated allowlist.
