# Testing with Podman Instead of Docker

This guide explains how to run MLServer's Docker-dependent tests for ODH supported functionality using Podman as a Docker replacement.

## Prerequisites

- Podman installed on your system
- No Docker installation required

## Setup Steps

### 1. Enable Podman Socket

Enable and start the Podman socket to provide a Docker-compatible API:

```bash
systemctl --user enable --now podman.socket
```

Verify it's running:

```bash
systemctl --user status podman.socket
```

### 2. Create Docker Symlink

Create a symlink so that commands calling `docker` will use `podman` instead:

```bash
ln -sf $(which podman) ~/.local/bin/docker
```

Verify the symlink works:

```bash
which docker
docker --version
```

You should see output indicating Podman version 5.6.0 or similar.

### 3. Set DOCKER_HOST Environment Variable

Export the `DOCKER_HOST` variable to point to the Podman socket:

```bash
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

To make this permanent, add it to your shell configuration file (`~/.bashrc` or `~/.zshrc`):

```bash
echo 'export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock' >> ~/.bashrc
source ~/.bashrc
```

## Running Tests

Once the setup is complete, you can run Docker-dependent tests normally:

### Kafka Tests

```bash
poetry run python -m pytest tests/kafka/
```

### All Tests

```bash
poetry run tox -e odh-mlserver
poetry run tox -e odh-all-runtimes
```

## Troubleshooting

### Issue: "Error while fetching server API version"

**Cause**: The `DOCKER_HOST` environment variable is not set.

**Solution**: Ensure you've exported `DOCKER_HOST` as described in step 3.

### Issue: "short-name resolution enforced but cannot prompt without a TTY"

**Cause**: Podman's registry configuration requires interactive prompts.

**Solution**: This should be resolved by the system-wide `/etc/containers/registries.conf` configuration. If the issue persists, create `~/.config/containers/registries.conf`:

```ini
short-name-mode = "permissive"
unqualified-search-registries = ["docker.io"]
```

## Technical Details

### Why These Steps Are Needed

1. **Podman Socket**: Provides a Docker-compatible API endpoint that the Python `docker` library can connect to
2. **Docker Symlink**: Allows subprocess calls to `docker` CLI commands (like `docker build`) to work with Podman
3. **DOCKER_HOST Variable**: Tells Docker clients where to find the API endpoint; must be passed through tox for test isolation
4. **tox.ini passenv**: Ensures the `DOCKER_HOST` variable is available in the test environment

### What Was NOT Changed

All modifications are environmental - no test code or application code needs to be modified.
The changes included:

- System configuration (Podman socket, symlinks, environment variables)
- Build configuration (`tox.ini` to pass environment variables)

This approach maintains compatibility with both Docker and Podman environments.
