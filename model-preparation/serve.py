"""Launch the prepared CPU or CUDA repository with the serving-env Python."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def serving_environment() -> dict[str, str]:
    environment = os.environ.copy()
    nvidia = importlib.util.find_spec("nvidia")
    libraries: list[str] = []
    if nvidia is not None:
        for location in nvidia.submodule_search_locations or []:
            libraries.extend(str(path) for path in sorted(Path(location).glob("*/lib")))
    existing = environment.get("LD_LIBRARY_PATH")
    if existing:
        libraries.append(existing)
    if libraries:
        # Spawned inference workers need these paths too; preloading only in the
        # frontend would not initialize the CUDA libraries in a fresh worker.
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    repository = args.bundle.resolve() / "repositories" / args.device
    if not (repository / "rfdetr-small" / "model-settings.json").is_file():
        raise SystemExit(f"Missing prepared model repository: {repository}")
    command = [
        sys.executable,
        "-c",
        "from mlserver.cli.main import root; root()",
        "start",
        str(repository),
    ]
    os.execve(sys.executable, command, serving_environment())


if __name__ == "__main__":
    main()
