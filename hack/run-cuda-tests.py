"""Start pytest with Poetry-installed NVIDIA libraries on the loader path."""

import importlib.util
import os
from pathlib import Path
import sys


def main() -> None:
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
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    # glibc reads LD_LIBRARY_PATH at process startup. Updating it from a pytest
    # hook is too late for ORT's native provider-library dependencies.
    os.execve(
        sys.executable,
        [sys.executable, "-m", "pytest", *sys.argv[1:]],
        environment,
    )


if __name__ == "__main__":
    main()
