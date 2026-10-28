"""Use the production CLI with the prepared CUDA library paths."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "model-preparation/serve.py"
    spec = importlib.util.spec_from_file_location("prepared_serve", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CUDA environment helper: {source}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    command = [
        sys.executable,
        "-c",
        "from mlserver.cli.main import main; main()",
        "start",
        str(args.repository.resolve()),
    ]
    os.execve(sys.executable, command, helper.serving_environment())


if __name__ == "__main__":
    main()
