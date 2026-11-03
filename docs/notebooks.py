"""Record the docs notebooks against the live game with `just docs-notebooks [names]`.

The notebooks are committed with their outputs. The docs link straight to their JupyterLite versions.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPOSITORY_ROOT / "docs" / "notebooks"
KERNEL_NAME = "python3"
CELL_TIMEOUT_SECONDS = 1800


def notebook_paths(names: list[str]) -> list[Path]:
    available = {path.stem: path for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb"))}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise SystemExit(f"Unknown notebooks {unknown}. Choose from {sorted(available)}.")
    return [available[name] for name in names] if names else list(available.values())


def execute(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    # Anything the notebook writes stays out of the repo.
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}-") as working_directory:
        for helper in NOTEBOOKS_DIR.glob("*.py"):
            shutil.copy(helper, working_directory)
        client = NotebookClient(
            notebook,
            kernel_name=KERNEL_NAME,
            timeout=CELL_TIMEOUT_SECONDS,
            coalesce_streams=True,
            record_timing=False,
            resources={"metadata": {"path": working_directory}},
        )
        client.execute()
    # Run timestamps, ours or JupyterLab's, would change every cell every time.
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.write(notebook, path)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Record the documentation notebooks.")
    parser.add_argument("names", nargs="*", help="Notebook names without .ipynb, defaulting to all of them.")
    arguments = parser.parse_args(argv)
    for path in notebook_paths(arguments.names):
        print(f"executing docs/notebooks/{path.name}")
        execute(path)


if __name__ == "__main__":
    main(sys.argv[1:])
