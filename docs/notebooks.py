"""Record the documentation notebooks against the live game and derive the pages the docs display.

The notebooks in `docs/notebooks/` are committed with their outputs, which are the recording. Each runs in two
phases, like the examples in `docs/record.py`:

- execute: the notebook runs top to bottom in a local kernel against the live game, and its outputs are written
  back into the committed file. This needs the `notebooks` dependency group.
- derive: each recorded notebook is converted to a page in `docs/pages/notebooks/`, beside a copy of the `.ipynb`
  for download, and the page links to the notebook's browser version, which `docs/jupyterlite.py` builds. The
  derived pages are ignored by Git and CI only derives, so the docs build never runs the game.

Run through the justfile:

    just docs-notebooks                  # re-run every notebook against the live game, then derive its page
    just docs-notebooks goal_room_dqn    # ... for a subset by name
    just docs-derive                     # derive the pages from the committed notebooks without a live game
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import MarkdownExporter

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPOSITORY_ROOT / "docs" / "notebooks"
PAGES_DIR = REPOSITORY_ROOT / "docs" / "pages" / "notebooks"
KERNEL_NAME = "python3"
CELL_TIMEOUT_SECONDS = 1800


def notebook_paths(names: list[str]) -> list[Path]:
    available = {path.stem: path for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb"))}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise SystemExit(f"Unknown notebooks {unknown}. Choose from {sorted(available)}.")
    return [available[name] for name in names] if names else list(available.values())


def execute(path: Path) -> None:
    """Run a notebook against the live game and write its outputs back into the committed file."""
    notebook = nbformat.read(path, as_version=4)
    # A scratch working directory keeps files the notebook writes, such as saved weights, out of the repository.
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}-") as working_directory:
        client = NotebookClient(
            notebook,
            kernel_name=KERNEL_NAME,
            timeout=CELL_TIMEOUT_SECONDS,
            coalesce_streams=True,
            # Timestamps would change every cell on every recording.
            record_timing=False,
            resources={"metadata": {"path": working_directory}},
        )
        client.execute()
    # JupyterLab saves when each cell last ran, which would change every cell on every edit.
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.write(notebook, path)


def page_markdown(path: Path, sibling_names: set[str]) -> tuple[str, dict[str, bytes]]:
    """Convert a recorded notebook to page Markdown and the image files its outputs refer to."""
    notebook = nbformat.read(path, as_version=4)
    body, resources = MarkdownExporter().from_notebook_node(
        notebook, resources={"metadata": {"name": path.stem}, "output_files_dir": f"{path.stem}_files"}
    )
    # Links between notebooks open the sibling's page on the site, and the downloaded file keeps its own links.
    for name in sibling_names:
        body = body.replace(f"]({name}.ipynb)", f"]({name}.md)")
    title, separator, rest = body.partition("\n")
    if not title.startswith("# "):
        raise ValueError(f"{path.name} must open with a level-one heading")
    run_in_browser = f"[Run it in your browser](../lite/notebooks/index.html?path={path.name})"
    links = f"{run_in_browser} with nothing to install, or [download {path.name}]({path.name}) to run it in your own Jupyter."
    return f"{title}{separator}\n{links}\n{rest}", resources.get("outputs", {})


def derive(paths: list[Path], pages_dir: Path = PAGES_DIR) -> None:
    """Write a page and a downloadable copy for each recorded notebook."""
    sibling_names = {path.stem for path in NOTEBOOKS_DIR.glob("*.ipynb")}
    pages_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.rmtree(pages_dir / f"{path.stem}_files", ignore_errors=True)
        markdown, outputs = page_markdown(path, sibling_names)
        (pages_dir / f"{path.stem}.md").write_text(markdown, encoding="utf-8")
        for relative_name, content in outputs.items():
            output_path = pages_dir / relative_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
        shutil.copyfile(path, pages_dir / path.name)
        print(f"derived {path.stem}.md")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Record and derive the documentation notebooks.")
    parser.add_argument("names", nargs="*", help="Notebook names without .ipynb, defaulting to all of them.")
    parser.add_argument("--execute", action="store_true", help="Re-run the notebooks against the live game first.")
    arguments = parser.parse_args(argv)
    paths = notebook_paths(arguments.names)
    if arguments.execute:
        for path in paths:
            print(f"executing docs/notebooks/{path.name}")
            execute(path)
    derive(paths)


if __name__ == "__main__":
    main(sys.argv[1:])
