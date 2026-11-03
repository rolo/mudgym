"""Check every docs notebook gets a browser version, built from the recorded notebook and its overlay."""

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("docs_jupyterlite", REPOSITORY_ROOT / "docs" / "jupyterlite.py")
docs_jupyterlite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs_jupyterlite)


def test_every_docs_notebook_gets_a_browser_version_ready_to_run(tmp_path):
    docs_jupyterlite.write_browser_notebooks(tmp_path / "files")

    for path in sorted(docs_jupyterlite.NOTEBOOKS_DIR.glob("*.ipynb")):
        notebook = json.loads((tmp_path / "files" / path.name).read_text())
        assert notebook["metadata"]["kernelspec"] == docs_jupyterlite.PYODIDE_KERNELSPEC
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells), path.name
        overlay_path = docs_jupyterlite.OVERLAYS_DIR / path.name
        overlay_cells = json.loads(overlay_path.read_text())["cells"] if overlay_path.is_file() else []
        sources = {cell["id"]: cell["source"] for cell in notebook["cells"]}
        for cell in overlay_cells:
            assert sources[cell["id"]] == cell["source"], f"{path.name} cell {cell['id']}"
    for helper in docs_jupyterlite.NOTEBOOKS_DIR.glob("*.py"):
        assert (tmp_path / "files" / helper.name).read_bytes() == helper.read_bytes()


def test_an_overlay_naming_a_cell_its_notebook_lacks_is_refused():
    source = {"cells": [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "# Quickstart"}]}
    overlay = {"cells": [{"cell_type": "markdown", "id": "renamed-intro", "metadata": {}, "source": "# Quickstart"}]}

    with pytest.raises(ValueError, match="renamed-intro"):
        docs_jupyterlite.browser_notebook(source, overlay)
