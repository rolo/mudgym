"""Check every docs notebook gets a browser version, built from the recorded notebook."""

import importlib.util
import json
from pathlib import Path

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
    for helper in docs_jupyterlite.NOTEBOOKS_DIR.glob("*.py"):
        assert (tmp_path / "files" / helper.name).read_bytes() == helper.read_bytes()
