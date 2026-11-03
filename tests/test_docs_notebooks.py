"""Check every documentation notebook is fully recorded."""

import importlib.util
from pathlib import Path

import nbformat
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("docs_notebooks", REPOSITORY_ROOT / "docs" / "notebooks.py")
docs_notebooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs_notebooks)

NOTEBOOK_PATHS = docs_notebooks.notebook_paths([])


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda path: path.stem)
def test_notebook_is_recorded_without_errors(path):
    code_cells = [cell for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == "code"]

    unrecorded = [cell.id for cell in code_cells if cell.execution_count is None]
    assert not unrecorded, (
        f"{path.name} has cells that never ran: {unrecorded}. Run `just docs-notebooks {path.stem}` and commit it."
    )
    failed = [cell.id for cell in code_cells if any(output.output_type == "error" for output in cell.outputs)]
    assert not failed, f"{path.name} recorded errors in cells {failed}."
