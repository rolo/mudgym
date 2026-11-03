"""Check every documentation notebook is fully recorded and derives its page without starting a live game."""

import importlib.util
from pathlib import Path

import nbformat
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("docs_notebooks", REPOSITORY_ROOT / "docs" / "notebooks.py")
docs_notebooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs_notebooks)

NOTEBOOK_PATHS = docs_notebooks.notebook_paths([])
HELPER_PATHS = sorted(docs_notebooks.NOTEBOOKS_DIR.glob("*.py"))


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda path: path.stem)
def test_notebook_is_recorded_without_errors(path):
    code_cells = [cell for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == "code"]

    unrecorded = [cell.id for cell in code_cells if cell.execution_count is None]
    assert not unrecorded, (
        f"{path.name} has cells that never ran: {unrecorded}. Run `just docs-notebooks {path.stem}` and commit it."
    )
    failed = [cell.id for cell in code_cells if any(output.output_type == "error" for output in cell.outputs)]
    assert not failed, f"{path.name} recorded errors in cells {failed}."


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda path: path.stem)
def test_notebook_derives_a_page_linking_its_browser_version_and_a_download(path, tmp_path):
    docs_notebooks.derive([path], pages_dir=tmp_path)

    page = (tmp_path / f"{path.stem}.md").read_text()
    browser_version = f"(../lite/notebooks/index.html?path={path.name})"
    assert f"[Run it in your browser]{browser_version}" in page
    assert f"[download {path.name}]({path.name})" in page
    # Links to sibling notebooks open their pages on the site.
    assert ".ipynb)" not in page.replace(f"({path.name})", "").replace(browser_version, "")
    assert (tmp_path / path.name).read_bytes() == path.read_bytes()
    for helper in HELPER_PATHS:
        assert (tmp_path / helper.name).read_bytes() == helper.read_bytes()


def test_deriving_removes_what_a_removed_notebook_left_and_keeps_the_other_pages(tmp_path):
    derived, other = NOTEBOOK_PATHS[:2]
    for name in ("retired.md", "retired.ipynb", f"{other.stem}.md"):
        (tmp_path / name).write_text("left from an earlier run")
    (tmp_path / "retired_files").mkdir()

    docs_notebooks.derive([derived], pages_dir=tmp_path)

    names = {entry.name for entry in tmp_path.iterdir()}
    assert not names & {"retired.md", "retired.ipynb", "retired_files"}
    assert {f"{derived.stem}.md", f"{other.stem}.md"} | {helper.name for helper in HELPER_PATHS} <= names
