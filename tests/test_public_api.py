import importlib
import re
from pathlib import Path

import pytest

API_PAGE = Path(__file__).resolve().parent.parent / "docs" / "pages" / "api.md"
API_PAGE_TARGETS = sorted(set(re.findall(r"^::: ([\w.]+)$", API_PAGE.read_text(), flags=re.MULTILINE)))


def test_the_api_page_documents_something():
    assert API_PAGE_TARGETS, f"no ::: targets found in {API_PAGE}, so the pattern or the page has changed"


@pytest.mark.parametrize("target", API_PAGE_TARGETS)
def test_every_api_page_target_still_exists(target):
    """Fail quickly when an API rename or removal leaves a stale mkdocstrings target."""
    module_path, _, name = target.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        # A class member has one more attribute segment than a module-level target.
        parent_path, _, class_name = module_path.rpartition(".")
        module = importlib.import_module(parent_path)
        assert hasattr(getattr(module, class_name), name), f"{target} is documented but no longer exists"
        return
    assert hasattr(module, name), f"{target} is documented but no longer exists"
