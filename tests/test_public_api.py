import importlib
import re
from pathlib import Path

import pytest

from mudgym import make_env, make_parallel_env
from mudgym.actions import DIRECTIONS, direction_index
from mudgym.envs.factory import make_env as factory_make_env
from mudgym.envs.factory import make_parallel_env as factory_make_parallel_env

API_PAGE = Path(__file__).resolve().parent.parent / "docs" / "pages" / "api.md"
API_PAGE_TARGETS = sorted(set(re.findall(r"^::: ([\w.]+)$", API_PAGE.read_text(), flags=re.MULTILINE)))


def test_the_api_page_documents_something():
    assert API_PAGE_TARGETS, f"no ::: targets found in {API_PAGE}; the pattern or the page has changed"


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


def test_environment_factories_are_available_from_the_public_package():
    assert make_env is factory_make_env
    assert make_parallel_env is factory_make_parallel_env


def test_direction_index_matches_the_public_direction_order():
    assert [direction_index(direction) for direction in DIRECTIONS] == list(range(len(DIRECTIONS)))
