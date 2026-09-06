"""Replay every documentation capture and derive its ignored fragments without Docker."""

import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("record_docs", REPOSITORY_ROOT / "docs" / "record.py")
record_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_docs)


@pytest.mark.parametrize("name", list(record_docs.EXAMPLES))
def test_capture_replays_and_derives_fragments(name, tmp_path):
    assert record_docs.capture_paths(name), (
        f"No committed connection capture for {name!r}. Captures are the source of truth for the docs "
        f"fragments; restore the deleted file or run `just docs-record {name}` and commit it."
    )

    record_docs.derive(name, fragments_dir=tmp_path)

    derived_names = sorted(path.name for path in tmp_path.iterdir())
    assert derived_names, f"Deriving {name!r} produced no fragments."
