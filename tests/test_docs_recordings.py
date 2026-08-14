"""Replays every recording example in docs/record.py over its committed capture -- no Docker, no live
game -- and derives the fragments into a temporary directory. A mismatch against the committed
fragments means someone edited an example or a fragment without re-running `just docs-derive`
(presentation drift), and a replay divergence means the example's commands no longer match their
capture and `just docs-record` is due.
"""

import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("record_docs", REPOSITORY_ROOT / "docs" / "record.py")
record_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_docs)


@pytest.mark.parametrize("name", list(record_docs.EXAMPLES))
def test_committed_fragments_match_their_captures(name, tmp_path):
    assert record_docs.capture_paths(name), (
        f"No committed connection capture for {name!r}. Captures are the source of truth for the docs "
        f"fragments; restore the deleted file or run `just docs-record {name}` and commit it."
    )

    record_docs.derive(name, fragments_dir=tmp_path)

    derived_names = sorted(path.name for path in tmp_path.iterdir())
    assert derived_names, f"Deriving {name!r} produced no fragments."

    committed_directory = record_docs.RECORDINGS_DIR
    committed_names = sorted(path.name for path in record_docs.fragment_paths(name, committed_directory))
    assert committed_names == derived_names, (
        f"Committed fragments for {name!r} are {committed_names} but its captures derive {derived_names}; "
        f"run `just docs-derive {name}` and commit the change."
    )
    for fragment_name in derived_names:
        committed = (committed_directory / fragment_name).read_bytes()
        derived = (tmp_path / fragment_name).read_bytes()
        assert committed == derived, (
            f"docs/recordings/{fragment_name} does not match what {name!r}'s capture derives; "
            f"run `just docs-derive {name}` and commit the change."
        )
