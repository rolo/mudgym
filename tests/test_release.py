"""Exercise release preparation in disposable repositories, without committing or pushing."""

import importlib.util
import shutil
import subprocess
import tomllib
from datetime import date
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("release", REPOSITORY_ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_citation_update_preserves_format_version_and_references(tmp_path):
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        'cff-version: 1.2.0\n# Keep this comment.\nversion: "0.4.6"\ndate-released: "2026-09-06"\n'
        'references:\n  - type: software\n    title: "MUD2"\n    version: "4.0"\n'
        '    date-released: "2020-01-01"\n'
    )

    release.write_citation_version(citation, "0.4.7rc1", date(2026, 9, 7))

    assert citation.read_text() == (
        'cff-version: 1.2.0\n# Keep this comment.\nversion: "0.4.7rc1"\ndate-released: "2026-09-07"\n'
        'references:\n  - type: software\n    title: "MUD2"\n    version: "4.0"\n'
        '    date-released: "2020-01-01"\n'
    )


@pytest.mark.parametrize("field", ["version", "date-released"])
@pytest.mark.parametrize("count", [0, 2])
def test_citation_update_refuses_missing_or_duplicate_release_fields(tmp_path, field, count):
    citation = tmp_path / "CITATION.cff"
    original = 'version: "0.4.6"\ndate-released: "2026-09-06"\n'
    original = "".join(line * count if line.startswith(f"{field}:") else line for line in original.splitlines(True))
    citation.write_text(original)

    with pytest.raises(SystemExit, match=f"exactly one top-level {field} field"):
        release.write_citation_version(citation, "0.4.7", date(2026, 9, 7))

    assert citation.read_text() == original


@pytest.fixture
def release_repository(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(REPOSITORY_ROOT / "scripts" / "release.py", scripts / "release.py")
    (tmp_path / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nmessage: "Please cite this software."\ntitle: "Release test"\n'
        'type: software\nauthors:\n  - name: "Test author"\n'
        'version: "0.4.6"\ndate-released: "2026-09-06"\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "release-test"\nversion = "0.4.6"\nrequires-python = ">=3.13"\n'
    )
    for command in (
        ["uvx", "cffconvert", "--validate"],
        ["uv", "lock"],
        ["git", "init", "--quiet"],
        ["git", "add", "--", *release.VERSIONED_FILES],
    ):
        subprocess.run(command, cwd=tmp_path, check=True)

    spec = importlib.util.spec_from_file_location("temporary_release", scripts / "release.py")
    temporary_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temporary_release)
    return temporary_release


def test_write_version_updates_package_lock_and_citation(release_repository):
    release_repository.write_version("0.4.7rc1")

    root = release_repository.REPOSITORY_ROOT
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert project["project"]["version"] == "0.4.7rc1"
    assert lock["package"][0]["version"] == "0.4.7rc1"
    citation = (root / "CITATION.cff").read_text()
    assert '\nversion: "0.4.7rc1"\n' in citation
    assert f'\ndate-released: "{date.today().isoformat()}"\n' in citation


def test_write_version_rejects_invalid_citation(release_repository, capfd):
    citation = release_repository.REPOSITORY_ROOT / "CITATION.cff"
    citation.write_text(citation.read_text().replace("type: software", "type: invalid"))

    with pytest.raises(subprocess.CalledProcessError) as error:
        release_repository.write_version("0.4.7")

    assert error.value.cmd == ["uvx", "cffconvert", "--validate"]
    captured = capfd.readouterr()
    assert "'invalid' is not one of" in captured.err
    assert "On instance['type']:" in captured.err
