"""Exercise release preparation with disposable repositories and local services."""

import importlib.util
import shutil
import subprocess
import tomllib
from contextlib import nullcontext
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

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


@pytest.mark.parametrize(("field", "count"), [("version", 0), ("date-released", 2)])
def test_citation_update_refuses_missing_or_duplicate_release_fields(tmp_path, field, count):
    citation = tmp_path / "CITATION.cff"
    original = 'version: "0.4.6"\ndate-released: "2026-09-06"\n'
    original = "".join(line * count if line.startswith(f"{field}:") else line for line in original.splitlines(True))
    citation.write_text(original)

    with pytest.raises(SystemExit, match=f"exactly one top-level {field} field"):
        release.write_citation_version(citation, "0.4.7", date(2026, 9, 7))

    assert citation.read_text() == original


@pytest.fixture
def release_script(tmp_path):
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ROOT / "scripts" / "release.py", scripts / "release.py")
    spec = importlib.util.spec_from_file_location("temporary_release", scripts / "release.py")
    temporary_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temporary_release)
    return temporary_release


@pytest.fixture
def release_repository(release_script):
    root = release_script.REPOSITORY_ROOT
    (root / "CITATION.cff").write_text(
        'cff-version: 1.2.0\nmessage: "Please cite this software."\ntitle: "Release test"\n'
        'type: software\nauthors:\n  - name: "Test author"\n'
        'version: "0.4.6"\ndate-released: "2026-09-06"\n'
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "release-test"\nversion = "0.4.6"\nrequires-python = ">=3.13"\n'
    )
    for command in (
        ["uv", "lock"],
        ["git", "init", "--quiet"],
        ["git", "add", "--", *release.VERSIONED_FILES],
    ):
        release_script.run(command)
    return release_script


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


def test_write_version_rejects_invalid_citation(release_repository):
    citation = release_repository.REPOSITORY_ROOT / "CITATION.cff"
    citation.write_text(citation.read_text().replace("type: software", "type: invalid"))

    with pytest.raises(subprocess.CalledProcessError):
        release_repository.write_version("0.4.7")


@pytest.fixture
def package_index(request):
    response_status = request.param

    class PackageHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/pypi/release-test/0.4.6/json"
            self.send_response(response_status)
            self.end_headers()
            self.wfile.write(b"{}")

    with ThreadingHTTPServer(("127.0.0.1", 0), PackageHandler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/pypi"
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    ("package_index", "error"),
    [(404, None), (200, "already published"), (503, "Cannot check")],
    indirect=["package_index"],
)
def test_retry_requires_an_unpublished_version(package_index, error):
    with pytest.raises(SystemExit, match=error) if error else nullcontext():
        release.refuse_if_published("release-test", "0.4.6", index_url=package_index)


@pytest.fixture
def failed_release_run():
    return {
        "id": 1,
        "head_branch": "v0.4.6",
        "head_sha": "current-commit",
        "event": "push",
        "status": "completed",
        "conclusion": "failure",
    }


def test_retry_accepts_a_failed_tag_push_run(failed_release_run):
    release.refuse_unless_release_failed([failed_release_run], "v0.4.6", "current-commit")


@pytest.mark.parametrize(
    ("status", "conclusion", "commit"),
    [("in_progress", None, "current-commit"), ("completed", "success", "earlier-commit")],
)
def test_retry_refuses_active_or_successful_release_runs(failed_release_run, status, conclusion, commit):
    with pytest.raises(SystemExit, match="has not failed"):
        release.refuse_unless_release_failed(
            [
                {**failed_release_run, "status": status, "conclusion": conclusion, "head_sha": commit},
                {**failed_release_run, "id": 2},
            ],
            "v0.4.6",
            "current-commit",
        )


@pytest.mark.parametrize(
    "changes", [{"head_branch": "main"}, {"event": "workflow_dispatch"}, {"head_sha": "different"}]
)
def test_retry_requires_a_release_run_for_the_tagged_commit(failed_release_run, changes):
    with pytest.raises(SystemExit, match="No release run"):
        release.refuse_unless_release_failed(
            [{**failed_release_run, **changes}],
            "v0.4.6",
            "current-commit",
        )


@pytest.fixture
def retry_repository(release_script, tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    for command in (
        ["git", "init", "--quiet", "--initial-branch=main"],
        ["git", "config", "core.hooksPath", "/dev/null"],
        ["git", "config", "user.name", "Release test"],
        ["git", "config", "user.email", "release-test@example.com"],
        ["git", "config", "commit.gpgSign", "false"],
        ["git", "config", "tag.gpgSign", "false"],
        ["git", "add", "scripts"],
        ["git", "commit", "--quiet", "-m", "Initial release."],
        ["git", "remote", "add", "origin", str(remote)],
        ["git", "tag", "-a", "v0.4.6", "-m", "Release 0.4.6"],
        ["git", "push", "--quiet", "origin", "main", "refs/tags/v0.4.6"],
    ):
        release_script.run(command)
    release_script.run(["git", "commit", "--quiet", "--allow-empty", "-m", "Fix release collection."])
    release_script.run(["git", "push", "--quiet", "origin", "main"])
    return release_script


def test_retry_reads_the_remote_tag_even_when_the_local_tag_has_moved(retry_repository):
    old_tag = retry_repository.read(["git", "rev-parse", "refs/tags/v0.4.6"])
    old_commit = retry_repository.read(["git", "rev-parse", "v0.4.6^{commit}"])
    retry_repository.run(["git", "tag", "-f", "-a", "v0.4.6", "-m", "An earlier retry."])

    assert (old_tag, old_commit) == retry_repository.read_remote_tag("v0.4.6")


def test_retry_pushes_only_the_selected_tag_at_the_checked_commit(retry_repository):
    old_tag = retry_repository.read(["git", "rev-parse", "refs/tags/v0.4.6"])
    checked_commit = retry_repository.read(["git", "rev-parse", "HEAD"])
    retry_repository.run(["git", "config", "push.followTags", "true"])
    retry_repository.run(["git", "tag", "-a", "unrelated", "-m", "Keep this tag local."])
    retry_repository.run(["git", "commit", "--quiet", "--allow-empty", "-m", "Another change during validation."])

    retry_repository.replace_release_tag("0.4.6", "v0.4.6", old_tag, checked_commit)

    assert checked_commit == retry_repository.read(["git", "ls-remote", "origin", "refs/tags/v0.4.6^{}"]).split()[0]
    assert "" == retry_repository.read(["git", "ls-remote", "origin", "refs/tags/unrelated"])


def test_retry_refuses_to_overwrite_a_remote_tag_changed_since_the_checks(retry_repository):
    old_tag = retry_repository.read(["git", "rev-parse", "refs/tags/v0.4.6"])
    head = retry_repository.read(["git", "rev-parse", "HEAD"])
    retry_repository.run(["git", "tag", "-a", "intervening", "-m", "Another release attempt."])
    intervening_tag = retry_repository.read(["git", "rev-parse", "refs/tags/intervening"])
    retry_repository.run(["git", "push", "--force", "origin", "refs/tags/intervening:refs/tags/v0.4.6"])

    with pytest.raises(subprocess.CalledProcessError):
        retry_repository.replace_release_tag("0.4.6", "v0.4.6", old_tag, head)

    assert intervening_tag == retry_repository.read(["git", "ls-remote", "origin", "refs/tags/v0.4.6"]).split()[0]


def test_retry_refuses_to_change_the_package_version(release_repository):
    with pytest.raises(SystemExit, match="current project version 0.4.6"):
        release_repository.retry_release("0.4.7", "v0.4.7")
