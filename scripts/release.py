"""Validate, version, commit, tag, and push a release. GitHub Actions publishes it via OIDC.

Use `just release VERSION` for a new release or `just release-retry VERSION` after fixing a failed release.
"""

import argparse
import json
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RELEASE_BRANCH = "main"
VERSIONED_FILES = ["CITATION.cff", "pyproject.toml", "uv.lock"]
LOCAL_GATES = ["lint", "format-check", "test", "check-dist"]


def run(command: list[str]) -> subprocess.CompletedProcess:
    """Run a command that must succeed, letting its output through to the terminal."""
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def read(command: list[str]) -> str:
    """Run a command that must succeed and return its stripped stdout.

    Its output is captured, so a failure has to hand the tool's own message back rather than
    die with a traceback that hides the one line explaining what went wrong.
    """
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise SystemExit(f"`{' '.join(command)}` failed:\n{completed.stderr.strip()}")
    return completed.stdout.strip()


def succeeds(command: list[str]) -> bool:
    """Report whether a command exited zero, discarding its output."""
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True)
    return completed.returncode == 0


def canonical_version(requested_version: str) -> str:
    """Return the PEP 440 spelling uv would write, refusing anything else.

    The release workflow asserts that the tag equals the built metadata version, so a
    non-canonical spelling here becomes a failed publish after the tag is already pushed.
    """
    if requested_version.startswith("v"):
        raise SystemExit("Pass the package version without the v prefix (for example, 0.4.0).")

    version = read(["uv", "version", requested_version, "--dry-run", "--short"])
    if version != requested_version:
        raise SystemExit(f"Use the canonical package version: {version}")
    return version


def refuse_unless_on_a_clean_release_branch() -> None:
    branch = read(["git", "branch", "--show-current"])
    if branch != RELEASE_BRANCH:
        raise SystemExit(f"Releases must start on {RELEASE_BRANCH}; currently on {branch or 'a detached HEAD'}.")

    changes = read(["git", "status", "--porcelain"])
    if changes:
        raise SystemExit(f"Commit, stash, or remove all working-tree changes before releasing:\n{changes}")


def refuse_unless_in_sync_with_origin(*, fetch_tags: bool = True) -> None:
    run(["git", "fetch", "origin", RELEASE_BRANCH, "--tags" if fetch_tags else "--no-tags"])
    if read(["git", "rev-parse", "HEAD"]) != read(["git", "rev-parse", f"origin/{RELEASE_BRANCH}"]):
        raise SystemExit(f"Local {RELEASE_BRANCH} and origin/{RELEASE_BRANCH} must point to the same commit.")


def refuse_if_the_release_already_exists(version: str, tag: str) -> None:
    if succeeds(["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"]):
        raise SystemExit(
            f"Tag {tag} already exists. For a failed unpublished release, use: just release-retry {version}"
        )
    if read(["uv", "version", "--short"]) == version:
        raise SystemExit(f"The project is already at version {version}.")


def run_local_gates() -> None:
    """Run every gate before a commit or tag exists, so a failure costs nothing to recover from."""
    for gate in LOCAL_GATES:
        run(["just", gate])


def write_citation_version(path: Path, version: str, release_date: date) -> None:
    """Update the release fields without rewriting references or other citation metadata."""
    citation = path.read_text()
    for field, value in (("version", version), ("date-released", release_date.isoformat())):
        citation, count = re.subn(rf"^{field}:.*$", f'{field}: "{value}"', citation, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(f"CITATION.cff must contain exactly one top-level {field} field.")
    path.write_text(citation)


def write_version(version: str) -> None:
    run(["uv", "version", version])

    written_version = read(["uv", "version", "--short"])
    if written_version != version:
        raise SystemExit(f"uv wrote package version {written_version} instead of {version}.")

    write_citation_version(REPOSITORY_ROOT / "CITATION.cff", version, date.today())
    run(["uvx", "cffconvert", "--validate"])

    changed_files = read(["git", "diff", "--name-only"]).split()
    if set(changed_files) != set(VERSIONED_FILES):
        listing = "\n".join(f"  {path}" for path in changed_files)
        raise SystemExit(f"Versioning should change exactly {' '.join(VERSIONED_FILES)}; got:\n{listing}")
    run(["git", "diff", "--check"])


def commit_and_tag(version: str, tag: str) -> None:
    run(["git", "add", "--", *VERSIONED_FILES])
    run(["git", "commit", "-m", f"Bump version to {version}"])
    run(["git", "tag", "-a", tag, "-m", f"Release {version}"])


def push_atomically(tag: str) -> None:
    """Push the release commit and its tag together, or neither.

    GitHub supports atomic pushes: neither main nor the tag is updated if one ref is rejected.
    CI publishes only after receiving the tag and passing its own gates.
    """
    push = subprocess.run(["git", "push", "--atomic", "origin", RELEASE_BRANCH, tag], cwd=REPOSITORY_ROOT)
    if push.returncode:
        raise SystemExit(f"Nothing was pushed; the release commit and {tag} remain local for inspection.")


def refuse_if_published(project: str, version: str, *, index_url: str = "https://pypi.org/pypi") -> None:
    url = f"{index_url}/{quote(project, safe='')}/{quote(version, safe='')}/json"
    try:
        with urlopen(url, timeout=30):
            pass
    except HTTPError as error:
        if error.code == 404:
            return
        raise SystemExit(f"Cannot check whether {project} {version} is published: {error}") from error
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Cannot check whether {project} {version} is published: {error}") from error
    raise SystemExit(f"{project} {version} is already published on PyPI. Release a new version instead.")


def refuse_unless_release_failed(workflow_runs: list[dict], tag: str, commit: str) -> None:
    tag_runs = [entry for entry in workflow_runs if entry["head_branch"] == tag and entry["event"] == "push"]
    commit_runs = [entry for entry in tag_runs if entry["head_sha"] == commit]
    if not commit_runs:
        raise SystemExit(f"No release run found for {tag}. Check GitHub Actions before replacing the tag.")
    latest = max(commit_runs, key=lambda entry: entry["id"])
    has_unfinished_or_successful_runs = any(
        entry["status"] != "completed" or entry["conclusion"] == "success" for entry in tag_runs
    )
    retryable_conclusions = ("failure", "cancelled", "timed_out", "startup_failure")
    if has_unfinished_or_successful_runs or latest["conclusion"] not in retryable_conclusions:
        raise SystemExit(f"The release for {tag} has not failed. Check {latest.get('html_url', 'GitHub Actions')}.")


def read_remote_tag(tag: str) -> tuple[str, str]:
    # Fetch without updating local tags so a rejected earlier push cannot prevent another retry.
    run(["git", "fetch", "--no-tags", "origin", f"refs/tags/{tag}"])
    return read(["git", "rev-parse", "FETCH_HEAD"]), read(["git", "rev-parse", "FETCH_HEAD^{commit}"])


def check_retry_publication(project: str, version: str, repository: str, commit: str, tag: str) -> None:
    endpoint = (
        f"repos/{repository}/actions/workflows/release.yml/runs?branch={quote(tag, safe='')}&event=push&per_page=100"
    )
    pages = json.loads(read(["gh", "api", "--hostname", "github.com", "--paginate", "--slurp", endpoint]))
    refuse_unless_release_failed([entry for page in pages for entry in page["workflow_runs"]], tag, commit)
    refuse_if_published(project, version)


def replace_release_tag(version: str, tag: str, expected_tag: str, commit: str) -> None:
    tag_ref = f"refs/tags/{tag}"
    run(["git", "tag", "-f", "-a", tag, "-m", f"Release {version}", commit])
    run(["git", "push", "--no-follow-tags", f"--force-with-lease={tag_ref}:{expected_tag}", "origin", tag_ref])


def retry_release(version: str, tag: str) -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]
    if project["version"] != version:
        raise SystemExit(f"A retry must use the current project version {project['version']}.")
    expected_tag, old_commit = read_remote_tag(tag)
    commit = read(["git", "rev-parse", "HEAD"])
    if commit == old_commit:
        raise SystemExit(f"{tag} already points at HEAD. Re-run the failed jobs in GitHub Actions.")
    origin = read(["git", "remote", "get-url", "origin"])
    repository = read(["gh", "repo", "view", origin, "--json", "nameWithOwner", "--jq", ".nameWithOwner"])
    check_retry_publication(project["name"], version, repository, old_commit, tag)

    run_local_gates()

    # Checks can take several minutes. Recheck publication and the checkout before moving the tag.
    refuse_unless_on_a_clean_release_branch()
    refuse_unless_in_sync_with_origin(fetch_tags=False)
    if read(["git", "rev-parse", "HEAD"]) != commit:
        raise SystemExit("HEAD changed while validating the release. Run the retry again from the intended commit.")
    check_retry_publication(project["name"], version, repository, old_commit, tag)
    replace_release_tag(version, tag, expected_tag, commit)
    print(f"Pushed a new release attempt for {tag}. Follow it with: gh run list --workflow release.yml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="package version without the v prefix")
    parser.add_argument(
        "--retry", action="store_true", help="replace a failed unpublished release tag with current main"
    )
    arguments = parser.parse_args()

    version = canonical_version(arguments.version)
    tag = f"v{version}"

    refuse_unless_on_a_clean_release_branch()
    refuse_unless_in_sync_with_origin(fetch_tags=not arguments.retry)
    if arguments.retry:
        retry_release(version, tag)
        return
    refuse_if_the_release_already_exists(version, tag)

    run_local_gates()

    write_version(version)
    commit_and_tag(version, tag)
    push_atomically(tag)

    print(f"Pushed {tag}. Follow publication with: gh run list --workflow release.yml")


if __name__ == "__main__":
    main()
