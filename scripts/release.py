"""Validate, version, commit, tag, and push a release; GitHub Actions publishes it via OIDC.

Run via justfile as `just release 0.4.0`
"""

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RELEASE_BRANCH = "main"
VERSIONED_FILES = ["pyproject.toml", "uv.lock"]
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


def refuse_unless_in_sync_with_origin() -> None:
    run(["git", "fetch", "origin", RELEASE_BRANCH, "--tags"])
    if read(["git", "rev-parse", "HEAD"]) != read(["git", "rev-parse", f"origin/{RELEASE_BRANCH}"]):
        raise SystemExit(f"Local {RELEASE_BRANCH} and origin/{RELEASE_BRANCH} must point to the same commit.")


def refuse_if_the_release_already_exists(version: str, tag: str) -> None:
    if succeeds(["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"]):
        raise SystemExit(f"Tag {tag} already exists.")
    if read(["uv", "version", "--short"]) == version:
        raise SystemExit(f"The project is already at version {version}.")


def run_local_gates() -> None:
    """Run every gate before a commit or tag exists, so a failure costs nothing to recover from."""
    for gate in LOCAL_GATES:
        run(["just", gate])


def write_version(version: str) -> None:
    run(["uv", "version", version])

    written_version = read(["uv", "version", "--short"])
    if written_version != version:
        raise SystemExit(f"uv wrote package version {written_version} instead of {version}.")

    changed_files = read(["git", "diff", "--name-only"]).split()
    if changed_files != VERSIONED_FILES:
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: release.py VERSION (for example, 0.4.0)")

    version = canonical_version(sys.argv[1])
    tag = f"v{version}"

    refuse_unless_on_a_clean_release_branch()
    refuse_unless_in_sync_with_origin()
    refuse_if_the_release_already_exists(version, tag)

    run_local_gates()

    write_version(version)
    commit_and_tag(version, tag)
    push_atomically(tag)

    print(f"Released {tag}; follow it with: gh run watch")


if __name__ == "__main__":
    main()
