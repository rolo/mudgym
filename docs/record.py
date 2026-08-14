"""Capture the documentation examples' game sessions and derive the fragments the docs display.

The example code shown on the docs pages lives in `docs/code/`, one file per example, inside
`--8<--` snippet regions the pages include -- so the code a reader sees is exactly the code that
runs. Each example runs in two phases:

- record: the example plays the live game once (requires Docker) and each connection transcript is
  written to `docs/recordings/<name>*.session.jsonl` -- the committed source of truth.
- derive: the example runs again over `ReplayConnection`, with no game behind it, and the displayed
  fragments (`docs/recordings/*.md`, `*.ansi`) are rewritten from the replayed transcript.

Because fragments are derived from a committed capture, presentation edits re-derive the same
world instead of rerolling a random one, and `tests/test_docs_recordings.py` can verify the
committed fragments still match their captures without touching the game. Replay is strict: if an
example's commands no longer match its capture, deriving fails loudly and the capture needs
re-recording.

Run through the justfile:

    just docs-record                  # play the examples against the live game, rewrite captures + fragments
    just docs-record actions-text     # ... for a subset by name
    just docs-derive                  # rewrite fragments from the committed captures; no Docker
    just docs-watch                   # serve the docs, refreshing whichever example is saved as you edit
"""

import argparse
import contextlib
import importlib.util
import io
import itertools
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from mudgym.connections import registry
from mudgym.connections.provider import DockerExecProvider
from mudgym.connections.recording import (
    RecordingConnection,
    RecordingProvider,
    ReplayConnection,
    ReplayMismatchError,
    ReplayProvider,
    _read_capture,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DOCS_CODE_DIR = REPOSITORY_ROOT / "docs" / "code"
RECORDINGS_DIR = REPOSITORY_ROOT / "docs" / "recordings"


def discover_examples() -> dict[str, Callable]:
    """Load every example in docs/code/; the recording name is the file stem with dashes."""
    examples: dict[str, Callable] = {}
    for path in sorted(DOCS_CODE_DIR.glob("*.py")):
        name = path.stem.replace("_", "-")
        # editor duplicates like "index_single copy.py" would otherwise record under a name no
        # ownership glob or dash-prefix check can see
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
            raise SystemExit(
                f"{path.name} yields recording name {name!r}; example file stems must use only "
                f"lowercase letters, digits and underscores."
            )
        spec = importlib.util.spec_from_file_location(f"docs_code_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        example = getattr(module, "example", None)
        if not callable(example):
            raise SystemExit(f"{path} must define an example(fragments) function.")
        examples[name] = example
    # fragment ownership globs on `<name>-*` require that no name extends another with a dash
    for name in examples:
        clashes = [other for other in examples if other != name and other.startswith(f"{name}-")]
        if clashes:
            raise SystemExit(f"Example name {name!r} is a dash-prefix of {clashes}; rename one of them.")
    return examples


EXAMPLES = discover_examples()


def single_env_capture_path(name: str, index: int) -> Path:
    suffix = ".session.jsonl" if index == 0 else f".env{index}.session.jsonl"
    return RECORDINGS_DIR / f"{name}{suffix}"


def agent_capture_path(name: str, env_index: int) -> Path:
    return RECORDINGS_DIR / f"{name}.player_{env_index}.session.jsonl"


def capture_paths(name: str) -> list[Path]:
    patterns = (f"{name}.session.jsonl", f"{name}.env*.session.jsonl", f"{name}.player_*.session.jsonl")
    return sorted(path for pattern in patterns for path in RECORDINGS_DIR.glob(pattern))


def fragment_paths(name: str, directory: Path) -> list[Path]:
    """Every fragment a recording owns; discover_examples() guarantees the dash-prefix invariant."""
    patterns = (f"{name}.md", f"{name}.ansi", f"{name}-*.md", f"{name}-*.ansi")
    return sorted(path for pattern in patterns for path in directory.glob(pattern))


def describe_repository() -> str:
    return subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def describe_path(path: Path) -> str:
    """The repo-relative path for the usual case, the absolute one when writing elsewhere (tests)."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


class FragmentWriters:
    """Write an example's displayed fragments, tying each one to its connection captures."""

    def __init__(self, example_name: str, fragments_dir: Path, captures_dir: Path):
        self.example_name = example_name
        self.fragments_dir = fragments_dir
        self.captures_dir = captures_dir

    def validated_fragment_name(self, name: str) -> str:
        """Fragments must glob under their example's name, or stale-cleanup and ownership fall apart."""
        if name != self.example_name and not name.startswith(f"{self.example_name}-"):
            raise SystemExit(
                f"Fragment {name!r} does not belong to example {self.example_name!r}; "
                f"name it {self.example_name!r} or prefix it with '{self.example_name}-'."
            )
        return name

    def provenance(self) -> str:
        """An HTML comment tying the fragment to its capture, invisible in the rendered docs."""
        name = self.example_name
        patterns = (f"{name}.session.jsonl", f"{name}.env*.session.jsonl", f"{name}.player_*.session.jsonl")
        paths = sorted(path for pattern in patterns for path in self.captures_dir.glob(pattern))
        header, _ = _read_capture(paths[0])
        return (
            f"<!-- Derived from the {name} connection capture (mudgym {header['mudgym']}) by docs/record.py. "
            f"Do not edit by hand: `just docs-derive {name}` rewrites this, "
            f"`just docs-record {name}` re-records it. -->"
        )

    def write_fragment(self, name: str, body: str) -> None:
        path = self.fragments_dir / f"{self.validated_fragment_name(name)}.md"
        path.write_text(f"{self.provenance()}\n\n{body.rstrip()}\n", encoding="utf-8")
        print(f"wrote {describe_path(path)}")

    def write_fenced(self, name: str, output: str, language: str = "text") -> None:
        self.write_fragment(name, f"```{language}\n{output.rstrip()}\n```")

    def write_ansi(self, name: str, render_bytes: bytes) -> None:
        path = self.fragments_dir / f"{self.validated_fragment_name(name)}.ansi"
        path.write_bytes(bytes(render_bytes))
        print(f"wrote {describe_path(path)}")

    @staticmethod
    @contextlib.contextmanager
    def captured_stdout() -> Iterator[io.StringIO]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            yield buffer

    @staticmethod
    def observation_table(observation: dict) -> str:
        def format_value(value: object) -> str:
            if isinstance(value, np.ndarray):
                return f"`{np.array2string(value, separator=', ', threshold=24)}`"
            if isinstance(value, tuple):
                return ", ".join(f"`{item}`" for item in value) if value else "_empty_"
            return f"`{value}`"

        lines = ["| Key | Value |", "|---|---|"]
        lines.extend(f"| `{key}` | {format_value(value)} |" for key, value in observation.items() if key != "text")
        return "\n".join(lines)


@contextlib.contextmanager
def swapped_default_backends(
    connection_factory,
    provider_factory,
    parallel_provider_factory=None,
) -> Iterator[None]:
    """Point the factory defaults at different backends while an example runs.

    The example code must stay a bare ``make_env()`` / ``make_parallel_env()`` call -- it is the
    code readers see -- so the swap happens at the registry defaults the factory resolves at call
    time, restored on the way out.
    """
    if parallel_provider_factory is None:
        parallel_provider_factory = provider_factory

    previous = (
        registry.default_connection,
        registry.default_provider_factory,
        registry.default_parallel_provider_factory,
    )
    registry.default_connection = connection_factory
    registry.default_provider_factory = provider_factory
    registry.default_parallel_provider_factory = parallel_provider_factory
    try:
        yield
    finally:
        (
            registry.default_connection,
            registry.default_provider_factory,
            registry.default_parallel_provider_factory,
        ) = previous


def record(name: str, version: str) -> None:
    """Play the example against the live game, capturing every session it opens.

    ``version`` is captured by the caller before any recording is touched, so rewriting captures
    does not mark the run's own provenance dirty.
    """
    metadata = {"recorded_by": f"just docs-record {name}", "mudgym": version}
    live_connection = registry.default_connection
    env_counter = itertools.count()
    written: list[Path] = []

    def connection_factory():
        path = single_env_capture_path(name, next(env_counter))
        written.append(path)
        return RecordingConnection(live_connection(), path, metadata)

    def agent_capture_path_written(env_index: int) -> Path:
        path = agent_capture_path(name, env_index)
        written.append(path)
        return path

    def provider_factory():
        return RecordingProvider(DockerExecProvider(), agent_capture_path_written, metadata)

    def parallel_provider_factory():
        return RecordingProvider(DockerExecProvider(worlds=1), agent_capture_path_written, metadata)

    with swapped_default_backends(connection_factory, provider_factory, parallel_provider_factory):
        EXAMPLES[name](FragmentWriters(name, RECORDINGS_DIR, RECORDINGS_DIR))

    # only after a successful run: captures the example no longer produces (a removed agent, fewer
    # envs) must not linger, but a failed run must not have deleted the previous good captures either
    for stale in set(capture_paths(name)) - set(written):
        stale.unlink()
    for path in capture_paths(name):
        print(f"wrote {describe_path(path)}")


def derive(name: str, fragments_dir: Path = RECORDINGS_DIR) -> None:
    """Replay the example's committed captures and rewrite the fragments the docs display."""
    if not capture_paths(name):
        raise SystemExit(f"No connection capture for {name!r}; run `just docs-record {name}` first.")

    # clear the recording's fragments first, so ones the replay no longer produces (a removed
    # agent, a renamed fragment) cannot linger and keep rendering in the docs
    for stale in fragment_paths(name, fragments_dir):
        stale.unlink()

    env_counter = itertools.count()
    replays: list[ReplayConnection] = []

    def connection_factory():
        connection = ReplayConnection(single_env_capture_path(name, next(env_counter)))
        replays.append(connection)
        return connection

    def provider_factory():
        class RetainedReplayProvider(ReplayProvider):
            def create_connections(self, count: int):
                connections = super().create_connections(count)
                replays.extend(connections)
                return connections

        return RetainedReplayProvider(lambda env_index: agent_capture_path(name, env_index))

    with swapped_default_backends(connection_factory, provider_factory):
        EXAMPLES[name](FragmentWriters(name, fragments_dir, RECORDINGS_DIR))

    try:
        for connection in replays:
            connection.assert_exhausted()
    except ReplayMismatchError as error:
        raise ReplayMismatchError(f"{error} Run `just docs-record {name}` to re-record it.") from error


def refresh_example(name: str) -> None:
    """Derive one example's fragments, recording it live first when its capture is stale or missing.

    Presentation edits keep the recorded world: derivation alone answers them. The live game is only
    contacted when the example has no capture yet, or replay proves its commands have changed.
    """
    if not capture_paths(name):
        print(f"{name}: no capture yet; recording against the live game")
        record(name, describe_repository())
        derive(name)
        return
    try:
        derive(name)
    except ReplayMismatchError as error:
        print(f"{name}: {error}")
        print(f"{name}: commands changed; re-recording against the live game")
        record(name, describe_repository())
        derive(name)


def snapshot_mtimes() -> dict[Path, int]:
    mtimes = {}
    for path in sorted(DOCS_CODE_DIR.glob("*.py")):
        with contextlib.suppress(FileNotFoundError):  # an editor save racing the scan; settled next poll
            mtimes[path] = path.stat().st_mtime_ns
    return mtimes


def watch(poll_seconds: float = 0.5) -> None:
    """Watch docs/code/, refreshing an example's recording and fragments whenever its file is saved.

    A failed refresh (a half-saved file, the game unreachable, a bug in the example) is reported and
    watching continues; the next save retries. Pair with `just docs`: zensical's server watches
    docs/code/ and docs/recordings/ too, so the browser reloads with whatever a refresh writes.
    """
    global EXAMPLES
    print(f"Watching {describe_path(DOCS_CODE_DIR)} (Ctrl-C to stop)")
    seen = snapshot_mtimes()
    try:
        while True:
            time.sleep(poll_seconds)
            current = snapshot_mtimes()
            for path in sorted(set(seen) - set(current)):
                print(f"{path.name} was removed; its captures and fragments remain until deleted")
            changed = [path for path, mtime in current.items() if seen.get(path) != mtime]
            seen = current
            for path in changed:
                name = path.stem.replace("_", "-")
                print(f"{path.name} changed; refreshing {name}")
                try:
                    EXAMPLES = discover_examples()
                    refresh_example(name)
                except (Exception, SystemExit):  # one save must not end the watch; the next retries
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("Stopped watching.")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Capture and derive the documentation recordings.")
    parser.add_argument("names", nargs="*", help="Recording names; defaults to all of them.")
    parser.add_argument(
        "--derive-only",
        action="store_true",
        help="Skip the live game and rewrite fragments from the committed connection captures.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch docs/code/ and refresh an example whenever its file changes.",
    )
    args = parser.parse_args(argv)

    if args.watch:
        if args.names or args.derive_only:
            parser.error("--watch watches every example and decides record vs derive itself; use it alone.")
        RECORDINGS_DIR.mkdir(exist_ok=True)
        watch()
        return

    names = args.names or list(EXAMPLES)
    unknown = [name for name in names if name not in EXAMPLES]
    if unknown:
        raise SystemExit(f"Unknown recordings: {', '.join(unknown)}. Valid names: {', '.join(EXAMPLES)}.")

    RECORDINGS_DIR.mkdir(exist_ok=True)
    # one version for the whole run, taken before any capture is rewritten: deleting or rewriting
    # tracked recordings mid-run must not turn the remaining headers' provenance dirty
    version = None if args.derive_only else describe_repository()
    for name in names:
        if not args.derive_only:
            record(name, version)
        derive(name)


if __name__ == "__main__":
    main(sys.argv[1:])
