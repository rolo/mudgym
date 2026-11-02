"""Record the docs examples against the live game, and turn the recordings into the fragments the docs show.

The examples live in `docs/code/` and the pages include their code, so what you read is what ran. Recording saves each
connection's transcript to `docs/recordings/`, which is committed. Deriving replays them with no game behind it and
writes the fragments, which aren't. If an example's commands stop matching its recording, deriving fails and it needs
recording again.

    just docs-record [names]    # play the examples against the live game
    just docs-derive            # rewrite the fragments from the committed recordings
    just docs-watch             # serve the docs and refresh whichever example you save
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
    """Load every example in docs/code/, named by its file stem with dashes."""
    examples: dict[str, Callable] = {}
    for path in sorted(DOCS_CODE_DIR.glob("*.py")):
        name = path.stem.replace("_", "-")
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
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


class FragmentWriters:
    def __init__(self, example_name: str, fragments_dir: Path, captures_dir: Path):
        self.example_name = example_name
        self.fragments_dir = fragments_dir
        self.captures_dir = captures_dir

    def validated_fragment_name(self, name: str) -> str:
        if name != self.example_name and not name.startswith(f"{self.example_name}-"):
            raise SystemExit(
                f"Fragment {name!r} does not belong to example {self.example_name!r}; "
                f"name it {self.example_name!r} or prefix it with '{self.example_name}-'."
            )
        return name

    def provenance(self) -> str:
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
    parallel_provider_factory,
) -> Iterator[None]:
    """Swap the registry's default backends while an example runs, so the example stays a plain `make_env()` call."""
    previous = (
        registry.default_connection,
        registry.default_parallel_provider_factory,
    )
    registry.default_connection = connection_factory
    registry.default_parallel_provider_factory = parallel_provider_factory
    try:
        yield
    finally:
        (
            registry.default_connection,
            registry.default_parallel_provider_factory,
        ) = previous


def record(name: str, version: str) -> None:
    """Play the example against the live game, saving every session it opens."""
    metadata = {"recorded_by": f"just docs-record {name}", "mudgym": version}
    live_connection = registry.default_connection
    live_parallel_provider = registry.default_parallel_provider_factory
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

    def parallel_provider_factory():
        return RecordingProvider(live_parallel_provider(), agent_capture_path_written, metadata)

    with swapped_default_backends(connection_factory, parallel_provider_factory):
        EXAMPLES[name](FragmentWriters(name, RECORDINGS_DIR, RECORDINGS_DIR))

    # Only after a successful run, so a failed one keeps the old recordings.
    for stale in set(capture_paths(name)) - set(written):
        stale.unlink()
    for path in capture_paths(name):
        print(f"wrote {describe_path(path)}")


def derive(name: str, fragments_dir: Path = RECORDINGS_DIR) -> None:
    if not capture_paths(name):
        raise SystemExit(f"No connection capture for {name!r}; run `just docs-record {name}` first.")

    # Clear the old fragments first, so ones the example stopped writing don't hang about.
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
    """Derive an example's fragments, recording it first if it has no recording yet or its commands changed."""
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
        with contextlib.suppress(FileNotFoundError):  # mid-save, the next poll picks it up
            mtimes[path] = path.stat().st_mtime_ns
    return mtimes


def watch(poll_seconds: float = 0.5) -> None:
    """Refresh an example whenever its file in docs/code/ is saved, and keep watching if a refresh fails."""
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
                except (Exception, SystemExit):  # noqa: BLE001 - report failed refreshes and keep watching
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
    # Take the version before rewriting any recordings, or git describe would call it dirty.
    version = None if args.derive_only else describe_repository()
    for name in names:
        if not args.derive_only:
            record(name, version)
        derive(name)


if __name__ == "__main__":
    main(sys.argv[1:])
