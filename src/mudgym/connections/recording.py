"""Record and replay the wire conversation of a game session.

A capture file holds one session conversation with the game at the `MudConnection.send_command` layer: the lines sent
and the raw bytes received, in order, including the session setup steps.
`RecordingConnection` wraps any live connection and writes a capture.
`ReplayConnection` implements the same interface over a capture file, so recorded sessions replay through the real
env machinery with no game engine.

Captures are JSONL. Header line carries provenance; each following line is one event. Raw bytes are stored as
latin-1-decoded JSON strings and fails loudly on anything invalid.
"""

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import ConnectionProvider
from mudgym.logs import get_logger

logger = get_logger(__name__)

CAPTURE_FORMAT = "mudgym-session-capture"
CAPTURE_VERSION = 1


def bytes_to_capture_text(raw_bytes: bytes) -> str:
    return bytes(raw_bytes).decode("latin-1")


def capture_text_to_bytes(text: str) -> bytes:
    return text.encode("latin-1")


def read_capture(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a capture file, returning its header and its events with ``raw_bytes`` restored."""
    path = Path(path)
    # split strictly on "\n": JSON strings never contain a raw newline, while str.splitlines()
    # would also split on characters like U+0085 that ensure_ascii=False writes unescaped
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not lines:
        raise ValueError(f"Capture {path} is empty.")

    header = json.loads(lines[0])
    if header.get("format") != CAPTURE_FORMAT:
        raise ValueError(f"Capture {path} has format {header.get('format')!r}, expected {CAPTURE_FORMAT!r}.")
    if header.get("version") != CAPTURE_VERSION:
        raise ValueError(f"Capture {path} has version {header.get('version')!r}, expected {CAPTURE_VERSION}.")

    events = []
    for line in lines[1:]:
        event = json.loads(line)
        if event["event"] == "step":
            event["raw_bytes"] = capture_text_to_bytes(event.pop("raw_text"))
        events.append(event)
    return header, events


class CaptureWriter:
    """Streams one session's events to a capture file, header first, flushing per line."""

    def __init__(self, path: str | Path, metadata: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self._write_line({"format": CAPTURE_FORMAT, "version": CAPTURE_VERSION, **(metadata or {})})

    def _write_line(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, ensure_ascii=False))
        self.handle.write("\n")
        self.handle.flush()

    def record_reset(self) -> None:
        self._write_line({"event": "reset"})

    def record_step(self, lines: list[str], raw_bytes: bytes, terminated: bool, incomplete: bool) -> None:
        self._write_line(
            {
                "event": "step",
                "lines": lines,
                "raw_text": bytes_to_capture_text(raw_bytes),
                "terminated": bool(terminated),
                "incomplete": bool(incomplete),
            }
        )

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


class RecordingConnection(MudConnection):
    """Wraps a live connection and records every reset and step to a capture file."""

    def __init__(self, connection: MudConnection, path: str | Path, metadata: dict[str, Any] | None = None):
        metadata = {"connection": type(connection).__name__, **(metadata or {})}
        self.connection = connection
        self.writer = CaptureWriter(path, metadata)
        logger.debug("recording.start", path=str(path), connection=type(connection).__name__)

    # the session configures its per-instance end-of-turn marker on the connection it holds, so the
    # wrapper must hand the attribute through to the connection that actually reads the wire
    @property
    def end_of_turn_marker(self) -> re.Pattern:
        return self.connection.end_of_turn_marker

    @end_of_turn_marker.setter
    def end_of_turn_marker(self, value: re.Pattern) -> None:
        self.connection.end_of_turn_marker = value

    def reset(self) -> None:
        self.connection.reset()
        self.writer.record_reset()

    def send_command(self, command: str | Sequence[str]) -> tuple[bytes, bool, bool, dict[str, Any]]:
        lines = [command] if isinstance(command, str) else list(command)
        raw_bytes, terminated, incomplete, debug_info = self.connection.send_command(command)
        self.writer.record_step(lines, raw_bytes, terminated, incomplete)
        return raw_bytes, terminated, incomplete, debug_info

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            self.writer.close()


class ReplayConnection(MudConnection):
    """Plays a capture file back through the `MudConnection` interface, verifying every step.
    The caller must issue exactly the recorded conversation. Any divergence raises right away."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header, self.events = read_capture(self.path)
        self.cursor = 0

    def _next_event(self, expected: str) -> dict[str, Any]:
        if self.cursor >= len(self.events):
            raise RuntimeError(f"Replay of {self.path} exhausted: no event left to answer a {expected!r}.")
        event = self.events[self.cursor]
        if event["event"] != expected:
            raise RuntimeError(
                f"Replay of {self.path} out of order at event {self.cursor}: "
                f"expected a {expected!r}, capture has a {event['event']!r}."
            )
        self.cursor += 1
        return event

    def remaining_events(self) -> int:
        return len(self.events) - self.cursor

    def reset(self) -> None:
        self._next_event("reset")

    def send_command(self, command: str | Sequence[str]) -> tuple[bytes, bool, bool, dict[str, Any]]:
        lines = [command] if isinstance(command, str) else list(command)
        event = self._next_event("step")
        if lines != event["lines"]:
            raise RuntimeError(
                f"Replay of {self.path} diverged at event {self.cursor - 1}: "
                f"sent {lines!r} but the capture recorded {event['lines']!r}. "
                f"The capture is stale for this code; re-record it."
            )
        return (
            event["raw_bytes"],
            event["terminated"],
            event["incomplete"],
            {"replayed": True, "capture": str(self.path)},
        )

    def close(self) -> None:
        pass


class RecordingProvider(ConnectionProvider):
    """Wraps a provider so every connection it creates records to its own capture file."""

    def __init__(
        self,
        provider: ConnectionProvider,
        capture_path_for_index: Callable[[int], Path],
        metadata: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.capture_path_for_index = capture_path_for_index
        self.metadata = metadata

    def create_connection(self, env_index: int) -> MudConnection:
        return RecordingConnection(
            self.provider.create_connection(env_index),
            self.capture_path_for_index(env_index),
            self.metadata,
        )

    def close(self) -> None:
        self.provider.close()


class ReplayProvider(ConnectionProvider):
    """Creates replay connections from per-index capture files; needs no game infrastructure."""

    def __init__(self, capture_path_for_index: Callable[[int], Path]):
        self.capture_path_for_index = capture_path_for_index
        self.connections: list[ReplayConnection] = []

    def create_connection(self, env_index: int) -> MudConnection:
        connection = ReplayConnection(self.capture_path_for_index(env_index))
        self.connections.append(connection)
        return connection

    def close(self) -> None:
        pass
