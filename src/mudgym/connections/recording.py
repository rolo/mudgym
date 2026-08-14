import json
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import ConnectionProvider
from mudgym.logs import get_logger

logger = get_logger(__name__)

CAPTURE_FORMAT = "mudgym-session-capture"
CAPTURE_VERSION = 2


class StaleCaptureError(RuntimeError):
    """The caller's conversation no longer matches the capture; re-recording is the only fix."""


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

    def record_step(
        self,
        lines: list[str],
        raw_bytes: bytes,
        terminated: bool,
        incomplete: bool,
        rejected: bool = False,
        marker_arrived: bool = False,
    ) -> None:
        event = {
            "event": "step",
            "lines": lines,
            "raw_text": bytes_to_capture_text(raw_bytes),
            "terminated": bool(terminated),
            "incomplete": bool(incomplete),
            "rejected": bool(rejected),
            "marker_arrived": bool(marker_arrived),
        }
        self._write_line(event)

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

    def reset(self) -> None:
        self.connection.reset()
        self.writer.record_reset()

    def send_line(self, line: str) -> None:
        self.connection.send_line(line)

    def read_response(self, lines: Sequence[str], end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        raw_bytes, terminated, incomplete, debug_info = self.connection.read_response(lines, end_of_turn_marker)
        self.writer.record_step(
            list(lines),
            raw_bytes,
            terminated,
            incomplete,
            rejected=bool(debug_info.get("rejected", False)),
            marker_arrived=bool(debug_info.get("marker_arrived", False)),
        )
        return raw_bytes, terminated, incomplete, debug_info

    def invalidate(self) -> None:
        self.connection.invalidate()

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
            raise StaleCaptureError(f"Replay of {self.path} exhausted: no event left to answer a {expected!r}.")
        event = self.events[self.cursor]
        if event["event"] != expected:
            raise StaleCaptureError(
                f"Replay of {self.path} out of order at event {self.cursor}: "
                f"expected a {expected!r}, capture has a {event['event']!r}."
            )
        self.cursor += 1
        return event

    def remaining_events(self) -> int:
        return len(self.events) - self.cursor

    def reset(self) -> None:
        self._next_event("reset")

    def read_response(self, lines: Sequence[str], end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        lines = list(lines)
        event = self._next_event("step")
        if lines != event["lines"]:
            raise StaleCaptureError(
                f"Replay of {self.path} diverged at event {self.cursor - 1}: "
                f"sent {lines!r} but the capture recorded {event['lines']!r}. "
                f"The capture is stale for this code; re-record it."
            )
        return (
            event["raw_bytes"],
            event["terminated"],
            event["incomplete"],
            {
                "rejected": event["rejected"],
                "marker_arrived": event["marker_arrived"],
                "replayed": True,
                "capture": str(self.path),
            },
        )

    def send_line(self, line: str) -> None:
        pass

    def invalidate(self) -> None:
        pass

    def close(self) -> None:
        pass


class RecordingProvider(ConnectionProvider):
    """Wraps each connection from another provider with its own session recording."""

    def __init__(
        self,
        provider: ConnectionProvider,
        capture_path_for_index: Callable[[int], Path],
        metadata: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.capture_path_for_index = capture_path_for_index
        self.metadata = metadata

    def create_connections(self, count: int) -> list[MudConnection]:
        connections: list[MudConnection] = []
        recorded_connections: list[MudConnection] = []
        try:
            connections = self.provider.create_connections(count)
            if len(connections) != count:
                raise RuntimeError(f"Provider returned {len(connections)} connections, expected {count}.")
            for index, connection in enumerate(connections):
                recorded_connections.append(
                    RecordingConnection(
                        connection,
                        self.capture_path_for_index(index),
                        self.metadata,
                    )
                )
            return recorded_connections
        except BaseException:
            # Wrappers already built own their raw connection. The unwrapped suffix does not. Close
            # each connection exactly once, then let the underlying provider release shared state.
            for connection in [*recorded_connections, *connections[len(recorded_connections) :]]:
                with suppress(Exception):
                    connection.close()
            with suppress(Exception):
                self.provider.close()
            raise

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        self.provider.reset(seed=seed)

    def close(self) -> None:
        self.provider.close()


class ReplayProvider(ConnectionProvider):
    """Provides one fixed batch of replay connections without needing a game behind them."""

    def __init__(self, capture_path_for_index: Callable[[int], Path]):
        self.capture_path_for_index = capture_path_for_index
        self._connections: list[ReplayConnection] = []
        self._batch_created = False

    def create_connections(self, count: int) -> list[MudConnection]:
        if count < 1:
            raise ValueError("count must be at least 1.")
        if self._batch_created:
            raise RuntimeError("Provider has already created its connection batch.")
        self._batch_created = True

        try:
            for index in range(count):
                self._connections.append(ReplayConnection(self.capture_path_for_index(index)))
            return list(self._connections)
        except BaseException:
            # A missing or stale capture can fail halfway through the batch. Nothing was returned,
            # so the provider still owns all replay connections created up to that point.
            for connection in self._connections:
                with suppress(Exception):
                    connection.close()
            self._connections.clear()
            raise

    def remaining_events(self) -> list[int]:
        return [connection.remaining_events() for connection in self._connections]

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        pass

    def close(self) -> None:
        pass
