import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from mudgym.connections.connection import MudConnection
from mudgym.connections.errors import ConnectionClosedError
from mudgym.connections.provider import ConnectionProvider
from mudgym.logs import get_logger

logger = get_logger(__name__)

CAPTURE_FORMAT = "mudgym-connection-capture"
CAPTURE_VERSION = 3


class ReplayMismatchError(RuntimeError):
    """Calls made during replay do not match the recorded connection transcript."""


def _read_capture(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one JSONL connection transcript."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Capture {path} is empty.")

    header, *calls = records
    if header.get("format") != CAPTURE_FORMAT:
        raise ValueError(f"Capture {path} has format {header.get('format')!r}, expected {CAPTURE_FORMAT!r}.")
    if header.get("version") != CAPTURE_VERSION:
        raise ValueError(f"Capture {path} has version {header.get('version')!r}, expected {CAPTURE_VERSION}.")

    for call in calls:
        if call["call"] == "read_response":
            call["raw_bytes"] = call.pop("raw_text").encode("latin-1")
    return header, calls


class _CaptureWriter:
    """Stream one connection transcript to disk, flushing after every call."""

    def __init__(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self._write({"format": CAPTURE_FORMAT, "version": CAPTURE_VERSION, **(metadata or {})})

    def _write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload))
        self.handle.write("\n")
        self.handle.flush()

    def write_call(self, call: str, **payload: Any) -> None:
        self._write({"call": call, **payload})

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


class RecordingConnection(MudConnection):
    """Record calls made at one live ``MudConnection`` boundary."""

    def __init__(
        self,
        connection: MudConnection,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = {"connection": type(connection).__name__, **(metadata or {})}
        self.connection = connection
        self.capture = _CaptureWriter(path, metadata)
        logger.debug("recording.start", path=str(path), connection=type(connection).__name__)

    def reset(self) -> None:
        self.connection.reset()
        self.capture.write_call("reset")

    def send_line(self, line: str) -> None:
        try:
            self.connection.send_line(line)
        except ConnectionClosedError:
            self.capture.write_call("send_line", line=line, error="connection_closed")
            raise
        self.capture.write_call("send_line", line=line)

    def read_response(self, end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        raw_bytes, terminated, incomplete, debug_info = self.connection.read_response(end_of_turn_marker)
        self.capture.write_call(
            "read_response",
            raw_text=raw_bytes.decode("latin-1"),
            terminated=bool(terminated),
            incomplete=bool(incomplete),
            rejected=bool(debug_info.get("rejected", False)),
            marker_arrived=bool(debug_info.get("marker_arrived", False)),
        )
        return raw_bytes, terminated, incomplete, debug_info

    def invalidate(self) -> None:
        self.connection.invalidate()
        self.capture.write_call("invalidate")

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            self.capture.close()


class ReplayConnection(MudConnection):
    """Replay and verify calls at one recorded ``MudConnection`` boundary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.header, self.calls = _read_capture(self.path)
        self.cursor = 0
        self._pending_lines: list[str] = []

    def _take(self, expected: str) -> dict[str, Any]:
        if self.cursor >= len(self.calls):
            raise ReplayMismatchError(f"Replay of {self.path} is exhausted; expected {expected!r}.")
        call = self.calls[self.cursor]
        if call["call"] != expected:
            raise ReplayMismatchError(
                f"Replay of {self.path} diverged at call {self.cursor}: "
                f"expected {expected!r}, capture has {call['call']!r}."
            )
        self.cursor += 1
        return call

    def reset(self) -> None:
        self._pending_lines.clear()
        self._take("reset")

    def send_line(self, line: str) -> None:
        call = self._take("send_line")
        if line != call["line"]:
            raise ReplayMismatchError(
                f"Replay of {self.path} diverged at call {self.cursor - 1}: "
                f"sent {line!r}, capture recorded {call['line']!r}."
            )
        if call.get("error") == "connection_closed":
            raise ConnectionClosedError(f"Recorded connection closed while sending {line!r}.")
        self._pending_lines.append(line)

    def read_response(self, end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        call = self._take("read_response")
        sent_lines = list(self._pending_lines)
        self._pending_lines.clear()
        return (
            call["raw_bytes"],
            call["terminated"],
            call["incomplete"],
            {
                "rejected": call["rejected"],
                "marker_arrived": call["marker_arrived"],
                "replayed": True,
                "capture": str(self.path),
                "sent_lines": sent_lines,
            },
        )

    def invalidate(self) -> None:
        self._take("invalidate")
        self._pending_lines.clear()

    def remaining_calls(self) -> int:
        return len(self.calls) - self.cursor

    def assert_exhausted(self) -> None:
        if remaining := self.remaining_calls():
            raise ReplayMismatchError(f"Replay of {self.path} finished with {remaining} unconsumed calls.")

    def close(self) -> None:
        pass


class RecordingProvider(ConnectionProvider):
    """Wrap each connection from another provider with its own connection capture."""

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
            # wrappers already built own their raw connection; the unwrapped suffix does not
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
    """Create replay connections without needing a game behind them."""

    def __init__(self, capture_path_for_index: Callable[[int], Path]):
        self.capture_path_for_index = capture_path_for_index

    def create_connections(self, count: int) -> list[MudConnection]:
        return [ReplayConnection(self.capture_path_for_index(index)) for index in range(count)]

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        pass

    def close(self) -> None:
        pass
