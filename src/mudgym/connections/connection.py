import re
from abc import ABC, abstractmethod
from typing import Any


class MudConnection(ABC):
    """Connection interface for live WASM sessions and recorded transcripts.

    Reset prepares a session in the tearoom. Sending and reading are separate so several players can act before collecting their observations. Closing releases the connection's resources.
    """

    requires_end_of_turn_marker = True

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None:
        """Prepare the session in the tearoom for a new episode."""

    @abstractmethod
    def send_line(self, line: str) -> None:
        """Send a line without collecting its response."""

    @abstractmethod
    def read_response(
        self,
        end_of_turn_marker: re.Pattern | None,
    ) -> tuple[bytes, bool, bool, dict[str, Any]]:
        """Collect pending output, game-over and incomplete flags, and transport details."""

    @abstractmethod
    def invalidate(self) -> None:
        """Discard a failed session so it requires a successful reset before reuse."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection's resources."""
