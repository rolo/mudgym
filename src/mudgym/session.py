from mudgym.connections.connection import MudConnection
from mudgym.connections.errors import ConnectionClosedError
from mudgym.featurizers.quickscore import QUICKSCORE_COMMAND, parse_quickscore


class MudSession:
    """
    Handles the lifecycle of a game session for a single player from a safe place of blissful unawareness of RL
    environments, specs or any of that jazz.
    """

    def __init__(self, connection: MudConnection, *, observation_line: str = "") -> None:
        self.connection = connection
        self.observation_line = observation_line

        # A pending command has been sent to the game but its response hasn't been read yet.
        # This is for the two step act/observe pattern to ensure we have the latest game text to act on.
        self.pending_command: str | None = None

    def reset(self, *, seed: int | None = None) -> tuple[str, int]:
        """Enter the tearoom and return the persona name and points from quickscore."""
        self.connection.reset(seed=seed)
        self.pending_command = None

        self.send(QUICKSCORE_COMMAND)
        raw_bytes, terminated, incomplete, _ = self.read_pending_response()
        if terminated or incomplete:
            raise RuntimeError(
                f"quickscore failed during reset (terminated={terminated}, incomplete={incomplete}): {raw_bytes!r}"
            )
        return parse_quickscore(raw_bytes)

    def send(self, command: str) -> None:
        """Send one player's command without waiting for its response."""
        if self.pending_command is not None:
            raise RuntimeError(
                f"Cannot send {command!r} while command {self.pending_command!r} is still waiting to be received."
            )
        self.connection.send_line(command)
        self.pending_command = command

    def receive(self) -> tuple[bytes, bool, bool, dict]:
        """Send any observation commands, then read the completed response.

        This is called without a pending player command during reset for the final observation sweep
        after every player has entered the world.
        """
        command = self.pending_command
        try:
            if self.observation_line:
                self.connection.send_line(self.observation_line)
        except ConnectionClosedError:
            # If the player action made it onto the wire, there may still be a response worth
            # reading. With no pending action there is nothing to recover, so surface the failure.
            if command is None:
                raise
        return self.read_pending_response()

    def read_pending_response(self) -> tuple[bytes, bool, bool, dict]:
        """Read the completed response without sending observation commands."""
        try:
            raw_bytes, terminated, incomplete, debug_info = self.connection.read_response()
        finally:
            self.pending_command = None

        if terminated or incomplete:
            self.connection.invalidate()
        return raw_bytes, terminated, incomplete, debug_info

    def command(self, command: str) -> tuple[bytes, bool, bool, dict]:
        """Send one command and immediately receive its framed response."""
        self.send(command)
        return self.receive()

    def close(self) -> None:
        self.connection.close()
