import re

from mudgym.connections.connection import MudConnection
from mudgym.connections.errors import ConnectionClosedError
from mudgym.featurizers.quickscore import parse_quickscore_name


class MudSession:
    """
    Handles the lifecycle of a game session for a single player from a safe place of blissful unawareness of RL
    environments, specs or any of that jazz.
    """

    def __init__(
        self,
        connection: MudConnection,
        *,
        observation_line: str,
        end_of_turn_marker: re.Pattern,
    ) -> None:
        self.connection = connection

        if not observation_line:
            raise ValueError("MudSession requires an observation command line.")
        self.observation_line = observation_line
        self.end_of_turn_marker = end_of_turn_marker

        self.persona: str | None = None
        # A pending command has been sent to the game but its response hasn't been read yet.
        # This is for the two step act/observe pattern to ensure we have the latest game text to act on. 
        self.pending_command: str | None = None

    def reset(self) -> None:
        """
        Reset the session: connect, enter the tearoom, and learn which persona we are playing.
        """
        self.pending_command = None
        self.connection.reset()

        # use quickscore to find out the current persona name, regardless of connection type or how we got here
        raw_bytes, terminated, incomplete, _ = self.command("qs")
        if terminated or incomplete:
            raise RuntimeError(
                f"quickscore failed during reset (terminated={terminated}, incomplete={incomplete}): {raw_bytes!r}"
            )
        self.persona = parse_quickscore_name(raw_bytes)

    def send(self, command: str) -> None:
        """Send one player's command without waiting for its response."""
        if self.pending_command is not None:
            raise RuntimeError(
                f"Cannot send {command!r}; command {self.pending_command!r} is still waiting to be received."
            )
        self.connection.send_line(command)
        self.pending_command = command

    def receive(self) -> tuple[bytes, bool, bool, dict]:
        """Send the observation commands, then receive one marker ended response.

        This is called without a pending player command during reset for the final observation sweep
        after every player has entered the world.
        """
        command = self.pending_command
        wire_lines = [] if command is None else [command]
        try:
            self.connection.send_line(self.observation_line)
        except ConnectionClosedError:
            # If the player action made it onto the wire, there may still be a response worth
            # reading. With no pending action there is nothing to recover, so surface the failure.
            if command is None:
                raise
        else:
            wire_lines.append(self.observation_line)

        try:
            raw_bytes, terminated, incomplete, debug_info = self.connection.read_response(
                wire_lines,
                self.end_of_turn_marker,
            )
        finally:
            self.pending_command = None

        debug_info["wire_lines"] = wire_lines
        if terminated or incomplete:
            self.connection.invalidate()
        return raw_bytes, terminated, incomplete, debug_info

    def command(self, command: str) -> tuple[bytes, bool, bool, dict]:
        """Send one command and immediately receive its framed response."""
        self.send(command)
        return self.receive()

    def close(self) -> None:
        self.connection.close()
