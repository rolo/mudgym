import argparse
import re
import sys
import uuid
from collections.abc import Callable, Sequence

from mudgym.connections.connection import MudConnection
from mudgym.connections.prompts import FINAL_COMMAND, FINAL_COMMAND_MARKER
from mudgym.connections.registry import available_connections_dict, default_connection
from mudgym.featurizers.strings import decode_text_bytes
from mudgym.logs import get_logger, setup_logging

logger = get_logger(__name__)

# Verbs that consume the rest of the input line as free text (message or emote body), confirmed
# against the live game: after one of these a comma is message content rather than a command
# separator, so chained commands would be spoken instead of executed. '"' and "'" are say
# shorthands. An unlisted speech verb degrades to the old behaviour: the auto commands are
# swallowed, the end-of-turn marker never arrives, and the step comes back incomplete.
SPEECH_VERBS = frozenset(
    {
        # speech and directed speech
        "say",
        "shout",
        "sh",
        "tell",
        "t",
        "whisper",
        "yell",
        "ask",
        "wish",
        # emotes and mood utterances (the game treats the rest of the line as the uttered text)
        "emote",
        "act",
        "mutter",
        "moan",
        "grumble",
        "groan",
        "sigh",
        "laugh",
        "cry",
        "scream",
        "giggle",
        "cackle",
        "sob",
        "whine",
        "chuckle",
        "sing",
        "hum",
        "gasp",
        "snarl",
        "growl",
    }
)
SPEECH_PREFIXES = ('"', "'")


def contains_speech_command(command: str) -> bool:
    """True when any comma-chained segment starts a speech command that eats the rest of the line."""
    for segment in command.split(","):
        stripped = segment.strip()
        if not stripped:
            continue
        if stripped.startswith(SPEECH_PREFIXES):
            return True
        if stripped.split(None, 1)[0].lower() in SPEECH_VERBS:
            return True
    return False


def wire_lines(command: str, auto_commands: Sequence[str]) -> list[str]:
    """Assemble the wire lines for one step's batch.

    Speech commands consume the rest of their input line, so when the player command contains one
    the auto commands go out on their own line; otherwise the whole batch is one comma-chained
    line. Each wire line is echoed separately by the game, so anything splitting on the echo must
    use the same line assembly.
    """
    if not auto_commands:
        return [command]
    if contains_speech_command(command):
        return [command, ",".join(auto_commands)]
    return [",".join([command, *auto_commands])]


class MudSession:
    """
    Handles the lifecycle of a game session from a safe place of blissful unawareness of RL environments,
    specs or any of that jazz.
    """

    def __init__(
        self,
        connection: MudConnection | Callable[[], MudConnection] = default_connection,
        auto_commands: list[str] | None = None,
        *,
        end_of_turn_marker: re.Pattern,
    ) -> None:
        self.session_id = uuid.uuid4()

        if isinstance(connection, MudConnection):
            self.connection = connection
            logger.debug("session.create", session_id=str(self.session_id), mode="prebuilt_connection")
        else:
            logger.debug(
                "session.create",
                session_id=str(self.session_id),
                mode="factory",
                connection_factory=getattr(connection, "__name__", repr(connection)),
            )
            self.connection = connection()

        self.auto_commands: list[str] = list(auto_commands or [])

        # the connection's read window closes on the response of the batch's final command
        self.connection.end_of_turn_marker = end_of_turn_marker

        self.step_count = 0
        self.last_raw_bytes: bytes = b""
        self.last_debug_info: dict = {}

    def reset(self) -> None:
        """
        Reset the session: connect, enter the tearoom.
        """
        logger.debug("session.reset", session_id=str(self.session_id))
        self.step_count = 0
        self.connection.reset()

    def send_command(
        self,
        command: str,
        add_auto_commands: bool = True,
    ) -> tuple[bytes, bool, bool, dict]:
        """
        Wrapper around MudConnection.send_command.

        Assembles the command string, adding any auto commands, and passes it to the connection instance
        to send.

        We are still speaking "game" rather than "RL" here.

        Args:
            command: The command to send to the game process.
            add_auto_commands: Whether to add the auto commands to the command string.
            defaults to True.

        Returns:
            tuple[bytes, bool, bool, dict]: The raw response bytes, terminated flag, incomplete flag, and debug
            info.
        """
        lines = wire_lines(command, self.auto_commands if add_auto_commands else [])

        raw_bytes, terminated, incomplete, debug_info = self.connection.send_command(
            lines[0] if len(lines) == 1 else lines
        )
        debug_info["wire_lines"] = lines

        # an unlisted speech verb swallows the comma-chained auto commands into the spoken text, so
        # the tail shows up twice: once in our echo and once quoted back in the speech output
        if incomplete and len(lines) == 1 and self.auto_commands and add_auto_commands:
            autos_tail = ("," + ",".join(self.auto_commands)).encode("ascii")
            if raw_bytes.count(autos_tail) > 1:
                logger.warning(
                    "session.send_command.autos_spoken",
                    command=command,
                    hint="the auto commands appear spoken in the output; is the verb missing from SPEECH_VERBS?",
                )

        self.last_raw_bytes = raw_bytes
        self.last_debug_info = debug_info
        debug_info["step"] = self.step_count
        self.step_count += 1
        return raw_bytes, terminated, incomplete, debug_info

    def close(self) -> None:
        logger.debug("session.close", session_id=str(self.session_id))
        try:
            self.connection.close()
        except Exception as e:
            logger.debug("session.close.error", session_id=str(self.session_id), error=str(e))


def play(
    connection: MudConnection | Callable[[], MudConnection] = default_connection,
) -> None:
    label = getattr(connection, "__name__", type(connection).__name__)
    print("Beginning game session with connection: ", label)
    # the batch must end with the marker command or every step waits out the read timeout
    session = MudSession(connection=connection, auto_commands=[FINAL_COMMAND], end_of_turn_marker=FINAL_COMMAND_MARKER)
    session.reset()

    # Show initial game state before waiting for first command
    initial_bytes, terminated, incomplete, _ = session.send_command("l")
    if initial_bytes:
        print(decode_text_bytes(initial_bytes), end="")
    print()
    sys.stdout.flush()

    while not (terminated or incomplete):
        command = input()
        raw_bytes, terminated, incomplete, debug_info = session.send_command(command)
        if raw_bytes:
            # Decode and print bytes directly so ANSI codes render properly
            print(decode_text_bytes(raw_bytes), end="")
        print()
        sys.stdout.flush()

    print("Exiting...")
    session.close()


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Play a MUD session")
    parser.add_argument(
        "--connection",
        type=str,
        choices=list(available_connections_dict.keys()),
        metavar="SLUG",
        help=f"Connection type to use. Available: {', '.join(available_connections_dict.keys())}",
    )
    args = parser.parse_args()

    connection_class = default_connection
    if args.connection:
        connection_class = available_connections_dict[args.connection]

    play(connection=connection_class)


if __name__ == "__main__":
    main()
