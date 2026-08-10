from collections.abc import Mapping, Sequence
from typing import Any

from mudgym.connections.connection import MudConnection
from mudgym.connections.prompts import INVENTORY_DIVIDER
from mudgym.envs.factory import make_env
from mudgym.featurizers.quickscore import QUICKSCORE_COMMAND

PROMPT = b"\r\n*"

# The canned world is Dally Lane (room groad3), quoted from real captures: the room, fes, fex, and
# divider shapes are in src/mudgym/envs/fields/tests/payloads.py, and the necklace visit with its
# sql listing is a captured episode of the same room. Persona names are generated fresh each
# session; Alexander was the captured one.
ROOM_TEXT = (
    b"\x1b[32mDally Lane\x1b[37m.\r\n"
    b"\x1b[0;32;40mYou are standing on a dusty road with rising ground both to the north and "
    b"south. Though dilapidated and disused, the route north of where you stand, with a building "
    b"at the far end, looks as if it once formed a grand driveway. To the south, the road twists "
    b"up the hill where, at the summit, an ancient walled monastery dominates the scene. Open "
    b"fields lie to the west, and east is a flat area of lawn. \x1b[1;37;40m"
    b"\x1b[0;32;40mIt is raining. \x1b[1;37;40m"
    b"\x1b[36mA splendid necklace lies on the ground. \x1b[37m\r\n"
)
TEAROOM_EXIT_TEXT = (
    b"As you step through the opening, you become swathed in a fine, gossamer mist. The "
    b"Elizabethan tearoom fades hazily away, and vague, new shapes begin to form around you. "
    b"Their outlines become more defined, their colours grow stronger, and the mist thins out "
    b"into pale wisps, which gradually disperse away to nothingness...\r\n"
)
SQL_RESPONSE = (
    b'The place known as "\x1b[1;32;40mDally Lane\x1b[0;37;40m" contains '
    b"\x1b[1;36;40mthe necklace\x1b[0;37;40m, "
    b"\x1b[32mrain\x1b[37m, "
    b"\x1b[31mAlexander the protector\x1b[37m and "
    b"\x1b[32mthe road\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
)
# Captured shape of the quickscore reply reset uses as its tearoom identity probe: the name line
# (with the level title the game appends) then the stats line, whose values mirror FES_RESPONSE.
QS_RESPONSE = (
    b"\x1b[0;37;40mAlexander the protector\r\n"
    b"eff str 52      eff dex 53      sta \x1b[1;32;40m75\x1b[0;37;40m/\x1b[1;32;40m75\x1b[0;37;40m"
    b"       pts 200 gam 1\x1b[1;37;40m\r\n"
)
FES_RESPONSE = b"75 75 52 52 53 53 0 75 0200 N N N N 53 R\r\n"
FEX_RESPONSE = b"up out swampward southwest south west east north\r\n"
FEI_RESPONSE = b"necklace0\r\n" + INVENTORY_DIVIDER + b"\r\n\r\n"
MGCHEATS_RESPONSE = (
    b"[mgcheats]room_id=groad3; room_name=dally lane; fighting=0; dark=0; glowing=0; "
    b"asleep=0; gifted=0; here=[necklace0, weather, road]; ticks=125; inventory=[][/mgcheats]\r\n"
)
# A canned transport answers every unscripted command the same way; the acknowledgment is the
# captured response to "dance".
OK_RESPONSE = b"\x1b[0;33;40mOK, Alexander the protector \x1b[1;33;40mdances.\x1b[0;33;40m\x1b[1;37;40m\r\n"

AUTO_COMMAND_RESPONSES = {
    "sql": SQL_RESPONSE,
    "fes": FES_RESPONSE,
    "fex": FEX_RESPONSE,
    "fei": FEI_RESPONSE,
    "mgcheats": MGCHEATS_RESPONSE,
}

ScriptedResponse = bytes | tuple[bytes, bool, bool, dict[str, Any]]


def scripted_response(command: str | Sequence[str], *, reset_step: bool = False) -> bytes:
    """Build a prompt-delimited game response for a command batch.

    A single joined line produces the standard one-echo wire format. A split batch (speech
    command, then the auto commands on their own line) produces one echo per line, mirroring how
    the game echoes each sent line separately.
    """
    lines = [command] if isinstance(command, str) else list(command)
    first_line_parts = lines[0].split(",")
    user_command = first_line_parts[0]

    parts = [lines[0].encode("latin-1"), b"\r\n"]
    if reset_step:
        parts.append(TEAROOM_EXIT_TEXT)

    if user_command in AUTO_COMMAND_RESPONSES:
        parts.append(AUTO_COMMAND_RESPONSES[user_command])
    elif user_command == QUICKSCORE_COMMAND:
        parts.append(QS_RESPONSE)
    elif reset_step or user_command == "look":
        parts.append(ROOM_TEXT)
    elif user_command.startswith("say "):
        # the game re-emits your speech in the third person, uniformly coloured
        message = user_command[len("say ") :].encode("latin-1")
        parts.append(
            b'\x1b[0;33;40mAlexander the protector says "\x1b[1;33;40m' + message + b'\x1b[0;33;40m".\x1b[1;37;40m\r\n'
        )
    else:
        parts.append(OK_RESPONSE)

    for auto_command in first_line_parts[1:]:
        parts.append(PROMPT)
        parts.append(AUTO_COMMAND_RESPONSES.get(auto_command, b"OK.\r\n"))

    for extra_line in lines[1:]:
        parts.append(PROMPT)
        parts.append(extra_line.encode("latin-1"))
        parts.append(b"\r\n")
        for position, auto_command in enumerate(extra_line.split(",")):
            if position:
                parts.append(PROMPT)
            parts.append(AUTO_COMMAND_RESPONSES.get(auto_command, b"OK.\r\n"))

    parts.append(PROMPT)
    return b"".join(parts)


class ScriptedConnection(MudConnection):
    """Deterministic transport double that returns canned game bytes to real MudEnv code.

    Responses are keyed by the bare user command (auto-command suffixes are ignored). A bare bytes
    response is treated as a normal non-terminal step. Use the tuple response form when a test needs
    specific terminated/incomplete flags or debug info. Unscripted commands get a generated response.
    """

    def __init__(self, responses: Mapping[str, ScriptedResponse] | None = None):
        super().__init__()
        self.responses = dict(responses or {})
        self.commands: list[str] = []
        self.sent_lines: list[list[str]] = []
        self.stepped_since_reset = False
        self.closed = False

    def reset(self) -> None:
        self.stepped_since_reset = False
        self.closed = False

    def send_command(self, command: str | Sequence[str]) -> tuple[bytes, bool, bool, dict[str, Any]]:
        lines = [command] if isinstance(command, str) else list(command)
        joined = ",".join(lines)
        self.commands.append(joined)
        self.sent_lines.append(lines)
        user_command = lines[0].split(",", 1)[0]
        # reset's identity quickscore happens in the tearoom, before the exit step, so it must not
        # consume the tearoom-exit narration that belongs to the following "move north"
        reset_step = not self.stepped_since_reset and user_command != QUICKSCORE_COMMAND
        if user_command != QUICKSCORE_COMMAND:
            self.stepped_since_reset = True

        response = self.responses.get(user_command)
        if response is None:
            return (
                scripted_response(lines, reset_step=reset_step),
                False,
                False,
                {"scripted": True, "command": joined},
            )
        if isinstance(response, bytes):
            return response, False, False, {"scripted": True, "command": joined}

        raw_bytes, terminated, incomplete, debug_info = response
        return raw_bytes, terminated, incomplete, dict(debug_info)

    def close(self) -> None:
        self.closed = True


def make_scripted_env(
    *,
    responses: Mapping[str, ScriptedResponse] | None = None,
    connection: ScriptedConnection | None = None,
    **kwargs: Any,
):
    if connection is not None and responses is not None:
        raise ValueError("Pass either connection or responses, not both.")

    return make_env(
        connection=connection or ScriptedConnection(responses=responses),
        **kwargs,
    )
