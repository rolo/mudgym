from collections.abc import Mapping, Sequence
from typing import Any

from mudgym.connections.connection import MudConnection
from mudgym.envs.factory import make_env
from mudgym.envs.fields.feinventory import INVENTORY_DIVIDER
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

OBSERVATION_COMMAND_RESPONSES = {
    "sql": SQL_RESPONSE,
    "fes": FES_RESPONSE,
    "fex": FEX_RESPONSE,
    "fei": FEI_RESPONSE,
    "mgcheats": MGCHEATS_RESPONSE,
}

ScriptedResponse = bytes | tuple[bytes, bool, bool, dict[str, Any]]


def scripted_response(lines: Sequence[str], *, reset_step: bool = False) -> bytes:
    """Build a prompt-delimited game response for the supplied wire lines."""
    lines = list(lines)
    if len(lines) == 1:
        observation_line = lines[0]
        parts = [observation_line.encode("latin-1"), b"\r\n"]
        for position, observation_command in enumerate(observation_line.split(",")):
            if position:
                parts.append(PROMPT)
            parts.append(OBSERVATION_COMMAND_RESPONSES.get(observation_command, b"OK.\r\n"))
        parts.append(PROMPT)
        return b"".join(parts)
    if len(lines) != 2:
        raise ValueError(f"Expected an action and observation line, or an observation line alone: {lines!r}")

    user_command = lines[0]

    parts = [lines[0].encode("latin-1"), b"\r\n"]
    if reset_step:
        parts.append(FES_RESPONSE)
        parts.append(PROMPT)
        parts.append(TEAROOM_EXIT_TEXT)

    if user_command in OBSERVATION_COMMAND_RESPONSES:
        parts.append(OBSERVATION_COMMAND_RESPONSES[user_command])
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

    parts.append(PROMPT)
    parts.append(scripted_response(lines[1:]))
    return b"".join(parts)


class ScriptedConnection(MudConnection):
    """Deterministic transport double that returns canned game bytes to real MudEnv code.

    Responses are keyed by the player's command line. A bare bytes response is treated as a normal
    non-terminal step. Use the tuple response form when a test needs specific terminated/incomplete
    flags or debug info. Unscripted commands get a generated response.
    """

    def __init__(
        self,
        responses: Mapping[str, ScriptedResponse] | None = None,
        *,
        send_errors: Mapping[str, Exception] | None = None,
    ):
        super().__init__()
        self.responses = dict(responses or {})
        self.send_errors = dict(send_errors or {})
        self.sent_lines: list[list[str]] = []
        self.pending_lines: list[str] = []
        self.read_markers = []
        self.entered_land = False
        self.invalidated = False
        self.closed = False

    def reset(self) -> None:
        self.pending_lines.clear()
        self.entered_land = False
        self.invalidated = False
        self.closed = False

    def send_line(self, line: str) -> None:
        if error := self.send_errors.get(line):
            raise error
        self.pending_lines.append(line)

    def read_response(self, end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        self.read_markers.append(end_of_turn_marker)
        lines = list(self.pending_lines)
        self.pending_lines.clear()
        raw_bytes, terminated, incomplete, debug_info = self.complete_command(lines)
        debug_info["sent_lines"] = lines
        return raw_bytes, terminated, incomplete, debug_info

    def complete_command(self, lines: list[str]) -> tuple[bytes, bool, bool, dict[str, Any]]:
        self.sent_lines.append(lines)
        user_command = lines[0]
        reset_step = user_command == "fes,move north" and not self.entered_land
        if reset_step:
            self.entered_land = True

        response = self.responses.get(user_command)
        if response is None:
            return (
                scripted_response(lines, reset_step=reset_step),
                False,
                False,
                {"scripted": True, "marker_arrived": True},
            )
        if isinstance(response, bytes):
            return response, False, False, {"scripted": True, "marker_arrived": True}

        raw_bytes, terminated, incomplete, debug_info = response
        return raw_bytes, terminated, incomplete, dict(debug_info)

    def invalidate(self) -> None:
        self.pending_lines.clear()
        self.invalidated = True

    def close(self) -> None:
        self.closed = True


class NoOpProvider:
    """Provider double for tests that construct a coordinator around existing children."""

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        pass

    def close(self) -> None:
        pass


class ScriptedProvider:
    """Supply scripted connections while exposing provider lifecycle calls to tests."""

    def __init__(self, *, returned_count: int | None = None):
        self.returned_count = returned_count
        self.connections: list[ScriptedConnection] = []
        self.requested_count: int | None = None
        self.reset_seeds: list[int | list[int | None] | None] = []
        self.closed = False

    def create_connections(self, count: int) -> list[MudConnection]:
        self.requested_count = count
        returned_count = count if self.returned_count is None else self.returned_count
        self.connections = [ScriptedConnection() for _ in range(returned_count)]
        return list(self.connections)

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        self.reset_seeds.append(seed)

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
