"""
Response processing for game bytes returned each step.

We send a command to the game and receive a response payload as a blob of bytes.

This payload will include any game events which have occurred since the last step, then our own command (including any
auto commands) echoed back, then any game events in response to our command. Each game event is delimited by a prompt
marker (*) to tell the user the game is ready for the next command.

The bytes map to text as Latin-1 (never UTF-8), but the game's text output is 7-bit: printable ASCII plus
ANSI escape sequences, sent over telnet. Bytes above 0x7F are protocol codes (fecodes, the unsupported
"client mode"), never text, and text paths reject them loudly as leaks. Diagnostics can still render
arbitrary wire bytes without masking the original error.

The functions in this module are for splitting these bytes in some different variations:

`raw_bytes` - the raw bytes payload as it comes from the game including ANSI and prompt markers.
`chunks` - the bytes split into a list of chunks delimited by prompt markers, with echo line removed.

No awareness of env, specs or RL concepts here.
"""

import re

from mudgym.connections.prompts import SGR, STARLINE
from mudgym.featurizers.strings import LINE_BREAK_RE, encode_command_bytes

# pattern for splitting by prompt markers. The line anchor keeps inline asterisks out of routing.
RESPONSE_PROMPT_RE = re.compile(rb"(?m)^" + SGR + STARLINE + SGR)

# the echo line if a newline, possibly some ANSI/control bytes and a prompt marker, our command and then a newline.
ECHO_CONTROL_PREFIX = rb"(?:\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x09\x0b-\x1a\x1c-\x1f])"
ECHO_PROMPT_PREFIX = SGR + STARLINE + SGR
ECHO_PREFIX = rb"(?:" + ECHO_CONTROL_PREFIX + rb"|" + ECHO_PROMPT_PREFIX + rb")*+"


def echo_pattern(last_command_joined: str | bytes) -> re.Pattern[bytes]:
    """
    Build a regex matching the echoed command line, optionally preceded by ANSI/control bytes.

    Example matches:

        b"move north,sql,fes,fex,fei\r\n"
        b"\x1b[1;37;40mmove east,sql,fes,fex,fei\r\n"
    """
    command_bytes = (
        last_command_joined if isinstance(last_command_joined, bytes) else encode_command_bytes(last_command_joined)
    )

    return re.compile(rb"(?m)^" + ECHO_PREFIX + re.escape(command_bytes) + LINE_BREAK_RE)


def contains_echo(raw_bytes: bytes, last_command_joined: str | bytes) -> bool:
    """
    Check if the raw bytes contain the echoed command line.
    """
    return echo_pattern(last_command_joined).search(raw_bytes) is not None


def split_on_echo(raw_bytes: bytes, last_command_joined: str | bytes) -> tuple[bytes, bytes]:
    """
    Split raw bytes around the echoed command line, removing the echo itself.

    Returns:
        (pre_echo_bytes, post_echo_bytes)
    """
    match = echo_pattern(last_command_joined).search(raw_bytes)
    if match is None:
        raise ValueError(f"Echoed command {last_command_joined!r} not found in raw bytes: {raw_bytes!r}")

    return raw_bytes[: match.start()], raw_bytes[match.end() :]


def split_on_echo_lines(raw_bytes: bytes, echo_lines: list[str | bytes]) -> list[bytes] | None:
    """
    Split raw bytes around each echoed line in order, removing the echoes themselves.

    A step's batch can go out on more than one wire line (speech commands take the auto commands
    on a separate line), and each sent line is echoed separately. Returns len(echo_lines) + 1
    segments, or None when any echo is missing from the bytes.
    """
    segments: list[bytes] = []
    remainder = raw_bytes
    for echo_line in echo_lines:
        match = echo_pattern(echo_line).search(remainder)
        if match is None:
            return None
        segments.append(remainder[: match.start()])
        remainder = remainder[match.end() :]
    segments.append(remainder)
    return segments


def split_on_prompt(raw_bytes: bytes) -> list[bytes]:
    """
    Split raw bytes into a list of chunks delimited by prompt markers.
    We discard the final chunk, which should be empty, as the game always sends a prompt marker after the last command.
    """
    chunks = RESPONSE_PROMPT_RE.split(raw_bytes)
    # the final chunk will be empty if a new prompt marker came after it so we discard it.
    if chunks[-1] == b"":
        return chunks[:-1]

    # if it's not empty it means the game didn't send a prompt marker which happens upon episode end (death usually),
    # so the final chunk is real content and we keep it. I'm logging this as it's helpful to see when it happens.
    # print(f"Non-empty final chunk in {raw_bytes!r} was {chunks[-1]!r}.")
    return chunks


def normalise_lines(raw_bytes: bytes) -> bytes:
    """
    Normalise line breaks from \r\n to \n.
    """
    return raw_bytes.replace(b"\r\n", b"\n")
