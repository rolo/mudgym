import re

# line break
LINE_BREAK_RE = rb"\r\n"

# Wire bytes above 0x7F are protocol, never text: the game uses them for fecodes (eg \x9b\xff)
# and the unsupported client mode, and its entire text database is 7-bit. Finding one in a text
# path means wire bytes leaked.
NON_TEXT_BYTE_RE = re.compile(rb"[\x80-\xff]")


def decode_text_bytes(data: bytes) -> str:
    """
    Decode game text bytes, raising on bytes above 0x7F.

    Latin-1 is the byte-to-text mapping throughout mudgym (the game predates UTF-8), but the
    game's text output is 7-bit: any high byte in a text path is a leaked protocol byte, so it
    fails loudly with nearby byte context instead of being silently decoded. Diagnostics that
    must render arbitrary wire bytes use decode_wire_bytes instead.

    ANSI escape sequences are valid and are not removed here, callers can strip them separately when plain text
    is wanted.
    """
    raw = bytes(data)

    match = NON_TEXT_BYTE_RE.search(raw)
    if match is not None:
        start = max(0, match.start() - 80)
        end = min(len(raw), match.end() + 80)
        context = raw[start:end]
        raise ValueError(f"Invalid bytes in text bytes at byte {match.start()}: {context!r}")

    return raw.decode("latin-1")


def decode_wire_bytes(data: bytes) -> str:
    """
    Decode arbitrary wire bytes as Latin-1 for diagnostics and logging.

    Total: every byte value maps, so error paths can always render what was on the wire without
    themselves raising and masking the original failure.
    """
    return bytes(data).decode("latin-1")


def encode_command_bytes(text: str) -> bytes:
    """
    Encode a player command line as ASCII, failing clearly before any bytes reach the wire
    when the text contains characters the game cannot receive.

    The game's text channel is 7-bit (see NON_TEXT_BYTE_RE): a high byte sent as a command is
    transliterated rather than echoed back, so the exact-echo anchoring in send_command would
    wait out its timeout instead of matching.
    """
    try:
        return text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"command contains characters outside ASCII: {text!r}") from exc
