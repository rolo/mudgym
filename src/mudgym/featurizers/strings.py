# line break
LINE_BREAK_RE = rb"\r\n"


def decode_text_bytes(data: bytes) -> str:
    """
    Decode game text bytes, raising on bytes above 0x7F.

    The game's text output is 7-bit. A high byte in a text path is a leaked protocol byte, so it
    fails loudly with nearby byte context. Diagnostics that must render arbitrary wire bytes use
    ``decode_wire_bytes`` instead.

    ANSI escape sequences are valid and are not removed here, callers can strip them separately when plain text
    is wanted.
    """
    raw = bytes(data)
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as exc:
        start = max(0, exc.start - 80)
        end = min(len(raw), exc.end + 80)
        raise ValueError(f"Invalid text byte at byte {exc.start}: {raw[start:end]!r}") from exc


def decode_text_lines(data: bytes) -> list[str]:
    """Decode newline-delimited game text, stripping ASCII whitespace around each line."""
    return [decode_text_bytes(line.strip()) for line in bytes(data).split(b"\n")]


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

    The game's text channel is 7-bit: a high byte sent as a command is
    transliterated rather than echoed back, so the read window's exact-echo anchoring would
    wait out its timeout instead of matching.
    """
    try:
        return text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"command contains characters outside ASCII: {text!r}") from exc
