import re

# The game colours its text with SGR codes and clears the screen with a CSI. It never sends a terminated
# string (OSC, DCS), so none is handled: ESC ] or ESC P is a two-byte escape here, its payload left as text.
ansi_escape_bytes = re.compile(
    rb"""
    \x1B  # ESC
    (?:  # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |    # CSI sequence
        \[ [0-?]* [ -/]* [@-~]
    )
    """,
    re.VERBOSE,
)


def strip_ansi(data: bytes) -> bytes:
    """Remove ANSI escape sequences from a byte string."""
    return ansi_escape_bytes.sub(b"", data)
