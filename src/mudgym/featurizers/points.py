import re

# A single ANSI SGR (colour) sequence.
SGR = rb"(?:\x1b\[[0-9;]*m)"

# A genuine points change event always colours the resulting total (green for gains, red for
# losses), eg "(Persona saved on +12,800 = \x1b[0;32;40m13,000\x1b[1;37;40m)". Player-authored
# text cannot reproduce that: command echoes are plain, spoken text is recoloured uniformly at
# the line level, and the game strips raw ESC bytes from input (a typed ESC echoes as "^[").
# Anchoring on the SGR before the total therefore rejects forged patterns such as
# 'say hello (+10 = 10)'. Everything must sit on one line: player speech that wraps gets colour
# re-applied mid-message, which could otherwise fake the anchor across a line break.
POINTS_CHANGE_PATTERN = re.compile(
    rb"\((?:[^()\r\n\x1b]|"
    + SGR
    + rb")*?"
    + SGR
    + rb"*(?P<reward>[+-][\d,]+)"
    + SGR
    + rb"*"
    + rb"[ \t]*=[ \t]*"
    + SGR
    + rb"+(?P<points>\d[\d,]*)"
    + SGR
    + rb"*"
    + rb"(?:[^()\r\n\x1b]|"
    + SGR
    + rb")*\)"
)


def parse_points_changes(
    raw_bytes: bytes,
) -> dict[str, int | None]:
    """
    Parse points change events found in *raw_bytes* game output.

    We remain in bytes for this so it can be used in envs that don't operate on text. The change
    pattern matches against the raw bytes including ANSI colour, because the colour around the
    total is what distinguishes a genuine game event from player-authored text.

    Returns
    -------
    dict with keys:
        • delta : int (total change in points, always an int)
        • points : int | None (final points total from the last trusted change event)
    """

    delta = 0
    points = None

    # points change pattern: eg, "(+100 = \x1b[0;32;40m300\x1b[1;37;40m)"
    for match in POINTS_CHANGE_PATTERN.finditer(raw_bytes):
        delta += int(match.group("reward").replace(b",", b""))
        points = int(match.group("points").replace(b",", b""))

    return {"delta": delta, "points": points}
