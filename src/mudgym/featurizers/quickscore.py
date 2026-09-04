"""
Quickscore command. Used at the start of a session to get the persona name and points.

Example, ANSI stripped:

Sir Dood
eff str 69      eff dex 63      sta 48/48       pts 140,000     gam 2

Magic user:

Dood the mage
eff str 68      eff dex 59      sta 28/48       mag 48  pts 140,000     gam 2

The first line is the persona's full name, so it carries whatever level, prefix and postfix they
have, and Sir/Lady and Brother/Sister put the title first: bare_persona_name knows the shapes.
The magic user's mag column is optional.
"""

import re

from mudgym.connections.prompts import SGR
from mudgym.featurizers.ansi import strip_ansi
from mudgym.featurizers.persona_names import bare_persona_name
from mudgym.featurizers.strings import decode_text_bytes

QUICKSCORE_COMMAND = "qs"  # qs or quickscore

# The FE expands tab-separated columns to spaces so accept either
COLUMN_GAP = SGR + rb"[ \t]+" + SGR
VALUE = rb"\d+" + SGR
STAMINA = VALUE + rb"/" + SGR + rb"\d+"
STATS_LINE = COLUMN_GAP.join([rb"eff str", VALUE, rb"eff dex", VALUE, rb"sta", STAMINA])
MAGIC_COLUMN = rb"(?:" + COLUMN_GAP + rb"mag" + COLUMN_GAP + VALUE + rb")?"
POINTS_COLUMN = COLUMN_GAP.join([rb"pts", rb"(?P<points>\d[\d,]*)"])

QUICKSCORE_PATTERN = re.compile(
    rb"(?m)^"
    + SGR
    + rb"(?P<full_name>[A-Za-z][^\r\n]*?)\r?\n"
    + SGR
    + STATS_LINE
    + MAGIC_COLUMN
    + COLUMN_GAP
    + POINTS_COLUMN
)


def parse_quickscore(raw_bytes: bytes) -> tuple[str, int]:
    """Parse a quickscore response into the current persona's bare name and points."""
    match = QUICKSCORE_PATTERN.search(raw_bytes)
    if match is None:
        raise ValueError(f"no quickscore found in: {raw_bytes!r}")
    full_name = decode_text_bytes(strip_ansi(match.group("full_name")))
    points = int(match.group("points").replace(b",", b""))
    return bare_persona_name(full_name), points
