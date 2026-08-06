"""
Quickscore command. Used at the start of a session to get persona name.

Example, ANSI stripped:

    Alexander the protector
    eff str 52      eff dex 53      sta 75/75       pts 200 gam 1
"""

import re

from mudgym.connections.prompts import SGR
from mudgym.featurizers.strings import decode_text_bytes

QUICKSCORE_COMMAND = "qs"  # qs or quickscore

# The FE expands tab-separated columns to spaces so accept either
COLUMN_GAP = SGR + rb"[ \t]+" + SGR
VALUE = rb"\d+" + SGR
STAMINA = VALUE + rb"/" + SGR + rb"\d+"
STATS_LINE = COLUMN_GAP.join([rb"eff str", VALUE, rb"eff dex", VALUE, rb"sta", STAMINA])

QUICKSCORE_NAME_PATTERN = re.compile(rb"(?m)^" + SGR + rb"(?P<name>[A-Za-z]+)[^\r\n]*\r?\n" + SGR + STATS_LINE)


def parse_quickscore_name(raw_bytes: bytes) -> str:
    """Parse the current persona's bare name from a quickscore response."""
    match = QUICKSCORE_NAME_PATTERN.search(raw_bytes)
    if match is None:
        raise ValueError(f"no quickscore name line found in: {raw_bytes!r}")
    return decode_text_bytes(match.group("name"))
