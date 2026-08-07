"""
Quickscore command. Used at the start of a session to get persona name.

Example, ANSI stripped:

Sir Dood
eff str 69      eff dex 63      sta 48/48       pts 140,000     gam 2

Magic user:

Dood the mage
eff str 68      eff dex 59      sta 28/48       mag 48  pts 140,000     gam 2

The first line is the persona's full name, so it carries whatever level, prefix and postfix they
have, and Sir/Lady and Brother/Sister put the title first: bare_persona_name knows the shapes.
Only the columns up to stamina are matched, which is what keeps a magic user's extra mag column
from mattering here.
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

QUICKSCORE_NAME_PATTERN = re.compile(rb"(?m)^" + SGR + rb"(?P<full_name>[A-Za-z][^\r\n]*?)\r?\n" + SGR + STATS_LINE)


def parse_quickscore_name(raw_bytes: bytes) -> str:
    """Parse the current persona's bare name from a quickscore response."""
    match = QUICKSCORE_NAME_PATTERN.search(raw_bytes)
    if match is None:
        raise ValueError(f"no quickscore name line found in: {raw_bytes!r}")
    full_name = decode_text_bytes(strip_ansi(match.group("full_name")))
    return bare_persona_name(full_name)
