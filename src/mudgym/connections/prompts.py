"""Patterns for parsing MUD2 game output and recorded transcripts."""

import enum
import re

# One-or-more variant for named-capture contexts (where matching empty is wrong)
SGR_ONE_PLUS_STR = r"(?:\x1b\[[0-9;]*m)+"

# game prompts:
# mortal:
# *
# (*)
#
# wiz:
# ----*
# (----*)
# ((----*))
# (((----*)))
#
# I'm not sure if a mortal can become double or triple invisible, I don't think during
# the regular course of play but possibly a wiz can do it and makes the regex simpler so can't hurt
# to support it.

# Optional ANSI SGR (bytes version for binary patterns)
SGR = rb"(?:\x1b\[[0-9;]*m)*"
SGR_ONE_PLUS_BYTES = rb"(?:\x1b\[[0-9;]*m)+"

# Prompt pieces (bytes)
DASHES = rb"-{4,}"

STARLINE = rb"(?:\({1,3})?(?:" + DASHES + rb")?\*(?:\){0,3})?"
PROMPT_CORE = rb"(?:" + STARLINE + rb"|" + DASHES + rb")"

# Possessive so a trailing colour code cannot be given back to sneak the boundary's
# not-a-complete-line lookahead past a \r\n
SGR_POSSESSIVE = rb"(?:\x1b\[[0-9;]*m)*+"

# the game can drop one of these in at any moment, including right behind a prompt, where its
# leading break would otherwise make the prompt look like a complete line
DATABASE_BROADCAST_LINE = rb"(?:\r?\n)+\+- (?:Database \d+|The database) has finished initialising"

# A genuine input prompt on the wire starts a physical line and is never a complete line: it
# dangles awaiting input, or continues with the echo of whatever the player types next. Spoken
# copies sit mid-line behind the speech quoting and narrative copies end in \r\n, so the two
# anchors together reject both. A broadcast behind a prompt does not make it a complete line.
DANGLES = rb"(?:(?![\r\n])|(?=" + DATABASE_BROADCAST_LINE + rb"))"

NEXT_PROMPT_BOUNDARY = rb"(?m:^)" + SGR + PROMPT_CORE + SGR_POSSESSIVE + DANGLES


def regex_up_to_next_prompt(needle: bytes, extra_flags: int = 0) -> re.Pattern:
    """
    Return a regex that matches up to the next game prompt beyond the given match.

    This should give us an easy and consistent way to match game responses of the form "any stuff
    in a response matching this but then up to the next prompt". The terminator stops at the first
    genuine prompt, so responses that ran ahead of the reader don't swallow the following
    command's echo and marker.
    """
    return re.compile(
        needle + rb".*?" + NEXT_PROMPT_BOUNDARY,
        re.MULTILINE | re.DOTALL | extra_flags,
    )


def system_line_up_to_next_prompt(needle: bytes) -> re.Pattern:
    """Match a game-generated response line through its following input prompt.

    The line anchor and optional colour prefix keep player speech and incidental copies of the
    response text from being mistaken for front-end protocol output.
    """
    return regex_up_to_next_prompt(rb"^" + SGR + needle)


class Prompt(enum.Enum):
    # Prompts which explicitly mark the end of an episode. Each is anchored to a whole wire line
    # (optionally colour-wrapped) because the bare words are forgeable: a player speaking
    # "Cheerio!" puts the text mid-line inside the speech quoting, and the command echo repeats
    # whatever the player typed. The genuine lines carry nothing but the message itself.
    GAME_OVER_EPISODE_POINTS = re.compile(
        rb"(?m)^" + SGR + rb"Overall, you (?:scored|lost) [\d,]+ points this game\." + SGR + rb"\r?$"
    )
    GAME_OVER_QUIT_CHEERIO = re.compile(rb"(?m)^" + SGR + rb"Cheerio!" + SGR + rb"\r?$")
    GAME_OVER_NOT_UPDATING_PERSONA = re.compile(
        rb"(?m)^" + SGR_ONE_PLUS_BYTES + rb"Not updating persona\." + SGR_ONE_PLUS_BYTES + rb"\r?$"
    )
    GAME_OVER_KILLED_FOR_SWEARING = re.compile(
        rb"(?m)^" + SGR + rb"In order to keep the game uncorrupted,\s+you have been killed\.\s*"
        rb"\(Persona saved on\s+[+-][\d,]+\s*=\s*(?:\x1b\[[0-9;]*m)+\d[\d,]*(?:\x1b\[[0-9;]*m)*\)\."
    )


GAME_OVER_PROMPTS = [
    Prompt.GAME_OVER_EPISODE_POINTS,
    Prompt.GAME_OVER_QUIT_CHEERIO,
    Prompt.GAME_OVER_NOT_UPDATING_PERSONA,
    Prompt.GAME_OVER_KILLED_FOR_SWEARING,
]


COMMAND_REJECTION_LINE_PATTERNS = (
    b"I made sense of some of that:",
    b"I made no sense of that:",
    rb'I don\'t know the word "(\w+)".',
    rb"I don't know to what \"(\w+)\" you're referring.",
    b"Your command is too long for me, sorry!",
)
INVALID_COMMAND_PROMPTS = [
    system_line_up_to_next_prompt(line_pattern) for line_pattern in COMMAND_REJECTION_LINE_PATTERNS
]
