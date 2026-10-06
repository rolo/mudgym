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
#
# decorations, each in its own colour, eg b"\x1b[0;34;40m(\x1b[1;34;40m*\x1b[0;34;40m)":
# >*    conversing
# &*    snooping (wiz)
# [*]   locally visible (wiz)

# Optional ANSI SGR (bytes version for binary patterns)
SGR = rb"(?:\x1b\[[0-9;]*m)*"
SGR_ONE_PLUS_BYTES = rb"(?:\x1b\[[0-9;]*m)+"

# Prompt pieces (bytes)
DASHES = rb"-{4,}"

DECORATIONS_BEFORE = rb"(?:&" + SGR + rb")?(?:>" + SGR + rb")?(?:\(" + SGR + rb"){0,3}(?:\[" + SGR + rb")?"
DECORATIONS_AFTER = rb"(?:" + SGR + rb"\])?(?:" + SGR + rb"\)){0,3}"

STARLINE = DECORATIONS_BEFORE + rb"(?:" + DASHES + rb")?\*" + DECORATIONS_AFTER


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


# The parser's replies to a line it cannot read. Each is the game's own line, so it starts a line
# (optionally colour-wrapped), while a spoken copy sits mid-line behind the speech quoting.
COMMAND_REJECTION_LINES = [
    re.compile(rb"(?m)^" + SGR + message)
    for message in (
        b"I made sense of some of that:",
        b"I made no sense of that:",
        rb'I don\'t know the word "(\w+)".',
        rb"I don't know to what \"(\w+)\" you're referring.",
        b"Your command is too long for me, sorry!",
    )
]
