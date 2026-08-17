"""
Prompts module for MUD2 connections.

Defines the various matching patterns for pexpect "prompts" to drive the state machine when
when connecting to and interacting with the MUD2 game and management software.
"""

import enum
import functools
import re

import pexpect

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

# Optional terminal CSI controls at the start of a raw wire line. Besides colour changes, menu
# redraws can emit screen-clear and cursor-positioning controls before their input prompt.
CSI_CONTROLS = rb"(?:\x1b\[[0-9;?]*[ -/]*[@-~])*"


def trusted_input_prompt(pattern: bytes, flags: int = 0) -> re.Pattern:
    """
    Compile an input prompt against its trusted raw-wire prefix.

    MUD2 emits menu, login, and pager input points at the start of a physical line, sometimes
    after terminal controls. Spoken or incidental copies have narrative text before the
    player-controlled words. Command echoes need an additional dynamic trust boundary: the
    command loop consumes every known echo before it considers these prompt patterns.
    """
    return re.compile(rb"(?m)^" + CSI_CONTROLS + pattern, flags)


# Prompt pieces (bytes)
DASHES = rb"-{4,}"

STARLINE = rb"(?:\({1,3})?(?:" + DASHES + rb")?\*(?:\){0,3})?"
PROMPT_CORE = rb"(?:" + STARLINE + rb"|" + DASHES + rb")"

# Possessive so a trailing colour code cannot be given back to sneak the boundary's
# not-a-complete-line lookahead past a \r\n
SGR_POSSESSIVE = rb"(?:\x1b\[[0-9;]*m)*+"

# A genuine input prompt on the wire starts a physical line and is never a complete line: it
# dangles awaiting input, or continues with the echo of whatever the player types next. Spoken
# copies sit mid-line behind the speech quoting and narrative copies end in \r\n, so the two
# anchors together reject both.
NEXT_PROMPT_BOUNDARY = rb"(?m:^)" + SGR + PROMPT_CORE + SGR_POSSESSIVE + rb"(?![\r\n])"

TEAROOM_PROMPT_PATTERN = (
    rb"^"
    + SGR
    + rb"(?:Elizabethan tearoom|Wizzes' room|Control room)"
    + SGR
    + rb".*?"
    + SGR
    + rb"Players:"
    + SGR
    + rb"\r?\n"
    + rb"(?P<players>.*?)"
    + rb"(?="
    + NEXT_PROMPT_BOUNDARY
    + rb")"
)

TEAROOM_PROMPT = re.compile(TEAROOM_PROMPT_PATTERN, re.DOTALL | re.MULTILINE)


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
    EOF = pexpect.EOF
    TIMEOUT = pexpect.TIMEOUT

    OPTION = trusted_input_prompt(rb"Option\s*(\(H for help\))?\s*:")
    TEAROOM = TEAROOM_PROMPT

    # where we chop the text at episode start upon entering The Land
    ENTERED_LAND = re.compile(rb"nothingness...")

    # * - mortal
    # (*) - invisible mortal
    # ((*)) - double invisible mortal (is this possible?)
    # ----* - wizard
    # (----*) - invisible wizard
    # ((----*)) - double invisible wizard
    # (((----*))) - triple invisible wizard
    # we anchor them like NEXT_PROMPT_BOUNDARY (line start, never a complete line).
    # A bare star stays bold-only because the login menus reprint a plain star ahead of
    # each echoed answer, which must not read as being in game. Possessive repeats so a spoof
    # can't shed its own trailing characters to satisfy the lookahead.
    GAME = re.compile(
        rb"(?m:^)"
        + SGR
        + rb"(?:"
        + rb"\({1,3}+"
        + SGR
        + rb"(?:-{4,}+)?\*"
        + SGR
        + rb"\){1,3}+"  # (*), ((----*)), ...
        + rb"|-{4,}+\*"  # ----*
        + rb"|\x1b\[1;[0-9;]*m\*\x1b\[[0-9;]*m"  # bold coloured mortal star
        + rb")"
        + SGR_POSSESSIVE
        + rb"(?![\r\n])"
    )

    TEA_SIPPED = regex_up_to_next_prompt(rb"You watch the world go by\.")

    # login prompts
    SUPERSEDE = trusted_input_prompt(rb"Do you want to supersede this other session\?")
    SESSION_DYING = trusted_input_prompt(rb"Session is dying")

    # persona selection prompts
    PERSONA_AVAILABLE = trusted_input_prompt(rb"By what name shall I call you \(Q to quit\)\?")
    PERSONA_NAME = trusted_input_prompt(rb"What shall I call you[^\r\n]*\?")
    PERSONA_SEX = trusted_input_prompt(rb"What sex do you wish to be\?")

    # consolidate variants into single patterns:
    RESET_IN_PROGRESS = re.compile(rb"(?:Reset in progress\.|The database is still initialising\.)")
    BOOT_COMPLETE = b"Boot complete."

    DATABASE_NOT_INITIALIZED = re.compile(rb"Database \d+ is not initialised\.")
    DATABASE_FINISHED_INITIALIZING = re.compile(rb"(?:Database \d+ has|The database has) finished initialising")
    FECODE_ZERO = b"\r\n\x9b\xff"

    # non game prompts
    EXAMINE = trusted_input_prompt(rb"EXAMINE>")
    LIBRARY = trusted_input_prompt(rb"LIBRARY>")
    PAGER = trusted_input_prompt(rb"\[Return to continue, S to stop\]" + SGR, re.I)
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


# A single prompt, or a collection of prompts, as accepted by expect() and the
# `initial_prompt` hooks. These are passed straight through to pexpect, which is
# happy with either one pattern or several.
PromptSpec = Prompt | list[Prompt] | tuple[Prompt, ...]

GAME_OVER_PROMPTS = [
    Prompt.GAME_OVER_EPISODE_POINTS,
    Prompt.GAME_OVER_QUIT_CHEERIO,
    Prompt.GAME_OVER_NOT_UPDATING_PERSONA,
    Prompt.GAME_OVER_KILLED_FOR_SWEARING,
]

# the game sends these to every session, one per database slot, so they say nothing about us
BROADCAST_PROMPTS = frozenset({Prompt.DATABASE_FINISHED_INITIALIZING})

# Use an explicit order for mapping idx -> Prompt
PROMPTS: tuple[Prompt, ...] = (
    Prompt.EOF,
    Prompt.TIMEOUT,
    Prompt.TEAROOM,
    Prompt.TEA_SIPPED,
    Prompt.ENTERED_LAND,
    Prompt.GAME,
    Prompt.OPTION,
    Prompt.SUPERSEDE,
    Prompt.SESSION_DYING,
    Prompt.PERSONA_SEX,
    Prompt.PERSONA_NAME,
    Prompt.EXAMINE,
    Prompt.PERSONA_AVAILABLE,
    Prompt.RESET_IN_PROGRESS,
    Prompt.BOOT_COMPLETE,
    Prompt.FECODE_ZERO,
    Prompt.DATABASE_NOT_INITIALIZED,
    Prompt.DATABASE_FINISHED_INITIALIZING,
    Prompt.PAGER,
    Prompt.LIBRARY,
    Prompt.GAME_OVER_EPISODE_POINTS,
    Prompt.GAME_OVER_QUIT_CHEERIO,
    Prompt.GAME_OVER_NOT_UPDATING_PERSONA,
    Prompt.GAME_OVER_KILLED_FOR_SWEARING,
)


class State(enum.Enum):
    INITIAL = enum.auto()
    LOGIN = enum.auto()
    OPTION = enum.auto()
    PERSONA_SELECT = enum.auto()  # choosing slot
    PERSONA_NAME_INPUT = enum.auto()  # entering name
    PERSONA_SEX_INPUT = enum.auto()  # choosing sex
    TEAROOM = enum.auto()  # in the Elizabethan tearoom
    TEA_SIPPED = enum.auto()  # in the tearoom after sipping tea
    GAME = enum.auto()  # in game, having stepped out of the tearoom
    GAME_OVER = enum.auto()  # in game over, having left the game
    CLOSING = enum.auto()  # backing out of the menus to log the account out
    RESETTING = enum.auto()  # in the menus, waiting for a reset to complete
    DEAD = enum.auto()  # ops dead rather than game dead


EXPECT_LIST = [p.value for p in PROMPTS]  # for expect()


@functools.lru_cache
def marker_up_to_next_prompt(marker: re.Pattern) -> re.Pattern:
    """Wire form of an end-of-turn marker: the marker's line then everything up to the next prompt."""
    return regex_up_to_next_prompt(marker.pattern, extra_flags=marker.flags)


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
