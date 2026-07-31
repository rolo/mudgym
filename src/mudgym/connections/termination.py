from mudgym.connections.prompts import GAME_OVER_PROMPTS, INVALID_COMMAND_PROMPTS, Prompt

# terminal conditions we watch for while waiting for our own command echo to anchor the read window.
# order matters for expect(): game-over first, then transport terminal conditions.
COMMAND_ECHO_BREAK_PATTERNS = [
    *[prompt.value for prompt in GAME_OVER_PROMPTS],
    Prompt.EOF.value,
    Prompt.TIMEOUT.value,
]

# prompts that tell us the session left the game and is sitting at another input point (menu,
# pager, login flow), so a step's end-of-turn marker is never coming. Deliberately narrower than
# NON_GAME_PROMPTS: that list also holds status lines the game broadcasts into live sessions
# mid-step ("The database has finished initialising"), which must never cut a step short.
NO_LONGER_IN_GAME_PROMPTS = [
    Prompt.OPTION,
    Prompt.EXAMINE,
    Prompt.LIBRARY,
    Prompt.PAGER,
    Prompt.SUPERSEDE,
    Prompt.SESSION_DYING,
    Prompt.PERSONA_AVAILABLE,
    Prompt.PERSONA_NAME,
    Prompt.PERSONA_SEX,
]

# everything besides the end-of-turn marker itself (configured on the state machine) that can
# complete send_command's read window: a game-over, no longer being in the game, transport
# EOF/TIMEOUT, or a rejected command that aborts the rest of the line. Prompt.GAME is included so
# each in-window response resets the expect timeout rather than the whole batch having to arrive
# inside a single expect window.
END_OF_STEP_PATTERNS = [
    *[prompt.value for prompt in GAME_OVER_PROMPTS],
    Prompt.EOF.value,
    Prompt.TIMEOUT.value,
    *[prompt.value for prompt in NO_LONGER_IN_GAME_PROMPTS],
    *INVALID_COMMAND_PROMPTS,
    Prompt.GAME.value,
]

# transport-level conditions that leave a step's read window incomplete: no end-of-turn marker arrived
TRANSPORT_BREAK_PROMPTS = (Prompt.EOF, Prompt.TIMEOUT)


def is_game_over_output(raw_bytes: bytes) -> bool:
    """Return True when raw game bytes contain a game-over prompt."""
    return any(prompt.value.search(raw_bytes) for prompt in GAME_OVER_PROMPTS)
