from mudgym.connections.prompts import GAME_OVER_PROMPTS, INVALID_COMMAND_PROMPTS, Prompt

# prompts that tell us the session left the game and is sitting at another input point (menu,
# pager, login flow), so a step's end-of-turn marker is never coming. Deliberately excludes status
# lines the game broadcasts into live sessions mid-step ("The database has finished initialising"),
# which must never cut a step short.
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

# Patterns read() handles before the marker. Prompt.GAME restarts the timeout for each response.
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


def is_permadeath(raw_bytes: bytes) -> bool:
    """
    Permadeath (eg, combat, touchstone, dragon flee) prints no points change events so we trap it
    specifically.
    """
    return Prompt.GAME_OVER_NOT_UPDATING_PERSONA.value.search(raw_bytes) is not None
