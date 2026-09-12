from mudgym.connections.prompts import GAME_OVER_PROMPTS, Prompt


def has_game_over_prompt(raw_bytes: bytes) -> bool:
    """Return whether trusted game bytes contain a game-over prompt."""
    return any(prompt.value.search(raw_bytes) for prompt in GAME_OVER_PROMPTS)


def is_permadeath(raw_bytes: bytes) -> bool:
    """
    Permadeath (eg, combat, touchstone, dragon flee) prints no points change events so we trap it
    specifically.
    """
    return Prompt.GAME_OVER_NOT_UPDATING_PERSONA.value.search(raw_bytes) is not None
