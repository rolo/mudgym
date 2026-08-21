from mudgym.connections.prompts import GAME_OVER_PROMPTS, Prompt, State
from mudgym.connections.termination import END_OF_STEP_PATTERNS, NO_LONGER_IN_GAME_PROMPTS, is_permadeath
from mudgym.connections.transitions import TRANSITIONS

# The game broadcasts status lines into live sessions mid-step
ASYNC_BROADCAST_PROMPTS = [
    Prompt.DATABASE_FINISHED_INITIALIZING,
    Prompt.DATABASE_NOT_INITIALIZED,
    Prompt.RESET_IN_PROGRESS,
    Prompt.BOOT_COMPLETE,
    Prompt.FECODE_ZERO,
]


def test_async_status_broadcasts_do_not_end_a_step():
    for prompt in ASYNC_BROADCAST_PROMPTS:
        assert prompt not in NO_LONGER_IN_GAME_PROMPTS
        assert prompt.value not in END_OF_STEP_PATTERNS


def test_input_awaiting_prompts_do_end_a_step():
    """Prompts that sit waiting for input mean the fei marker is never coming."""
    for prompt in [Prompt.OPTION, Prompt.PAGER, Prompt.SUPERSEDE, Prompt.PERSONA_NAME]:
        assert prompt in NO_LONGER_IN_GAME_PROMPTS
        assert prompt.value in END_OF_STEP_PATTERNS


def test_every_game_over_prompt_moves_the_game_state_to_game_over():
    for prompt in GAME_OVER_PROMPTS:
        assert TRANSITIONS[State.GAME][prompt].next_state is State.GAME_OVER


def test_not_updating_persona_means_permadeath():
    combat_death = (
        b"\x1b[0;30;41mYou feel your very existence severed from you...\r\n"
        b"You have been killed by the vampire.\x1b[1;37;40m\r\n"
        b"\x1b[0;31;40mNot updating persona.\x1b[1;37;40m\r"
    )
    assert is_permadeath(combat_death) is True


def test_tempdeath_is_not_permadeath():
    swearing_death = (
        b"In order to keep the game uncorrupted, you have been killed.\r\n"
        b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"
    )
    assert is_permadeath(swearing_death) is False


def test_spoken_not_updating_persona_is_not_a_permadeath():
    """The words alone are forgeable, so player speech must never read as a death."""
    spoken = b'Dumbo the novice says "\x1b[1;33;40mNot updating persona.\x1b[0;33;40m".\r\n'
    assert is_permadeath(spoken) is False
