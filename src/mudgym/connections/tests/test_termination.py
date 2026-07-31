from mudgym.connections.prompts import Prompt
from mudgym.connections.termination import END_OF_STEP_PATTERNS, NO_LONGER_IN_GAME_PROMPTS

# The game broadcasts status lines into live sessions mid-step
ASYNC_BROADCAST_PROMPTS = [
    Prompt.DATABASE_FINISHED_INITIALIZING,
    Prompt.DATABASE_NOT_INITIALIZED,
    Prompt.RESET_IN_PROGRESS,
    Prompt.BOOT_COMPLETE,
    Prompt.MAIL_UNAVAILABLE,
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
