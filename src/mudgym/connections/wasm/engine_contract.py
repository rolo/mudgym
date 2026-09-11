"""Shared engine ABI limits, validation and statuses."""

from enum import IntEnum

# Engine ABI limits, shared by topology validation and direct runtime calls.
MAX_WORLD_SESSIONS = 50
MAX_SAFE_INTEGER = 2**53 - 1


def validate_seed(seed: int, *, label: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SAFE_INTEGER:
        raise ValueError(f"{label} must fit the WebAssembly safe integer range, got {seed!r}")
    return seed


def validate_session_count(count: int) -> None:
    if not 1 <= count <= MAX_WORLD_SESSIONS:
        raise ValueError(f"sessions per world must be between 1 and {MAX_WORLD_SESSIONS}, got {count}")


class SessionResultCode(IntEnum):
    """Negative results from session step, receive and removal calls."""

    OUTPUT_OVERFLOW = -2
    PLAYER_GONE = -3
    GAME_HALTED = -4
    GAME_INDETERMINATE = -5
    INPUT_REQUIRED = -6
    SESSION_GONE = -7
    WORLD_GONE = -8


class WorldStatus(IntEnum):
    """The shared world's state (mmud_shared_world_status_t)."""

    NEVER_STARTED = 0
    ACTIVE = 1
    HALTED = 2
    INDETERMINATE = 3
    SHUT_DOWN = 4


class CommandTicketStatus(IntEnum):
    """Terminal status of one submitted command (mmud_command_ticket_status_t)."""

    COMPLETED = 0
    HALTED = 1
    INPUT_REQUIRED = 2
    PLAYER_DEPARTED = 3
    FAILED = 4
