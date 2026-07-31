"""Public action vocabulary helpers."""

from mudgym.db.directions import DIRECTION_INDEX_BY_NAME, DIRECTIONS


def direction_index(direction: str) -> int:
    """Return the zero-based action and exit-mask index for a direction."""
    return DIRECTION_INDEX_BY_NAME[direction]


__all__ = ["DIRECTIONS", "direction_index"]
