import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from mudgym.db.directions import DIRECTION_INDEX_BY_NAME, DIRECTIONS
from mudgym.db.index import direction_count
from mudgym.envs.specs import BIT_DTYPE, IDENTIFIER_CHARSET

from .field import ObservationField

MAX_DIRECTION_LENGTH = max(len(direction) for direction in DIRECTIONS)


def all_exits() -> dict[str, Any]:
    """All directions available - the default when no exits line is recognised."""
    return {
        "available_exits": np.ones(direction_count, dtype=BIT_DTYPE),
        "available_exit_names": tuple(DIRECTIONS),
    }


class FEXitsField(ObservationField):
    """
    FEX exit data field.

    Provides known available exits as:
      - available_exits: MultiBinary vector over all directions.
      - available_exit_names: Tuple of direction names.
    """

    command = "fex"

    # Each direction must be followed by whitespace or end-of-line.
    DIRECTION_GROUP = r"(?:" + "|".join(re.escape(direction) for direction in DIRECTIONS) + r")(?=\s|$)"
    REGEX = re.compile(
        rf"^\s*(?P<exits>{DIRECTION_GROUP}(?:\s+{DIRECTION_GROUP})*)\s*$",
        flags=re.ASCII,
    )

    def full_space(self) -> dict[str, spaces.Space]:
        return {
            "available_exits": spaces.MultiBinary(direction_count),
            "available_exit_names": spaces.Sequence(
                spaces.Text(
                    max_length=MAX_DIRECTION_LENGTH,
                    min_length=0,
                    charset=IDENTIFIER_CHARSET,
                ),
                stack=False,
            ),
        }

    def full_empty(self) -> dict[str, Any]:
        # we return all exits available for the empty/unknown case so we don't get stuck, this is practical, even if not pure
        return all_exits()

    def matches(self, chunk: bytes) -> bool:
        lines = [line.strip() for line in self.decode(chunk).splitlines()]
        # a dark room returns a blank exits response, which is a valid, if uninformative
        if not any(lines):
            return True
        return any(self.REGEX.match(line) for line in lines)

    def exits_to_vector(self, exit_names: Sequence[str]) -> np.ndarray:
        vector = np.zeros(direction_count, dtype=BIT_DTYPE)
        for d in exit_names:
            try:
                vector[DIRECTION_INDEX_BY_NAME[d]] = 1
            except KeyError:
                raise ValueError(f"Unknown direction in fex output: {d!r}") from None
        return vector

    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """Parse the latest FEX exits line.

        When no exits line is recognised (e.g. a dark room returns a blank exits response), default to all
        exits available so the agent can still attempt any direction.
        """
        match = self.find_last_line(self.REGEX, chunks)
        if match is None:
            return all_exits()

        available_exits = self.exits_to_vector(match.group("exits").split())
        return {
            "available_exits": available_exits,
            "available_exit_names": tuple(
                direction for index, direction in enumerate(DIRECTIONS) if available_exits[index]
            ),
        }
