import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from mudgym.db.index import UNKNOWN, WEATHER_COUNT, weather_to_index
from mudgym.db.weather import WEATHER_CODE_TO_NAME
from mudgym.envs.specs import BIT_DTYPE, INDEX_DTYPE, INT_DTYPE, SINGLE_LINE_CHARSET

from .field import ObservationField

MAX_RESET_MINUTES = 105


class FEScoreField(ObservationField):
    """
    Parsed FES line values:
      - vitals (8-dim) - stamina, max_stamina, effective_strength, strength, effective_dexterity, dexterity, magic, max_magic
      - flags (4-dim) - blind, deaf, crippled, dumb
      - reset_minutes (scalar)
      - weather (text) - fair, raining, snowing, etc.
      - weather_index (scalar) - index of the weather
    """

    command = "fes"

    REGEX = re.compile(
        r"""^\s*
        (?P<stamina>\d+)\s+
        (?P<max_stamina>\d+)\s+
        (?P<effective_strength>\d+)\s+
        (?P<strength>\d+)\s+
        (?P<effective_dexterity>\d+)\s+
        (?P<dexterity>\d+)\s+
        (?P<magic>\d+)\s+
        (?P<max_magic>\d+)\s+
        \d{2,}\s+
        (?P<is_blind>[YN])\s+
        (?P<is_deaf>[YN])\s+
        (?P<is_crippled>[YN])\s+
        (?P<is_dumb>[YN])\s+
        (?P<reset_minutes>\d+)\s+
        (?P<weather>[SBRTCOF])\s*
        $""",
        re.VERBOSE | re.ASCII,
    )

    def full_space(self) -> dict[str, spaces.Space]:
        return {
            "vitals": spaces.Box(low=0, high=200, shape=(8,), dtype=INT_DTYPE),
            "flags": spaces.MultiBinary(4),
            "reset_minutes": spaces.Box(low=0, high=MAX_RESET_MINUTES, shape=(), dtype=INT_DTYPE),
            "weather": spaces.Text(max_length=16, min_length=0, charset=SINGLE_LINE_CHARSET),
            "weather_index": spaces.Discrete(WEATHER_COUNT + 1),
        }

    def full_empty(self) -> dict[str, Any]:
        return {
            "vitals": np.zeros(8, dtype=INT_DTYPE),
            "flags": np.zeros(4, dtype=BIT_DTYPE),
            "reset_minutes": INT_DTYPE(0),
            "weather": UNKNOWN,
            "weather_index": INDEX_DTYPE(0),
        }

    def matches(self, chunk: bytes) -> bool:
        return any(self.REGEX.match(line) for line in self.lines(chunk))

    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """Parse the latest FES status line from the turn chunks, or the empty default if none is present."""
        match = self.find_last_line(self.REGEX, chunks)
        if match is None:
            return self.full_empty()

        weather_name = WEATHER_CODE_TO_NAME[match.group("weather")]

        vitals = np.array(
            [
                int(match.group("stamina")),
                int(match.group("max_stamina")),
                int(match.group("effective_strength")),
                int(match.group("strength")),
                int(match.group("effective_dexterity")),
                int(match.group("dexterity")),
                int(match.group("magic")),
                int(match.group("max_magic")),
            ],
            dtype=INT_DTYPE,
        )

        flags = np.array(
            [
                int(match.group("is_blind") == "Y"),
                int(match.group("is_deaf") == "Y"),
                int(match.group("is_crippled") == "Y"),
                int(match.group("is_dumb") == "Y"),
            ],
            dtype=BIT_DTYPE,
        )

        return {
            "vitals": vitals,
            "flags": flags,
            "reset_minutes": INT_DTYPE(int(match.group("reset_minutes"))),
            "weather": weather_name,
            "weather_index": INDEX_DTYPE(weather_to_index(weather_name)),
        }
