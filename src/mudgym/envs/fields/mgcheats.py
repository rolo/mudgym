import re
from collections.abc import Sequence
from typing import Any

from gymnasium import spaces

from mudgym.db.index import (
    indexed_discrete_size,
    room_id_count,
    room_id_to_index,
    room_name_count,
    room_name_to_index,
)
from mudgym.envs.specs import (
    BIT_DTYPE,
    IDENTIFIER_CHARSET,
    IDENTIFIER_SPACE,
    INDEX_DTYPE,
    ROOM_ID_MAX_LENGTH,
    ROOM_NAME_MAX_LENGTH,
    SINGLE_LINE_CHARSET,
)
from mudgym.featurizers.strings import decode_text_bytes

from .field import ObservationField

MGCHEATS_BLOCK = re.compile(rb"\[mgcheats\](.*?)\[/mgcheats\]", re.DOTALL)


class MGCheatsField(ObservationField):
    """
    Reads the mgcheats block.

    Sample game response:
    `[mgcheats]room_id=mtrack1; room_name=beaten track near cliff; fighting=0; dark=0; glowing=0; asleep=0; gifted=0; here=[rain, cliff, road]; inventory=[][/mgcheats]`
    """

    command = "mgcheats"

    # the closing tag closes the read window when mgcheats terminates the auto-command batch
    end_of_turn_marker = re.compile(rb"\[/mgcheats\]\r?\n")

    BIT_KEYS = ("fighting", "dark", "glowing", "asleep", "gifted")

    def full_space(self) -> dict[str, spaces.Space]:
        return {
            "room_id": spaces.Text(max_length=ROOM_ID_MAX_LENGTH, min_length=0, charset=IDENTIFIER_CHARSET),
            "room_id_index": spaces.Discrete(indexed_discrete_size(room_id_count)),
            "room_name": spaces.Text(max_length=ROOM_NAME_MAX_LENGTH, min_length=0, charset=SINGLE_LINE_CHARSET),
            "room_name_index": spaces.Discrete(indexed_discrete_size(room_name_count)),
            "fighting": spaces.Discrete(2, dtype=BIT_DTYPE),
            "dark": spaces.Discrete(2, dtype=BIT_DTYPE),
            "glowing": spaces.Discrete(2, dtype=BIT_DTYPE),
            "asleep": spaces.Discrete(2, dtype=BIT_DTYPE),
            "gifted": spaces.Discrete(2, dtype=BIT_DTYPE),
            "here": IDENTIFIER_SPACE,
        }

    def full_empty(self) -> dict[str, Any]:
        return {
            "room_id": "",
            "room_id_index": INDEX_DTYPE(0),
            "room_name": "",
            "room_name_index": INDEX_DTYPE(0),
            **{k: BIT_DTYPE(0) for k in self.BIT_KEYS},
            "here": (),
        }

    def matches(self, chunk: bytes) -> bool:
        return MGCHEATS_BLOCK.search(chunk) is not None

    def parse(self, payload_bytes: bytes) -> dict[str, Any]:
        """Parse a single mgcheats payload (semicolon-separated ``key=value`` pairs) into a dict of values."""
        parsed_values: dict[str, Any] = {}
        text = decode_text_bytes(payload_bytes).strip()
        for part in (segment.strip() for segment in text.split(";")):
            if not part or "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            lowered = raw_value.lower()

            if raw_value.startswith("[") and raw_value.endswith("]"):
                inner = raw_value[1:-1].strip()
                if inner:
                    entries = [item.strip() for item in inner.split(",") if item.strip()]
                    parsed_values[key] = [entry.lower() for entry in entries]
                else:
                    parsed_values[key] = []
            else:
                parsed_values[key] = int(lowered) if lowered in ("0", "1") else lowered
        return parsed_values

    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """Parse the latest ``[mgcheats]`` block, or the empty default if none is present."""
        payloads = MGCHEATS_BLOCK.findall(b"".join(chunks))
        if not payloads:
            return self.full_empty()

        parsed = self.parse(payloads[-1])

        room_id = str(parsed.get("room_id", ""))
        room_name = str(parsed.get("room_name", ""))

        return {
            "room_id": room_id,
            "room_id_index": INDEX_DTYPE(room_id_to_index(room_id) if room_id else 0),
            "room_name": room_name,
            "room_name_index": INDEX_DTYPE(room_name_to_index(room_name) if room_name else 0),
            **{key: BIT_DTYPE(int(parsed.get(key, 0))) for key in self.BIT_KEYS},
            "here": tuple(str(item) for item in (parsed.get("here") or [])),
        }
