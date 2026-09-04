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
    INDEX_DTYPE,
    ITEM_SPACE,
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

    # the closing tag closes the read window when mgcheats is the final observation command
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
            "here": ITEM_SPACE,
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

    def parse(self, payload_bytes: bytes) -> dict[str, str]:
        """Read the wire pairs without accepting duplicate keys."""
        values: dict[str, str] = {}
        for pair in decode_text_bytes(payload_bytes).split("; "):
            key, value = pair.split("=", 1)
            if key in values:
                raise ValueError(f"mgcheats block contains duplicate key {key!r}")
            values[key] = value
        return values

    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """Parse the latest ``[mgcheats]`` block, or the empty default if none is present."""
        payloads = MGCHEATS_BLOCK.findall(b"".join(chunks))
        if not payloads:
            return self.full_empty()

        parsed = self.parse(payloads[-1])

        room_id = parsed["room_id"].lower()
        room_name = parsed["room_name"].lower()
        bits = {}
        for key in self.BIT_KEYS:
            value = parsed[key]
            if value not in ("0", "1"):
                raise ValueError(f"mgcheats {key}={value!r} is not a 0 or 1 flag")
            bits[key] = BIT_DTYPE(value)
        here = parsed["here"]
        if not here.startswith("[") or not here.endswith("]"):
            raise ValueError(f"mgcheats here={here!r} is not a bracketed list")
        inner = here[1:-1].lower()

        return {
            "room_id": room_id,
            "room_id_index": INDEX_DTYPE(room_id_to_index(room_id) if room_id else 0),
            "room_name": room_name,
            "room_name_index": INDEX_DTYPE(room_name_to_index(room_name) if room_name else 0),
            **bits,
            "here": tuple(inner.split(", ")) if inner else (),
        }
