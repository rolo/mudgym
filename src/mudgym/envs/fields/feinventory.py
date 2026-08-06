from collections.abc import Sequence
from typing import Any

from gymnasium import spaces

from mudgym.connections.prompts import FINAL_COMMAND_MARKER, INVENTORY_DIVIDER
from mudgym.envs.specs import IDENTIFIER_SPACE
from mudgym.featurizers.ansi import strip_ansi
from mudgym.featurizers.strings import decode_text_bytes

from .field import ObservationField

# A dark room shows "oo" in place of the room's visible portables (you can't see the floor), and a blinded
# persona shows "--". The divider and the player's own inventory below it still show normally, so we just
# drop these markers from the portables.
DARK_PORTABLES = "oo"
BLIND_PORTABLES = "--"


def parse_inventory_lines(block: bytes) -> tuple[str, ...]:
    """Decode a divider-delimited inventory block into a tuple of cleaned item names."""
    cleaned = strip_ansi(block)
    lines: list[str] = []
    for raw in cleaned.splitlines():
        raw = raw.strip()
        if raw:
            lines.append(decode_text_bytes(raw))
    return tuple(lines)


class FEInventoryField(ObservationField):
    """
    Parses the ``fei`` command output, split by the inventory divider into portables (lying around) and
    the player's own inventory.
    """

    command = "fei"

    # the ======== divider is the stock end-of-turn marker the transport falls back to
    end_of_turn_marker = FINAL_COMMAND_MARKER

    def full_space(self) -> dict[str, spaces.Space]:
        return {
            # the real fei grammar emits identifiers only ("brand39", "cloth-of-gold", "key50" in
            # the live captures), never descriptive phrases
            "portables": IDENTIFIER_SPACE,
            "inventory": IDENTIFIER_SPACE,
        }

    def full_empty(self) -> dict[str, Any]:
        return {
            "portables": (),
            "inventory": (),
        }

    def matches(self, chunk: bytes) -> bool:
        return INVENTORY_DIVIDER in chunk

    def full_extract(self, chunks: Sequence[bytes], **context: Any) -> dict[str, Any]:
        """Find the fei response chunk and split it on the divider into portables / inventory."""
        chunk = next(
            (c for c in reversed(chunks) if INVENTORY_DIVIDER in c),
            None,
        )
        if chunk is None:
            return self.full_empty()

        before, _, after = chunk.partition(INVENTORY_DIVIDER)
        portables = tuple(
            item for item in parse_inventory_lines(before) if item not in (DARK_PORTABLES, BLIND_PORTABLES)
        )
        return {
            "portables": portables,
            "inventory": parse_inventory_lines(after),
        }
