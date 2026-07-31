import re
from collections.abc import Sequence
from typing import Any

from gymnasium import spaces

from mudgym.connections.prompts import SGR_ONE_PLUS_STR
from mudgym.db.index import room_name_count, room_name_to_index
from mudgym.envs.specs import INDEX_DTYPE, ITEM_SPACE, ROOM_NAME_MAX_LENGTH, SINGLE_LINE_CHARSET
from mudgym.featurizers.ansi import strip_ansi
from mudgym.featurizers.strings import decode_text_bytes

from .field import ObservationField

ROOM_MARKER = "The place known as "
ROOM_MARKER_BYTES = ROOM_MARKER.encode("ascii")

TOKEN_RE = re.compile(
    rf"^(?P<sgr>{SGR_ONE_PLUS_STR})?"
    r"(?P<text>[^\x1b]+?)"
    rf"(?:{SGR_ONE_PLUS_STR})*$"
)
ROOM_LINE_RE = re.compile(
    rf'{ROOM_MARKER}(?:\[[^\]]+\]\s*)?"(?P<place>[^"]+)" contains '
    r"(?P<contents>[^\r\n]*?)\.(?:\r?\n|$)"
)
CARRYING_RE = re.compile(
    r"^(?P<carrier>.+?)\s+is carrying the following:\s*$",
    re.MULTILINE,
)
INVENTORY_RE = re.compile(
    r"^You are carrying the following:\s*$",
    re.MULTILINE,
)
THE_PREFIX_RE = re.compile(r"^the\s+", re.IGNORECASE)
CLASSIFIED_KEYS = ("features", "portables", "mobiles", "players")

FG_TO_CATEGORY: dict[int, str] = {
    32: "features",  # green - feature / hereobj
    31: "players",  # red - player
    33: "mobiles",  # yellow - mobile
    36: "portables",  # cyan - portable
    35: "access",  # magenta - doors, grates, etc.
}


def parse_token(chunk: str) -> tuple[str | None, str]:
    """Extract the SGR prefix and clean text from a token chunk."""
    chunk = chunk.strip()
    match = TOKEN_RE.match(chunk)
    if match is None:
        return None, chunk
    return match.group("sgr"), match.group("text").strip()


def sgr_foreground(sgr: str | None) -> int | None:
    """Return the foreground colour code from the last SGR sequence."""
    if not sgr:
        return None

    matches = re.findall(r"\x1b\[([0-9;]*)m", sgr)
    for match in reversed(matches):
        params = [int(param) for param in match.split(";") if param]
        foreground = next((param for param in params if 30 <= param <= 37), None)
        if foreground is not None:
            return foreground
    return None


def clean_name(text: str) -> str:
    """Strip leading 'the ' from an item name."""
    return THE_PREFIX_RE.sub("", text).strip()


def find_last_room_line(text: str) -> re.Match[str] | None:
    match = None
    for room_match in ROOM_LINE_RE.finditer(text):
        match = room_match
    return match


def empty_classified() -> dict[str, list[str]]:
    return {key: [] for key in CLASSIFIED_KEYS}


def parse_room_contents(contents_chunk: str) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    classified = empty_classified()
    contents_chunk = contents_chunk.strip()
    if not contents_chunk or contents_chunk.lower() == "nothing":
        return (), classified

    normalized = re.sub(r"\s+and\s+", ", ", contents_chunk)
    parts = [part.strip() for part in normalized.split(",") if part.strip()]

    here: list[str] = []
    for part in parts:
        sgr, text = parse_token(part)
        name = clean_name(text)
        if not name:
            continue

        here.append(name)
        category = FG_TO_CATEGORY.get(sgr_foreground(sgr))
        if category in classified:
            classified[category].append(name)

    return tuple(here), classified


def extract_indented_block(text: str, start: int) -> str:
    lines = text[start:].splitlines()
    block_lines: list[str] = []

    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent == 0 and not stripped.startswith("You can't"):
            break
        block_lines.append(stripped)

    return "\n".join(block_lines)


def parse_item_list(text: str) -> tuple[str, ...]:
    text = text.strip().rstrip(".")
    if not text or text.lower() == "nothing":
        return ()

    if " and " in text:
        main, last = text.rsplit(" and ", 1)
        parts = [part.strip().rstrip(".") for part in main.split(",")]
        parts.append(last.strip().rstrip("."))
    else:
        parts = [part.strip().rstrip(".") for part in text.split(",")]

    return tuple(clean_name(part) for part in parts if part)


def parse_block_items(block: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in block.splitlines():
        stripped = line.strip().rstrip(".")
        lowered = stripped.lower()
        if not stripped or "contains:" in lowered or "can't see" in lowered:
            continue
        items.extend(parse_item_list(stripped))
    return tuple(items)


def parse_carrying_and_inventory(clean_text: str, block_start: int) -> tuple[str, tuple[str, ...]]:
    block_text = clean_text[block_start:]

    carrying: list[str] = []
    for match in CARRYING_RE.finditer(block_text):
        carrier = clean_name(match.group("carrier").strip())
        block = extract_indented_block(block_text, match.start())
        items = parse_block_items(block)
        if carrier and items:
            carrying.append(f"{carrier}: [{', '.join(items)}]")

    inventory: tuple[str, ...] = ()
    inventory_match = INVENTORY_RE.search(block_text)
    if inventory_match:
        block = extract_indented_block(block_text, inventory_match.start())
        inventory = parse_block_items(block)

    return ", ".join(carrying), inventory


class SuperQuickLookField(ObservationField):
    """
    Parses room contents and inventory from the superquicklook command. Pure: reads the step bytes and
    returns its own keys only.
    """

    command = "sql"

    def full_space(self) -> dict[str, spaces.Space]:
        return {
            "room_name": spaces.Text(max_length=ROOM_NAME_MAX_LENGTH, min_length=0, charset=SINGLE_LINE_CHARSET),
            "room_name_index": spaces.Discrete(room_name_count),
            "here": ITEM_SPACE,
            "inventory": ITEM_SPACE,
            "features": ITEM_SPACE,
            "portables": ITEM_SPACE,
            "mobiles": ITEM_SPACE,
            "players": ITEM_SPACE,
        }

    def full_empty(self) -> dict[str, Any]:
        return {
            "room_name": "",
            "room_name_index": INDEX_DTYPE(0),
            "here": (),
            "inventory": (),
            "features": (),
            "portables": (),
            "mobiles": (),
            "players": (),
        }

    def matches(self, chunk: bytes) -> bool:
        return ROOM_MARKER_BYTES in chunk or b"It's too dark for you to see anything." in chunk

    def full_extract(self, chunks: Sequence[bytes]) -> dict[str, Any]:
        """Parse the latest superquicklook room view, or the empty default if none is present."""
        raw_bytes = b"".join(chunks)
        if ROOM_MARKER_BYTES not in raw_bytes:
            return self.full_empty()

        text = decode_text_bytes(raw_bytes)
        room_match = find_last_room_line(text)
        if room_match is None:
            return self.full_empty()

        _, room_name = parse_token(room_match.group("place"))
        room_name = room_name.lower()

        here, classified = parse_room_contents(room_match.group("contents"))

        clean_text = decode_text_bytes(strip_ansi(raw_bytes))
        block_start = clean_text.rfind(ROOM_MARKER)
        if block_start == -1:
            block_start = 0

        _, inventory = parse_carrying_and_inventory(clean_text, block_start)

        return {
            "room_name": room_name,
            "room_name_index": INDEX_DTYPE(room_name_to_index(room_name) if room_name else 0),
            "here": here,
            "inventory": inventory,
            "features": tuple(classified["features"]),
            "portables": tuple(classified["portables"]),
            "mobiles": tuple(classified["mobiles"]),
            "players": tuple(classified["players"]),
        }
