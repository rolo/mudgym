"""
Room IDs, room names, weather and database directions use one-based indices. Observation fields assign 0 explicitly when data is missing. Encoding an unrecognised supplied value raises ValueError.

Action indices and exit-mask positions are zero-based and follow DIRECTIONS order. For example, available_exits[i] is 1 when DIRECTIONS[i] is available and 0 otherwise.
"""

from collections.abc import Sequence

from mudgym.db.directions import DIRECTIONS
from mudgym.db.rooms import ROOM_IDS, ROOM_NAMES
from mudgym.db.weather import WEATHER

# Unknown sentinel value for empty, missing or unrecognised members
UNKNOWN = "unknown"


def member_to_index(collection: Sequence[str], member: str) -> int:
    """Return a one-based member index, raising ValueError if the member is absent."""
    try:
        return collection.index(member) + 1
    except ValueError:
        raise ValueError(f"{member!r} not in collection") from None


def index_to_member(collection, index: int, unknown: str = UNKNOWN) -> str:
    """Return the member corresponding to a one-based index, or the unknown placeholder if the index is 0."""
    # we subtract 1 from the index to get the index of the member in the collection
    if index == 0:
        return unknown
    return collection[index - 1]


def room_id_to_index(room_id: str) -> int:
    """Return a one-based room ID index, raising ValueError for an unrecognised ID."""
    return member_to_index(ROOM_IDS, room_id)


def index_to_room_id(index: int) -> str:
    """Convert a one-based index to a room ID (0 returns the missing placeholder)."""
    return index_to_member(ROOM_IDS, index)


def room_name_to_index(room_name: str) -> int:
    """Return a one-based room name index, raising ValueError for an unrecognised name."""
    return member_to_index(ROOM_NAMES, room_name)


def index_to_room_name(index: int) -> str:
    """Convert a one-based index to a room name (0 returns the missing placeholder)."""
    return index_to_member(ROOM_NAMES, index)


def weather_to_index(weather: str) -> int:
    """Return a one-based weather index, raising ValueError for an unrecognised name."""
    return member_to_index(WEATHER, weather)


def index_to_weather(index: int) -> str:
    """Convert a one-based index to a weather name (0 returns the missing placeholder)."""
    return index_to_member(WEATHER, index)


def direction_to_index(direction: str) -> int:
    """Return a one-based direction index, raising ValueError for a noncanonical name."""
    return member_to_index(DIRECTIONS, direction)


def index_to_direction(index: int) -> str:
    """Convert a one-based index to a direction name (0 returns the missing placeholder)."""
    return index_to_member(DIRECTIONS, index)


def direction_to_bit(direction: str) -> int:
    """Return the zero-based action index or exit-mask position for a canonical direction."""
    return DIRECTIONS.index(direction)


def bit_to_direction(bit: int) -> str:
    """
    Convert a bit index to a direction name.
    """
    return DIRECTIONS[bit]


# Count constants
DIRECTION_COUNT = len(DIRECTIONS)
ROOM_ID_COUNT = len(ROOM_IDS)
ROOM_NAME_COUNT = len(ROOM_NAMES)
WEATHER_COUNT = len(WEATHER)
