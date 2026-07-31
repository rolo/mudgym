from mudgym.db.directions import DIRECTION_ALIASES, DIRECTIONS
from mudgym.db.rooms import ROOM_IDS, ROOM_NAMES
from mudgym.db.weather import WEATHER

"""
Indexing utility functions for collections of members.

Collections use 1-based indexing with 0 reserved for unknown/missing values.
This applies to room IDs, room names, directions, and weather.

Binary or multi-hot feature vectors (e.g. available_exits) instead use 0-based indexing where index
i corresponds directly to collection[i]. For example, the available_exits vector is a bit vector
where the bit at index i is 1 if the direction is available, 0 otherwise.

I've added helpers to try and make this clear. See direction_to_bit and bit_to_direction.
"""


def indexed_discrete_size(num_members: int) -> int:
    """
    Return the size needed for a Discrete space with 1-based indexing.

    For a collection with num_members items, we need num_members + 1 slots:
    - Index 0: unknown/missing sentinel
    - Indices 1..num_members: actual members

    Args:
        num_members: Number of actual members in the collection

    Returns:
        Size for spaces.Discrete (num_members + 1)
    """
    return num_members + 1


def member_to_index(collection, member: str) -> int:
    """
    Convert a member name to an index.

    Args:
        collection: The collection to search in
        member: The member to find

    Returns:
        The index of the member in the collection (1-based), or 0 if not found
    """
    try:
        idx = collection.index(member)
        # we add 1 to the returned index to allow for an unknown/missing value at index 0
        return idx + 1
    except ValueError:
        # Return 0 for unknown/missing values
        return 0


def index_to_member(collection, index: int, unknown: str | None = "unknown") -> str:
    # we subtract 1 from the index to get the index of the member in the collection
    return collection[index - 1] if index > 0 else unknown


def room_id_to_index(room_id: str) -> int:
    """Convert room ID to 1-based index (0 = unknown)."""
    return member_to_index(ROOM_IDS, room_id)


def index_to_room_id(index: int) -> str:
    """Convert 1-based index to room ID (0 = unknown)."""
    return index_to_member(ROOM_IDS, index)


def room_name_to_index(room_name: str) -> int:
    """Convert room name to 1-based index (0 = unknown)."""
    return member_to_index(ROOM_NAMES, room_name)


def index_to_room_name(index: int) -> str:
    """Convert 1-based index to room name (0 = unknown)."""
    return index_to_member(ROOM_NAMES, index)


def weather_to_index(weather: str) -> int:
    """Convert weather name to 1-based index (0 = unknown)."""
    return member_to_index(WEATHER, weather)


def index_to_weather(index: int) -> str:
    """Convert 1-based index to weather name (0 = unknown)."""
    return index_to_member(WEATHER, index)


def direction_to_index(direction: str) -> int:
    """Convert a direction name or alias to a 1-based index (0 = unknown)."""
    return member_to_index(DIRECTIONS, DIRECTION_ALIASES.get(direction, direction))


def index_to_direction(index: int) -> str:
    """Convert 1-based index to direction name (0 = unknown)."""
    return index_to_member(DIRECTIONS, index)


def direction_to_bit(direction: str) -> int:
    """
    Convert a direction name or alias to a bit index.

    Direction vectors use 0-based positions while the other database indices are
    1-based, so this helper makes the distinction explicit.
    """
    return DIRECTIONS.index(DIRECTION_ALIASES.get(direction, direction))


def bit_to_direction(bit: int) -> str:
    """
    Convert a bit index to a direction name.
    """
    return DIRECTIONS[bit]


# Count constants
direction_count = len(DIRECTIONS)
room_id_count = len(ROOM_IDS)
room_name_count = len(ROOM_NAMES)
weather_count = len(WEATHER)
