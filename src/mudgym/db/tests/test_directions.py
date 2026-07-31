from mudgym.db.directions import DIRECTION_ALIASES, DIRECTIONS
from mudgym.db.index import bit_to_direction, direction_to_bit, direction_to_index, index_to_direction
from mudgym.db.rooms import ROOM_IDS, ROOM_NAME_BY_ID, ROOM_NAMES


def test_directions_follow_game_exit_order():
    assert DIRECTIONS == [
        "north",
        "east",
        "south",
        "west",
        "northeast",
        "southeast",
        "southwest",
        "northwest",
        "up",
        "down",
        "in",
        "out",
        "over",
        "swampward",
    ]


def test_room_lookups_are_sorted_and_consistent():
    assert ROOM_IDS == sorted(ROOM_NAME_BY_ID)
    assert ROOM_NAMES == sorted({room_name.lower() for room_name in ROOM_NAME_BY_ID.values()})
    assert ROOM_NAME_BY_ID["mtrack2"] == "beaten track"


def test_direction_indices_share_game_direction_order():
    for bit, direction in enumerate(DIRECTIONS):
        assert direction_to_bit(direction) == bit
        assert bit_to_direction(bit) == direction
        assert direction_to_index(direction) == bit + 1
        assert index_to_direction(bit + 1) == direction


def test_direction_indices_accept_aliases_and_return_direction_names():
    for alias, direction in DIRECTION_ALIASES.items():
        assert direction_to_bit(alias) == direction_to_bit(direction)
        assert direction_to_index(alias) == direction_to_index(direction)
        assert bit_to_direction(direction_to_bit(alias)) == direction
        assert index_to_direction(direction_to_index(alias)) == direction
