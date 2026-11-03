from mudgym.db.directions import DIRECTIONS
from mudgym.db.index import bit_to_direction, direction_to_bit, direction_to_index, index_to_direction


def test_direction_indices_share_game_direction_order():
    for bit, direction in enumerate(DIRECTIONS):
        assert direction_to_bit(direction) == bit
        assert bit_to_direction(bit) == direction
        assert direction_to_index(direction) == bit + 1
        assert index_to_direction(bit + 1) == direction
