"""Where the static room map puts each room on its grid."""

import pytest

from mudgym.notebooks.map_layout import NEAR_OFFSETS, compass_step, find_free_cell, layout_world_from_edges


def test_recorded_action_order_is_kept_before_aliases_resolve():
    edges = [("a", "n", "b"), ("a", "move north", "z")]
    placed = layout_world_from_edges(["a", "b", "z"], edges)
    assert placed["z"] == (0, -2)
    assert placed["b"] == (0, -4)
    assert layout_world_from_edges(["a", "b", "z"], reversed(edges)) == placed


def test_repeated_actions_keep_every_destination_sorted_by_room_id():
    placed = layout_world_from_edges(["a", "b", "c"], [("a", "east", "c"), ("a", "east", "b")])
    assert placed["b"] == (2, 0)
    assert placed["c"] == (4, 0)


def test_disconnected_rooms_reseed_in_node_order_and_unknown_actions_take_a_nearby_cell():
    edges = [("x", "move north", "y"), ("y", "enter portal", "z")]
    placed = layout_world_from_edges(["start", "x", "y", "z"], edges)
    assert placed == {"start": (0, 0), "x": (2, 6), "y": (2, 4), "z": (4, 4)}


@pytest.mark.parametrize(
    ("action", "step"),
    [
        ("n", (0, -2)),
        ("s", (0, 2)),
        ("e", (2, 0)),
        ("w", (-2, 0)),
        ("ne", (2, -2)),
        ("nw", (-2, -2)),
        ("se", (2, 2)),
        ("sw", (-2, 2)),
        ("u", (1, -1)),
        ("d", (-1, 1)),
        (" MOVE NORTHWEST ", (-2, -2)),
        ("enter portal", None),
    ],
)
def test_compass_steps_accept_aliases_and_move_prefixes(action, step):
    assert compass_step(action) == step
    placed = layout_world_from_edges(["a", "b"], [("a", action, "b")])
    assert placed["b"] == (step if step else (2, 0))


def test_collisions_stretch_along_the_compass_before_nearby_cells():
    occupied = {(2, 0), (4, 0), (8, 0)}
    assert find_free_cell(occupied, (0, 0), (2, 0)) == (16, 0)
    occupied.add((16, 0))
    assert find_free_cell(occupied, (0, 0), (2, 0)) == (0, 2)


def test_free_cell_search_sweeps_every_radius():
    occupied = {(delta_x, delta_y) for delta_x, delta_y in NEAR_OFFSETS}
    spot = find_free_cell(occupied, (0, 0), None)
    assert max(abs(spot[0]), abs(spot[1])) <= 2


def test_a_room_that_only_leaves_into_a_placed_room_sits_opposite_its_exit():
    placed = layout_world_from_edges(["a", "b", "c"], [("a", "north", "b"), ("c", "west", "b")])
    assert placed == {"a": (0, 0), "b": (0, -2), "c": (2, -2)}


def test_an_incoming_chain_is_positioned_before_reseeding_its_first_room():
    placed = layout_world_from_edges(
        ["a", "b", "c", "d"], [("a", "north", "b"), ("c", "west", "d"), ("d", "west", "b")]
    )
    assert placed == {"a": (0, 0), "b": (0, -2), "d": (2, -2), "c": (4, -2)}


def test_empty_and_single_room_layouts():
    assert layout_world_from_edges([], []) == {}
    assert layout_world_from_edges(["c"], []) == {"c": (0, 0)}
