import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from mudgym.db.directions import DIRECTIONS
from mudgym.db.index import direction_count, direction_to_bit
from mudgym.envs.fields.fexits import FEXitsField
from mudgym.envs.specs import BIT_DTYPE


def expected_vector(names: set[str]) -> np.ndarray:
    v = np.zeros(len(DIRECTIONS), dtype=BIT_DTYPE)
    for i, d in enumerate(DIRECTIONS):
        if d in names:
            v[i] = 1
    return v


def test_matches_valid_line():
    obs = FEXitsField().extract([b"up down out swampward southwest northeast northwest west"])

    expected = {"up", "down", "out", "swampward", "southwest", "northeast", "northwest", "west"}
    assert obs["available_exit_names"] == tuple(direction for direction in DIRECTIONS if direction in expected)

    for direction in expected:
        assert obs["available_exits"][direction_to_bit(direction)] == 1

    for direction in ("east", "north", "south"):
        assert obs["available_exits"][direction_to_bit(direction)] == 0


def test_single_exit():
    obs = FEXitsField().extract([b"north"])

    assert obs["available_exit_names"] == ("north",)
    assert obs["available_exits"][direction_to_bit("north")] == 1
    assert np.sum(obs["available_exits"]) == 1


def test_available_exits_is_a_gymnasium_action_mask():
    obs = FEXitsField().extract([b"north"])
    action_space = gym_spaces.Discrete(direction_count)

    assert obs["available_exits"].dtype == np.dtype(np.int8)
    assert action_space.sample(mask=obs["available_exits"]) == direction_to_bit("north")


def test_all_exits():
    obs = FEXitsField().extract([" ".join(DIRECTIONS).encode()])

    assert obs["available_exit_names"] == tuple(DIRECTIONS)
    assert np.all(obs["available_exits"] == 1)


def test_exit_names_use_game_direction_order_regardless_of_fex_order():
    raw_exit_names = tuple(reversed(DIRECTIONS))
    obs = FEXitsField().extract([" ".join(raw_exit_names).encode()])

    assert obs["available_exit_names"] == tuple(DIRECTIONS)


def test_fex_direction_names_project_into_game_exit_order():
    obs = FEXitsField().extract(
        [b"up in over down out swampward southwest south southeast northeast northwest west east north"]
    )

    assert obs["available_exit_names"] == tuple(DIRECTIONS)
    assert obs["available_exits"][direction_to_bit("over")] == 1
    assert obs["available_exits"][direction_to_bit("swampward")] == 1


@pytest.mark.parametrize(
    "line",
    [
        b"no exits here",
        b"not an exit line",
        b"58 58 61 61 61 61 0 58 0200 N N N N 53 F",
        b"move north,fex,fei",
        b"north and south",
        b"oo\r\n",
        b"",
        b"gibberish xyz 123",
    ],
)
def test_invalid_lines_return_all_exits(line):
    """When no valid FEX line is found, default to all exits available."""
    obs = FEXitsField().extract([line])

    assert np.all(obs["available_exits"] == 1)
    assert set(obs["available_exit_names"]) == set(DIRECTIONS)


def test_latest_wins_when_multiple_matches():
    raw = b"\r\n".join([b"north south", b"It is raining. ", b"east west"])
    obs = FEXitsField().extract([raw])

    assert obs["available_exit_names"] == ("east", "west")


def test_prefix_direction_regression():
    """Ensure prefix directions (e.g. north/northwest) are matched correctly."""
    pair = None
    for a in DIRECTIONS:
        for b in DIRECTIONS:
            if a != b and b.startswith(a):
                pair = (a, b)
                break
        if pair:
            break

    if pair is None:
        pytest.skip("No prefix directions in DIRECTIONS set")

    a, b = pair
    obs = FEXitsField().extract([f"{b} {a}".encode()])
    assert set(obs["available_exit_names"]) == {a, b}


def test_empty_returns_valid_defaults():
    defaults = FEXitsField().empty()

    assert defaults["available_exits"].dtype == BIT_DTYPE
    assert len(defaults["available_exits"]) == direction_count
    assert np.all(defaults["available_exits"] == 1)
    assert defaults["available_exit_names"] == tuple(DIRECTIONS)


def test_space_types():
    spaces = FEXitsField().space()

    assert isinstance(spaces["available_exits"], gym_spaces.MultiBinary)
    assert spaces["available_exits"].dtype == np.dtype(np.int8)
    assert isinstance(spaces["available_exit_names"], gym_spaces.Sequence)


def test_matches_real_exits_line():
    assert FEXitsField().matches(b"up out swampward south west east north\r\n")


def test_matches_blank_dark_room_response():
    # in the dark fex returns a blank exits line; matches() must still recognise it as valid fex output
    # so the env's positional-routing cross-check holds.
    assert FEXitsField().matches(b"\r\n")


def test_does_not_match_other_command_output():
    # the fes status line is another command's output, not a valid fex response
    assert not FEXitsField().matches(b"58 58 61 61 61 61 0 58 0200 N N N N 53 F\r\n")


def test_extracts_fex_from_real_captures(bytes_case):
    obs = FEXitsField().extract(bytes_case["chunks"])

    expected_names = tuple(direction for direction in DIRECTIONS if direction in bytes_case["fex"]["names"])
    assert obs["available_exit_names"] == expected_names
    assert np.array_equal(obs["available_exits"], expected_vector(bytes_case["fex"]["names"]))
