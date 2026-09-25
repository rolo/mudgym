import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from mudgym.db.directions import DIRECTIONS
from mudgym.db.index import DIRECTION_COUNT
from mudgym.envs.fields.fexits import FEXitsField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


@pytest.mark.parametrize(
    ("raw", "expected_names", "expected_mask"),
    [
        pytest.param(b"north", ("north",), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], id="single"),
        pytest.param(
            b"up down out swampward southwest northeast northwest west",
            ("west", "northeast", "southwest", "northwest", "up", "down", "out", "swampward"),
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            id="mixed-subset",
        ),
        pytest.param(
            b"up in over down out swampward southwest south southeast northeast northwest west east north",
            tuple(DIRECTIONS),
            [1] * DIRECTION_COUNT,
            id="all-in-noncanonical-order",
        ),
        pytest.param(
            b"swampward over",
            ("over", "swampward"),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            id="special-directions",
        ),
        pytest.param(
            b"northwest north",
            ("north", "northwest"),
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            id="prefix-pair",
        ),
        pytest.param(
            b"up out swampward south west east north\r\n",
            ("north", "east", "south", "west", "up", "out", "swampward"),
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            id="real-exits-line",
        ),
    ],
)
def test_extracts_exits_in_game_direction_order(raw, expected_names, expected_mask):
    field = FEXitsField()
    obs = field.extract([raw])

    assert field.matches(raw)
    np.testing.assert_equal(obs, {"available_exit_names": expected_names, "available_exits": expected_mask})
    assert_valid_observation(field, obs)


def test_available_exits_is_a_gymnasium_action_mask():
    obs = FEXitsField().extract([b"north"])
    action_space = gym_spaces.Discrete(DIRECTION_COUNT)

    assert obs["available_exits"].dtype == np.dtype(np.int8)
    assert action_space.sample(mask=obs["available_exits"]) == DIRECTIONS.index("north")


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(b"58 58 61 61 61 61 0 58 0200 N N N N 53 F", id="another-command"),
        pytest.param(b"move north,fex,fei", id="command-echo"),
        pytest.param(b"north and south", id="prose-containing-directions"),
        pytest.param(b"oo\r\n", id="darkness-marker"),
        pytest.param(b"", id="empty"),
        pytest.param(b"\r\n", id="blank-dark-room"),
    ],
)
def test_missing_exits_leave_every_direction_available(line):
    field = FEXitsField()
    expected = {"available_exit_names": tuple(DIRECTIONS), "available_exits": [1] * DIRECTION_COUNT}

    np.testing.assert_equal(field.empty(), expected)
    np.testing.assert_equal(field.extract([line]), expected)


def test_latest_wins_when_multiple_matches():
    raw = b"\r\n".join([b"north south", b"It is raining. ", b"east west"])
    obs = FEXitsField().extract([raw])

    np.testing.assert_equal(
        obs,
        {
            "available_exit_names": ("east", "west"),
            "available_exits": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
    )


def test_matches_blank_dark_room_response():
    # Dark rooms return a blank line that the env must recognise as valid FEX output.
    assert FEXitsField().matches(b"\r\n")


def test_does_not_match_other_command_output():
    assert not FEXitsField().matches(b"58 58 61 61 61 61 0 58 0200 N N N N 53 F\r\n")


def test_extracts_fex_from_real_captures(bytes_case):
    field = FEXitsField()
    obs = field.extract(bytes_case["chunks"])
    names = bytes_case["fex"]["names"]
    expected = {
        "available_exit_names": tuple(direction for direction in DIRECTIONS if direction in names),
        "available_exits": [int(direction in names) for direction in DIRECTIONS],
    }

    np.testing.assert_equal(obs, expected)
    assert_valid_observation(field, obs)


def test_separator_controls_are_not_whitespace_around_an_exits_line():
    assert not FEXitsField().matches(b"\x1fnorth south\x1f\r\n")
