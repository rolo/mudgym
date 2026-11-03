import numpy as np
import pytest

from mudgym.db.index import UNKNOWN
from mudgym.envs.fields.fescore import FEScoreField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


def test_extracts_status_without_points():
    field = FEScoreField()
    raw = b"58 78 41 61 32 52 7 68 3000000000 N N N N 53 F"
    obs = field.extract([raw])

    assert field.matches(raw)
    np.testing.assert_equal(
        obs,
        {
            "vitals": [58, 78, 41, 61, 32, 52, 7, 68],
            "flags": [0, 0, 0, 0],
            "reset_minutes": 53,
            "weather": "fair",
            "weather_index": 1,
        },
    )
    assert_valid_observation(field, obs)


def test_no_match_returns_zero_defaults_and_unknown_weather():
    line = b"58 58"
    field = FEScoreField()
    expected = {
        "vitals": [0] * 8,
        "flags": [0] * 4,
        "reset_minutes": 0,
        "weather": UNKNOWN,
        "weather_index": 0,
    }

    assert not field.matches(line)
    np.testing.assert_equal(field.empty(), expected)
    np.testing.assert_equal(field.extract([line]), expected)


def test_latest_wins_when_multiple_matches():
    raw = b"\r\n".join(
        [
            b"50 50 10 10 10 10 0 50 0100 N N N N 20 F",
            b"It is raining. ",
            b"60 60 12 12 12 12 0 60 0150 N N N N 25 C",
        ]
    )
    obs = FEScoreField().extract([raw])

    np.testing.assert_equal(
        obs,
        {
            "vitals": [60, 60, 12, 12, 12, 12, 0, 60],
            "flags": [0, 0, 0, 0],
            "reset_minutes": 25,
            "weather": "cloudy",
            "weather_index": 2,
        },
    )


def test_weather_mapping():
    field = FEScoreField()
    obs = field.extract([b"58 58 61 61 61 61 0 58 0200 N N N N 53 B"])

    assert obs["weather"] == "blizzard"
    assert obs["weather_index"] == 7
    assert_valid_observation(field, obs)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param("Y N N N", [1, 0, 0, 0], id="blind"),
        pytest.param("N Y N N", [0, 1, 0, 0], id="deaf"),
        pytest.param("N N Y N", [0, 0, 1, 0], id="crippled"),
        pytest.param("N N N Y", [0, 0, 0, 1], id="dumb"),
    ],
)
def test_flags_keep_their_positions(flags, expected):
    field = FEScoreField()
    obs = field.extract([f"58 58 61 61 61 61 0 58 0200 {flags} 53 S".encode()])

    np.testing.assert_equal(obs["flags"], expected)
    assert_valid_observation(field, obs)


def test_extracts_fes_from_real_captures(bytes_case):
    field = FEScoreField()
    obs = field.extract(bytes_case["chunks"])

    np.testing.assert_equal(obs, bytes_case["fes"])
    assert_valid_observation(field, obs)


def test_separator_controls_are_not_whitespace_around_a_status_line():
    # \x1f is not ASCII whitespace, so it remains part of the line and prevents a match
    assert not FEScoreField().matches(b"\x1f58 58 61 61 61 61 0 58 0200 N N N N 53 F\x1f\r\n")
