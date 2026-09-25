import numpy as np
import pytest

from mudgym.db.index import UNKNOWN
from mudgym.envs.fields.fescore import FEScoreField
from mudgym.envs.specs import INT_DTYPE


def test_matches_valid_line():
    obs = FEScoreField().extract([b"58 58 61 61 61 61 0 58 0200 N N N N 53 F"])

    assert "points" not in obs
    np.testing.assert_array_equal(obs["vitals"], [58, 58, 61, 61, 61, 61, 0, 58])
    np.testing.assert_array_equal(obs["flags"], [0, 0, 0, 0])
    assert obs["reset_minutes"] == 53
    assert obs["weather"] == "fair"
    assert obs["weather_index"] == 1


def test_handles_no_match_returns_empty_defaults():
    obs = FEScoreField().extract([b"no fes here"])
    assert obs["weather"] == UNKNOWN
    assert obs["weather_index"] == 0
    np.testing.assert_array_equal(obs["vitals"], np.zeros(8, dtype=INT_DTYPE))


@pytest.mark.parametrize(
    "line",
    [b"not a fes line", b"58 58", b"move north,fex,fei", b""],
)
def test_rejects_invalid_lines(line):
    obs = FEScoreField().extract([line])
    assert obs["reset_minutes"] == 0
    np.testing.assert_array_equal(obs["vitals"], np.zeros(8, dtype=INT_DTYPE))


def test_latest_wins_when_multiple_matches():
    raw = b"\r\n".join(
        [
            b"50 50 10 10 10 10 0 50 0100 N N N N 20 F",
            b"It is raining. ",
            b"60 60 12 12 12 12 0 60 0150 N N N N 25 C",
        ]
    )
    obs = FEScoreField().extract([raw])

    assert obs["reset_minutes"] == 25
    assert obs["weather"] == "cloudy"


@pytest.mark.parametrize(
    ("code", "expected_idx", "expected_name"),
    [
        ("F", 1, "fair"),
        ("C", 2, "cloudy"),
        ("O", 3, "overcast"),
        ("R", 4, "raining"),
        ("T", 5, "stormy"),
        ("S", 6, "sunny"),
        ("B", 7, "blizzard"),
    ],
)
def test_weather_mapping(code, expected_idx, expected_name):
    obs = FEScoreField().extract([f"58 58 61 61 61 61 0 58 0200 N N N N 53 {code}".encode()])

    assert obs["weather"] == expected_name
    assert obs["weather_index"] == expected_idx


def test_all_flags_set():
    obs = FEScoreField().extract([b"58 58 61 61 61 61 0 58 0200 Y Y Y Y 53 S"])
    np.testing.assert_array_equal(obs["flags"], [1, 1, 1, 1])


def test_no_flags_set():
    obs = FEScoreField().extract([b"58 58 61 61 61 61 0 58 0200 N N N N 53 S"])
    np.testing.assert_array_equal(obs["flags"], [0, 0, 0, 0])


def test_empty_returns_valid_defaults():
    defaults = FEScoreField().empty()

    assert defaults["vitals"].shape == (8,)
    assert defaults["vitals"].dtype == INT_DTYPE
    assert defaults["flags"].shape == (4,)
    assert defaults["weather"] == UNKNOWN
    assert defaults["weather_index"] == 0


def test_extracts_fes_from_real_captures(bytes_case):
    obs = FEScoreField().extract(bytes_case["chunks"])
    expected = bytes_case["fes"]

    np.testing.assert_array_equal(obs["vitals"], expected["vitals"])
    np.testing.assert_array_equal(obs["flags"], expected["flags"])
    assert obs["reset_minutes"] == expected["reset_minutes"]
    assert obs["weather"] == expected["weather"]


def test_include_keys_must_be_space_keys():
    with pytest.raises(ValueError, match="not in full_space"):
        FEScoreField(include_keys=("vitals", "bogus"))


def test_separator_controls_are_not_whitespace_around_a_status_line():
    # \x1f is not ASCII whitespace, so it remains part of the line and prevents a match
    assert not FEScoreField().matches(b"\x1f58 58 61 61 61 61 0 58 0200 N N N N 53 F\x1f\r\n")
