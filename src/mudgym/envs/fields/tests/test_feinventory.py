import pytest

from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (b"streetsign\r\n========\r\n", {"portables": ("streetsign",), "inventory": ()}),
        (b"========\r\nambrosia\r\n", {"portables": (), "inventory": ("ambrosia",)}),
        (b"necklace0\r\n========\r\n", {"portables": ("necklace0",), "inventory": ()}),
        (b"oo\r\n========\r\nbroadsword\r\n", {"portables": (), "inventory": ("broadsword",)}),
    ],
)
def test_extracts_fei_chunk(chunk, expected):
    field = FEInventoryField()
    obs = field.extract([chunk])

    assert obs == expected
    assert_valid_observation(field, obs)


def test_missing_divider_returns_empty_tuples():
    field = FEInventoryField()
    expected = {"portables": (), "inventory": ()}

    assert field.empty() == expected
    assert field.extract([b"x"]) == expected


def test_extracts_fei_from_real_captures(bytes_case):
    field = FEInventoryField()
    obs = field.extract(bytes_case["chunks"])

    assert obs == bytes_case["fei"]
    assert_valid_observation(field, obs)
