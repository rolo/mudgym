from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


def test_extracts_fei_chunk():
    field = FEInventoryField()
    obs = field.extract([b"oo\r\n========\r\nbroadsword\r\n"])

    assert obs == {"portables": (), "inventory": ("broadsword",)}
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
