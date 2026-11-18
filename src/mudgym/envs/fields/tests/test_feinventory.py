from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


def test_extracts_fei_chunk():
    field = FEInventoryField()
    obs = field.extract([b"oo\r\n========\r\ndaisy chain\r\n"])

    assert obs == {"portables": (), "inventory": ("daisy chain",)}
    assert_valid_observation(field, obs)


def test_extracts_fei_from_real_captures(bytes_case):
    field = FEInventoryField()
    obs = field.extract(bytes_case["chunks"])

    assert obs == bytes_case["fei"]
    assert_valid_observation(field, obs)
