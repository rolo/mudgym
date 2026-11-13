import numpy as np
import pytest

from mudgym.envs.fields.fescore import FEScoreField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


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
