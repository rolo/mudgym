import numpy as np
from gymnasium import spaces as gym_spaces

from mudgym.db.directions import DIRECTIONS
from mudgym.db.index import DIRECTION_COUNT
from mudgym.envs.fields.fexits import FEXitsField
from mudgym.envs.fields.tests.helpers import assert_valid_observation


def test_available_exits_is_a_gymnasium_action_mask():
    obs = FEXitsField().extract([b"north"])
    action_space = gym_spaces.Discrete(DIRECTION_COUNT)

    assert obs["available_exits"].dtype == np.dtype(np.int8)
    assert action_space.sample(mask=obs["available_exits"]) == DIRECTIONS.index("north")


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
