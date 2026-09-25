"""Assertions shared by environment-level tests."""

import numpy as np


def assert_observations_equal(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert isinstance(actual[key], np.ndarray), key
            assert actual[key].shape == value.shape, key
            assert actual[key].dtype == value.dtype, key
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key


def assert_observation_in_space(observation_space, obs):
    assert set(obs) == set(observation_space.spaces)
    outside = sorted(key for key, value in obs.items() if not observation_space[key].contains(value))
    assert not outside, f"{outside} outside their declared spaces"
