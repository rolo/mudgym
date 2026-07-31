"""
These tests check that every preset's live observations satisfy the advertised space specs.

Gymnasium only checks for real under `gym.make` with the PassiveEnvChecker wrapping the env.
"""

import pytest

PRESETS = ["bytes", "text", "parsed", "cheats"]


@pytest.mark.parametrize("observation", PRESETS)
def test_reset_observation_is_within_the_observation_space(live_env_factory, observation):
    env = live_env_factory(observation=observation)
    obs, _info = env.reset()

    outside = {key: value for key, value in obs.items() if not env.observation_space.spaces[key].contains(value)}
    assert not outside, f"{observation}: {sorted(outside)} outside their declared spaces"


@pytest.mark.parametrize("observation", PRESETS)
def test_step_observation_is_within_the_observation_space(live_env_factory, observation):
    env = live_env_factory(observation=observation)
    env.reset()
    obs, _reward, _terminated, _truncated, _info = env.step("look")

    outside = {key: value for key, value in obs.items() if not env.observation_space.spaces[key].contains(value)}
    assert not outside, f"{observation}: {sorted(outside)} outside their declared spaces"
