"""Check that recorded and scripted observations satisfy every preset's advertised space."""

from pathlib import Path

import pytest

from mudgym.connections.recording import ReplayConnection
from mudgym.envs.factory import make_env
from mudgym.envs.tests.assertions import assert_observation_in_space

PRESETS = ["bytes", "text", "parsed", "cheats"]
RECORDINGS = Path(__file__).parents[4] / "docs" / "recordings"


@pytest.mark.parametrize("preset", PRESETS)
def test_recorded_reset_observation_is_within_the_observation_space(preset):
    replay = ReplayConnection(RECORDINGS / f"observations-{preset}.session.jsonl")
    env = make_env(observation=preset, connection=replay)
    try:
        obs, _ = env.reset()
        assert_observation_in_space(env.observation_space, obs)
        replay.assert_exhausted()
    finally:
        env.close()


@pytest.mark.parametrize("preset", PRESETS)
def test_scripted_reset_and_step_observations_fit_the_space(scripted_env_factory, preset):
    env = scripted_env_factory(observation=preset)

    obs, _ = env.reset()
    assert_observation_in_space(env.observation_space, obs)

    obs, _, _, _, _ = env.step("look")
    assert_observation_in_space(env.observation_space, obs)
