"""Check that recorded and scripted observations satisfy every preset's advertised space."""

from pathlib import Path

import pytest

from mudgym.connections.recording import ReplayConnection
from mudgym.envs.factory import make_env

PRESETS = ["bytes", "text", "parsed", "cheats"]
RECORDINGS = Path(__file__).parents[4] / "docs" / "recordings"


@pytest.mark.parametrize("observation", PRESETS)
def test_recorded_reset_observation_is_within_the_observation_space(observation):
    replay = ReplayConnection(RECORDINGS / f"observations-{observation}.session.jsonl")
    env = make_env(observation=observation, connection=replay)
    try:
        obs, _info = env.reset()
        replay.assert_exhausted()
    finally:
        env.close()

    outside = {key: value for key, value in obs.items() if not env.observation_space.spaces[key].contains(value)}
    assert not outside, f"{observation}: {sorted(outside)} outside their declared spaces"


@pytest.mark.parametrize("observation", PRESETS)
def test_step_observation_is_within_the_observation_space(scripted_env_factory, observation):
    env = scripted_env_factory(observation=observation)
    env.reset()
    obs, _reward, _terminated, _truncated, _info = env.step("look")

    outside = {key: value for key, value in obs.items() if not env.observation_space.spaces[key].contains(value)}
    assert not outside, f"{observation}: {sorted(outside)} outside their declared spaces"
