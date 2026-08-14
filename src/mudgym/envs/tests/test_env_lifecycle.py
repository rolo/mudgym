from itertools import product

import numpy as np
import pytest

from mudgym.connections.registry import default_connection
from mudgym.db.rooms import ROOM_NAMES
from mudgym.envs.env import MudEnv

observations = [
    "bytes",
    "text",
    "parsed",
]
actions = [
    "directions",
]
maker_kwarg_sets = [{"observation": obs, "actions": act} for obs, act in product(observations, actions)]


@pytest.mark.parametrize("maker_kwarg_set", maker_kwarg_sets)
def test_env_lifecycle(live_env_factory, maker_kwarg_set, subtests):
    env = live_env_factory(**maker_kwarg_set)

    with subtests.test("construct"):
        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "close")

    for label in ["first", "second"]:
        with subtests.test(f"reset-{label}"):
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)
            persona = info["persona"]
            assert persona.isalpha()

        with subtests.test(f"step-{label}"):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert info["persona"] == persona


def test_env_reset(live_env_factory):
    """
    Check that the reset puts us in The Land with the correct step data.
    Uses cheats preset to get room_name from MGCheatsField.
    """
    env = live_env_factory(observation="cheats")
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, dict)

    # cheats preset includes MGCheatsField with room_name
    assert obs["room_name"] != ""
    assert obs["room_name"] != "elizabethan tearoom"
    assert obs["room_name"] in ROOM_NAMES

    # cheats includes room_id (unlike old parsed preset)
    assert "room_id" in obs


def test_bare_env_runs_against_the_live_game():
    """A directly-built MudEnv(), no field_parsers and no wrappers, resets and steps with a text observation."""
    env = MudEnv(connection=default_connection())
    try:
        obs, info = env.reset()
        assert obs["text"]
        assert info["raw_bytes"]

        obs, reward, terminated, truncated, info = env.step("look")
        assert set(obs) == {"text"}
        assert obs["text"]
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_step_matches_act_followed_by_observe(scripted_env_factory):
    stepped_env = scripted_env_factory()
    split_env = scripted_env_factory()
    stepped_env.reset()
    split_env.reset()

    stepped_transition = stepped_env.step("look")
    split_env.act("look")
    split_transition = split_env.observe()

    stepped_observation, *stepped_rest = stepped_transition
    split_observation, *split_rest = split_transition
    assert set(stepped_observation) == set(split_observation)
    for key, expected_value in stepped_observation.items():
        actual_value = split_observation[key]
        if isinstance(expected_value, np.ndarray):
            assert np.array_equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value, key

    assert split_rest[:3] == stepped_rest[:3]
    stepped_info = stepped_rest[3]
    split_info = split_rest[3]
    for key in ("raw_bytes", "step", "persona", "action_rejected"):
        assert split_info[key] == stepped_info[key], key


def test_only_player_actions_advance_the_env_step_count(scripted_env_factory):
    env = scripted_env_factory(tearoom_commands="dance")

    _, reset_info = env.reset()
    assert env.unwrapped.step_count == 0
    assert reset_info["step"] == 0

    _, _, _, _, refresh_info = env.unwrapped.observe()
    assert env.unwrapped.step_count == 0
    assert refresh_info["step"] == 0

    _, _, _, _, step_info = env.step("look")
    assert env.unwrapped.step_count == 1
    assert step_info["step"] == 1
    assert set(step_info) == {
        "raw_bytes",
        "step",
        "persona",
        "action_rejected",
    }

    env.reset()
    assert env.unwrapped.step_count == 0
