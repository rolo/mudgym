from itertools import product

import pytest

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

        with subtests.test(f"step-{label}"):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert info["last_command"] is not None


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
    env = MudEnv()
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
