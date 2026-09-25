import pytest

from mudgym.connections.registry import default_connection
from mudgym.db.index import room_id_to_index, room_name_to_index, weather_to_index
from mudgym.db.rooms import ROOM_NAMES
from mudgym.db.weather import WEATHER
from mudgym.envs.env import MudEnv
from mudgym.envs.fields.rawbytes import DEFAULT_MAX_BYTES
from mudgym.envs.tests.assertions import assert_observation_in_space, assert_observations_equal


def assert_live_observation(env, obs, info, preset):
    assert_observation_in_space(env.observation_space, obs)
    assert obs["points"] == env.unwrapped.points

    if preset in {"parsed", "cheats"}:
        assert obs["room_name"] in ROOM_NAMES
        assert obs["room_name_index"] == room_name_to_index(obs["room_name"])
        assert obs["vitals"].any()
        assert obs["available_exits"].any()
        assert obs["weather"] in WEATHER
        assert obs["weather_index"] == weather_to_index(obs["weather"])
    if preset == "parsed":
        assert "room_id" not in obs
        assert "room_id_index" not in obs
    if preset == "cheats":
        assert obs["room_id_index"] == room_id_to_index(obs["room_id"])
    if preset == "bytes":
        raw_bytes = info["raw_bytes"]
        assert obs["raw_bytes"].shape == (DEFAULT_MAX_BYTES,)
        assert obs["raw_bytes"][: len(raw_bytes)].tobytes() == raw_bytes
        assert not obs["raw_bytes"][len(raw_bytes) :].any()


@pytest.mark.parametrize("preset", ["bytes", "text", "parsed", "cheats"])
def test_live_presets_reset_step_and_reset(live_env_factory, preset):
    env = live_env_factory(observation=preset, actions="directions")

    for _ in range(2):
        obs, info = env.reset()
        assert_live_observation(env, obs, info, preset)
        persona = info["persona"]
        assert persona.isalpha()

        obs, reward, terminated, truncated, info = env.step(0)
        assert_live_observation(env, obs, info, preset)
        assert isinstance(reward, (int, float))
        assert terminated is False
        assert truncated is False
        assert info["persona"] == persona


def test_bare_env_runs_against_the_live_game():
    """A directly-built MudEnv(), no field_parsers and no wrappers, resets and steps with a text observation."""
    env = MudEnv(connection=default_connection())
    try:
        obs, info = env.reset()
        assert obs["text"]
        assert info["raw_bytes"]

        obs, reward, terminated, truncated, info = env.step("look")
        assert set(obs) == {"text", "points"}
        assert obs["points"] == env.points
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

    stepped_obs, *stepped_rest = stepped_transition
    split_obs, *split_rest = split_transition
    assert_observations_equal(split_obs, stepped_obs)

    assert split_rest[:3] == stepped_rest[:3]
    stepped_info = stepped_rest[3]
    split_info = split_rest[3]
    for key in ("raw_bytes", "render_bytes", "step", "persona", "action_rejected"):
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
        "render_bytes",
        "step",
        "persona",
        "action_rejected",
        "transport",
    }

    env.reset()
    assert env.unwrapped.step_count == 0
