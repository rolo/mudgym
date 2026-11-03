import pytest
from gymnasium import spaces

from mudgym import make_env, make_parallel_env
from mudgym.envs.factory import OBSERVATION_PRESETS
from mudgym.envs.fields import (
    FEScoreField,
    MGCheatsField,
    ObservationField,
    RawBytesField,
    SuperQuickLookField,
)
from tests.scripted import ScriptedConnection, ScriptedProvider

PRESET_KEYS = {
    "bytes": {"text", "raw_bytes", "points"},
    "text": {"text", "points"},
    "parsed": {
        "text",
        "points",
        "vitals",
        "flags",
        "reset_minutes",
        "weather",
        "weather_index",
        "available_exits",
        "available_exit_names",
        "portables",
        "inventory",
        "room_name",
        "room_name_index",
        "here",
        "features",
        "mobiles",
        "players",
    },
    "cheats": {
        "text",
        "points",
        "vitals",
        "flags",
        "reset_minutes",
        "weather",
        "weather_index",
        "available_exits",
        "available_exit_names",
        "portables",
        "inventory",
        "room_name",
        "room_name_index",
        "room_id",
        "room_id_index",
        "fighting",
        "dark",
        "glowing",
        "asleep",
        "gifted",
        "here",
    },
}


def observation_keys(env) -> set[str]:
    return set(env.observation_space.spaces)


@pytest.mark.parametrize("preset", PRESET_KEYS)
def test_preset_exposes_exact_observation_keys(scripted_env_factory, preset):
    env = scripted_env_factory(observation=preset)

    assert observation_keys(env) == PRESET_KEYS[preset]


def test_default_observation_is_parsed(scripted_env_factory):
    env = scripted_env_factory()

    assert observation_keys(env) == PRESET_KEYS["parsed"]


EXPLICIT_RAW_BYTES_KEYS = {
    "text",
    "raw_bytes",
    "points",
    "vitals",
    "flags",
    "reset_minutes",
    "weather",
    "weather_index",
}


@pytest.mark.parametrize(
    "field_parsers",
    [[RawBytesField, FEScoreField], [RawBytesField(), FEScoreField()]],
    ids=["classes", "instances"],
)
def test_explicit_fields_accept_classes_and_instances(scripted_env_factory, field_parsers):
    env = scripted_env_factory(field_parsers=field_parsers)
    assert observation_keys(env) == EXPLICIT_RAW_BYTES_KEYS


def test_explicit_fields_ignores_preset(scripted_env_factory):
    env = scripted_env_factory(
        observation="unknown",
        field_parsers=[RawBytesField(), FEScoreField()],
    )

    assert observation_keys(env) == EXPLICIT_RAW_BYTES_KEYS


def test_parallel_explicit_commandless_fields_ignore_preset():
    env = make_parallel_env(2, provider=ScriptedProvider(), observation="unknown", field_parsers=[RawBytesField])
    try:
        for child in env.unwrapped.envs.values():
            assert child.session.observation_line == ""
            assert observation_keys(child) == PRESET_KEYS["bytes"]
    finally:
        env.close()


@pytest.mark.parametrize("preset", ["text", "bytes"])
@pytest.mark.parametrize("mode", ["scalar", "parallel"])
def test_commandless_presets_send_only_the_player_action(preset, mode):
    if mode == "scalar":
        env = make_env(connection=ScriptedConnection(), observation=preset)
        action = "look"
    else:
        env = make_parallel_env(2, provider=ScriptedProvider(), observation=preset)
        action = {"player_0": "look", "player_1": "look"}
    try:
        children = [env.unwrapped] if mode == "scalar" else env.unwrapped.envs.values()
        assert all(child.session.observation_line == "" for child in children)
        env.reset()
        result = env.step(action)
        observations = [result[0]] if mode == "scalar" else result[0].values()
        for obs in observations:
            assert set(obs) == PRESET_KEYS[preset]
            assert obs["points"] == 200
            assert "75 75" not in obs["text"]
        infos = [result[-1]] if mode == "scalar" else result[-1].values()
        for info in infos:
            assert info["transport"]["sent_lines"] == ["look"]
            assert b"fes\r\n" not in info["raw_bytes"]
    finally:
        env.close()


class TextField(ObservationField):
    def full_space(self):
        return {"text": spaces.Text(max_length=10)}

    def full_empty(self):
        return {"text": ""}

    def full_extract(self, chunks, **context):
        return {"text": ""}


def test_fields_cannot_replace_the_env_text_observation(scripted_env_factory):
    with pytest.raises(ValueError, match=r"Duplicate observation keys: \['text'\]"):
        scripted_env_factory(field_parsers=[TextField, FEScoreField(include_keys=())])


def test_include_keys_resolves_duplicate_keys(scripted_env_factory):
    """Restricting one field's keys lets overlapping fields coexist."""
    env = scripted_env_factory(
        field_parsers=[SuperQuickLookField, MGCheatsField(include_keys=("room_id", "fighting"))],
    )

    keys = observation_keys(env)
    assert "room_name" in keys
    assert "room_id" in keys
    assert "fighting" in keys
    assert "dark" not in keys


def test_preset_names_match_the_public_contract():
    assert set(OBSERVATION_PRESETS) == set(PRESET_KEYS)
