import pytest
from gymnasium import spaces

from mudgym import make_env, make_parallel_env
from mudgym.envs.factory import OBSERVATION_PRESETS
from mudgym.envs.fields import (
    FEInventoryField,
    FEScoreField,
    FEXitsField,
    MGCheatsField,
    ObservationField,
    RawBytesField,
    SuperQuickLookField,
    instantiate_field,
)
from tests.scripted import ScriptedConnection, ScriptedProvider

PRESET_FIELD_TYPES = {
    "bytes": {RawBytesField},
    "text": set(),
    "parsed": {SuperQuickLookField, FEScoreField, FEXitsField, FEInventoryField},
    "cheats": {FEScoreField, FEXitsField, FEInventoryField, MGCheatsField},
}

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
def test_preset_uses_expected_field_types(preset):
    field_types = {type(instantiate_field(field)) for field in OBSERVATION_PRESETS[preset]}

    assert field_types == PRESET_FIELD_TYPES[preset]


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


@pytest.mark.parametrize("preset", ["cheats", "text", "bytes", "unknown"])
def test_explicit_fields_ignores_preset(scripted_env_factory, preset):
    env = scripted_env_factory(
        observation=preset,
        field_parsers=[RawBytesField(), FEScoreField()],
    )

    assert observation_keys(env) == EXPLICIT_RAW_BYTES_KEYS


@pytest.mark.parametrize("preset", ["text", "bytes", "unknown"])
def test_parallel_explicit_commandless_fields_ignore_preset(preset):
    env = make_parallel_env(2, provider=ScriptedProvider(), observation=preset, field_parsers=[RawBytesField])
    try:
        for child in env.unwrapped.envs.values():
            assert child.session.observation_line == ""
            assert observation_keys(child) == PRESET_KEYS["bytes"]
    finally:
        env.close()


def test_explicit_commandless_fields_need_no_observation_probe(scripted_env_factory):
    env = scripted_env_factory(field_parsers=[RawBytesField])
    assert env.session.observation_line == ""
    assert observation_keys(env) == PRESET_KEYS["bytes"]


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


def test_fields_providing_the_same_key_raise(scripted_env_factory):
    """Two fields may not both provide an observation key; include_keys resolves the clash."""
    with pytest.raises(ValueError, match="Duplicate observation keys"):
        scripted_env_factory(field_parsers=[SuperQuickLookField, MGCheatsField])


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
    assert set(OBSERVATION_PRESETS) == set(PRESET_KEYS) == set(PRESET_FIELD_TYPES)
