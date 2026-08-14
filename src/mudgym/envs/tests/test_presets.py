import pytest
from gymnasium import spaces

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

PRESET_FIELD_TYPES = {
    "bytes": {RawBytesField, FEScoreField},
    "text": {FEScoreField},
    "parsed": {SuperQuickLookField, FEScoreField, FEXitsField, FEInventoryField},
    "cheats": {FEScoreField, FEXitsField, FEInventoryField, MGCheatsField},
}

PRESET_KEYS = {
    "bytes": {"text", "raw_bytes"},
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


@pytest.mark.parametrize("preset", OBSERVATION_PRESETS)
def test_preset_uses_expected_field_types(preset):
    field_types = {type(instantiate_field(field)) for field in OBSERVATION_PRESETS[preset]}

    assert field_types == PRESET_FIELD_TYPES[preset]


@pytest.mark.parametrize("preset", OBSERVATION_PRESETS)
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


def test_explicit_fields_instances(scripted_env_factory):
    env = scripted_env_factory(field_parsers=[RawBytesField(), FEScoreField()])

    assert observation_keys(env) == EXPLICIT_RAW_BYTES_KEYS


def test_explicit_fields_classes(scripted_env_factory):
    env = scripted_env_factory(field_parsers=[RawBytesField, FEScoreField])

    assert observation_keys(env) == EXPLICIT_RAW_BYTES_KEYS


def test_explicit_fields_ignores_preset(scripted_env_factory):
    env = scripted_env_factory(
        observation="cheats",
        field_parsers=[RawBytesField(), FEScoreField()],
    )

    assert observation_keys(env) == EXPLICIT_RAW_BYTES_KEYS


def test_explicit_fields_without_a_marker_field_raise(scripted_env_factory):
    """Every env needs a batch ender: a configured field declaring an end_of_turn_marker."""
    with pytest.raises(ValueError, match="declare a command"):
        scripted_env_factory(field_parsers=[RawBytesField])


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
