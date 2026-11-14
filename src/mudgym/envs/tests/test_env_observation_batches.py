import pytest

from mudgym.envs.env import MudEnv
from mudgym.envs.fields import (
    FEInventoryField,
    FEScoreField,
    FEXitsField,
    MGCheatsField,
    SuperQuickLookField,
)
from tests.scripted import ScriptedConnection


def test_bare_env_defaults_to_text_and_points_without_a_score_probe():
    env = MudEnv(connection=ScriptedConnection())
    try:
        env.reset()
        obs = env.step("look")[0]
        connection = env.session.connection

        assert env.session.observation_line == ""
        assert connection.sent_lines[-1] == ["look"]

        assert set(obs) == {"text", "points"}
        assert obs["points"] == 200
        assert obs["text"]
        assert "75 75" not in obs["text"]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("field_parsers", "observation_line", "room_key", "room_value"),
    [
        pytest.param(
            [
                SuperQuickLookField(
                    include_keys=("room_name", "room_name_index", "here", "features", "mobiles", "players")
                ),
                FEXitsField,
                FEInventoryField,
                FEScoreField,
            ],
            "sql,fex,fei,fes",
            "room_name",
            "dally lane",
            id="score-last",
        ),
        pytest.param(
            [FEScoreField, FEXitsField, FEInventoryField, MGCheatsField],
            "fes,fex,fei,mgcheats",
            "room_id",
            "groad3",
            id="cheats-last",
        ),
    ],
)
def test_field_order_controls_commands_and_response_claiming(
    scripted_env_factory, field_parsers, observation_line, room_key, room_value
):
    env = scripted_env_factory(field_parsers=field_parsers)
    env.reset()
    obs, _, terminated, truncated, _ = env.step("look")

    assert env.session.observation_line == observation_line
    assert env.session.connection.sent_lines[-1] == ["look", observation_line]
    assert (terminated, truncated) == (False, False)
    assert obs["points"] == 200
    assert obs["vitals"][0] == 75
    assert obs["available_exits"].sum() == 8
    assert obs[room_key] == room_value
    assert "ticks" not in obs
    assert "========" not in obs["text"]
    assert "[/mgcheats]" not in obs["text"]


def test_speech_keeps_narrative_and_populates_observation_fields(scripted_env_factory):
    env = scripted_env_factory()
    env.reset()

    obs, _reward, terminated, truncated, _info = env.step("say hello")

    assert truncated is False
    assert terminated is False
    connection = env.unwrapped.session.connection
    assert connection.sent_lines[-1] == ["say hello", "sql,fes,fex,fei"]
    # the auto command responses were claimed into fields, not swallowed into the speech
    assert obs["available_exits"].sum() == 8
    assert "say hello" not in obs["text"]
    assert "sql,fes,fex,fei" not in obs["text"]
    assert 'says "hello"' in obs["text"]
