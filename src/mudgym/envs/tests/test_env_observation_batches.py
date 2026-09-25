import pytest

from mudgym.db.index import UNKNOWN
from mudgym.envs.env import MudEnv
from mudgym.envs.fields import (
    FEInventoryField,
    FEScoreField,
    FEXitsField,
    MGCheatsField,
    RawBytesField,
    SuperQuickLookField,
)
from tests.scripted import ScriptedConnection

PROMPT = b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"

FEI_MARKER_CHUNK = b"\x1b[0;37;40m========\r\n" + PROMPT

DRAGONFLY_WINDOW_BYTES = (
    b"\x1b[1;37;40mmove jump\r\n"
    b"\x1b[0;35;40mThe dragonfly has just flown away.\x1b[1;37;40m\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
    b"You cannot go over from here.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40msql,fes,fex,fei\r\n\x1b[0;37;40m"
    b'The place known as "\x1b[1;32;40mfast-flowing river\x1b[0;37;40m" contains '
    b"\x1b[31mJessica the protector\x1b[37m and \x1b[32ma river\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n" + PROMPT + b"\x1b[0;37;40m"
    b"\x1b[1;32;40m64\x1b[0;37;40m \x1b[1;32;40m64\x1b[0;37;40m 59 59 57 57 0 64 0200 N N N N 52 F\r\n"
    + PROMPT
    + b"\x1b[0;37;40m"
    b"up down out swampward southwest south southeast northeast northwest west east north\r\n" + PROMPT
)

DRAGONFLY_INTERLEAVED_BYTES = DRAGONFLY_WINDOW_BYTES + FEI_MARKER_CHUNK


def parse(env, raw_bytes: bytes, command: str, *, response_complete: bool = True):
    obs, _, _ = env.bytes_to_observation(
        raw_bytes,
        sent_lines=[command, "sql,fes,fex,fei"],
        response_complete=response_complete,
    )
    return obs


def test_interleaved_async_output_lands_in_text_and_fields_align(scripted_env_factory):
    """An async line before the field responses stays in narrative text without shifting field assignment."""
    env = scripted_env_factory(observation="parsed")

    obs = parse(env, DRAGONFLY_INTERLEAVED_BYTES, "move jump")

    assert obs["room_name"] == "fast-flowing river"
    assert obs["available_exits"].sum() == 12
    assert "The dragonfly has just flown away." in obs["text"]
    assert "You cannot go over from here." in obs["text"]
    assert "========" not in obs["text"]


# A live arena capture with combat rounds between fex and fei responses. Each round reprints the prompt, but the fight output must stay in narrative text.
COMBAT_INTERLEAVED_BYTES = (
    b"You hear sounds of combat, as Matthew the necromancer attacks Stephen the necromancer.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
    b"kill matthew\r\n"
    b"\x1b[0;31;40mYou attack Matthew the necromancer.\x1b[1;37;40m\r\n" + PROMPT + b"sql,fes,fex,fei\r\n\x1b[0;37;40m"
    b'The place known as "\x1b[1;32;40mtreacherous swamp\x1b[0;37;40m" contains '
    b"\x1b[31mMatthew the necromancer\x1b[37m and \x1b[32mthe methane\x1b[37m.\r\n"
    b"Matthew the necromancer is carrying the following:\r\n"
    b"        nothing.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n" + PROMPT + b"\x1b[0;37;40m"
    b"\x1b[1;32;40m100\x1b[0;37;40m \x1b[1;32;40m100\x1b[0;37;40m 100 100 100 100 100 100 36798 N N N N 52 F\r\n"
    + PROMPT
    + b"\x1b[0;37;40m"
    b"out southwest south southeast northeast northwest west east north\r\n" + PROMPT + b"\x1b[0;31;40m"
    b"With singleminded determination, Matthew the necromancer ducks your dreadful, frontal attack.\x1b[1;37;40m\r\n"
    b"\x1b[31mYou are bruised by the power of a deliberate burst by Matthew the necromancer.\r\n"
    b"Stamina=\x1b[0;32;40m89\x1b[1;31;40m/\x1b[32m100\x1b[31m.\x1b[37m\r\n"
    + PROMPT
    + b"\x1b[0;37;40m========\r\n"
    + PROMPT
)


def test_combat_rounds_interleaved_between_responses_still_align(scripted_env_factory):
    """Fight output between two observation-command responses must not shift field assignment."""
    env = scripted_env_factory(observation="parsed")

    obs = parse(env, COMBAT_INTERLEAVED_BYTES, "kill matthew")

    assert obs["room_name"] == "treacherous swamp"
    assert obs["available_exits"].sum() == 9
    assert "You attack Matthew the necromancer." in obs["text"]
    assert "ducks your dreadful, frontal attack" in obs["text"]
    assert "========" not in obs["text"]


@pytest.mark.parametrize("raw_bytes", [DRAGONFLY_WINDOW_BYTES, DRAGONFLY_INTERLEAVED_BYTES])
def test_incomplete_response_keeps_field_output_as_text(scripted_env_factory, raw_bytes):
    """Incomplete responses leave fields empty even if the inventory divider arrived."""
    env = scripted_env_factory(observation="parsed")

    obs = parse(env, raw_bytes, "move jump", response_complete=False)

    assert obs["room_name"] == UNKNOWN
    assert 'The place known as "fast-flowing river"' in obs["text"]


def test_default_observation_batch_parses_fields_and_hides_inventory_divider(scripted_env_factory):
    env = scripted_env_factory(observation="parsed")
    env.reset()
    obs, _, terminated, truncated, _ = env.step("look")
    connection = env.unwrapped.session.connection

    assert (terminated, truncated) == (False, False)
    assert connection.sent_lines[-1] == ["look", "sql,fes,fex,fei"]
    assert obs["available_exits"].sum() == 8
    assert "========" not in obs["text"]


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


@pytest.mark.parametrize("field_parsers", [[FEScoreField, FEXitsField], [FEScoreField, FEXitsField, RawBytesField]])
def test_observation_fields_do_not_require_a_final_marker(scripted_env_factory, field_parsers):
    env = scripted_env_factory(field_parsers=field_parsers)
    env.reset()
    obs, _, terminated, truncated, _ = env.step("look")

    assert env.session.observation_line == "fes,fex"
    assert obs["available_exits"].sum() == 8
    assert obs["vitals"].size == 8
    assert (terminated, truncated) == (False, False)


@pytest.mark.parametrize(
    ("field_parsers", "observation_line", "room_key", "room_value"),
    [
        pytest.param(
            [
                FEXitsField,
                FEScoreField,
                SuperQuickLookField(
                    include_keys=("room_name", "room_name_index", "here", "features", "mobiles", "players")
                ),
                FEInventoryField,
            ],
            "fex,fes,sql,fei",
            "room_name",
            "dally lane",
            id="exits-first",
        ),
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


def test_reordered_fields_extract_from_the_live_game(live_env_factory):
    env = live_env_factory(
        field_parsers=[
            SuperQuickLookField(
                include_keys=("room_name", "room_name_index", "here", "features", "mobiles", "players")
            ),
            FEXitsField,
            FEInventoryField,
            FEScoreField,
        ]
    )
    env.reset()
    obs, _, _, truncated, _ = env.step("look")

    assert not truncated
    assert obs["vitals"].sum() > 0


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
