import pytest

from mudgym.envs.env import MudEnv
from tests.scripted import ScriptedConnection, make_scripted_env

PROMPT = b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"

FEI_MARKER_CHUNK = b"\x1b[0;37;40m========\r\n" + PROMPT

DRAGONFLY_WINDOW_BYTES = (
    b"\x1b[1;37;40mmove jump,sql,fes,fex,fei\r\n"
    b"\x1b[0;35;40mThe dragonfly has just flown away.\x1b[1;37;40m\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
    b"You cannot go over from here.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
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


def test_interleaved_async_output_lands_in_text_and_fields_align(scripted_env_factory):
    """An async line flushed at the head of the window must not shift field assignment.

    The fei marker anchors the tail: the four field responses are the chunks it bounds,
    and the flushed dragonfly line is narrative text the player really saw.
    """
    env = scripted_env_factory(observation="parsed")
    info = {"last_command": "move jump"}

    obs = env.bytes_to_observation(DRAGONFLY_INTERLEAVED_BYTES, info)

    assert obs["room_name"] == "fast-flowing river"
    assert obs["available_exits"].sum() == 12
    assert len(info["auto_command_chunks"]) == 4

    assert "The dragonfly has just flown away." in info["text"]
    assert "You cannot go over from here." in info["text"]
    assert "========" not in info["text"]
    assert "========" not in obs["text"]


# Condensed from a live arena capture (exports/diagnostics/mudgym-arena-dump.txt): during a fight
# the game flushes combat rounds BETWEEN auto-command responses -- here between fex and fei --
# each with a prompt reprint. The marker still bounds the window; matcher-scan assignment must
# route the fight block to narrative text and keep every field aligned.
COMBAT_INTERLEAVED_BYTES = (
    b"You hear sounds of combat, as Matthew the necromancer attacks Stephen the necromancer.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
    b"kill matthew,sql,fes,fex,fei\r\n"
    b"\x1b[0;31;40mYou attack Matthew the necromancer.\x1b[1;37;40m\r\n" + PROMPT + b"\x1b[0;37;40m"
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
    """Fight output flushed between two auto-command responses must not shift field assignment."""
    env = scripted_env_factory(observation="parsed")
    info = {"last_command": "kill matthew"}

    obs = env.bytes_to_observation(COMBAT_INTERLEAVED_BYTES, info)

    assert obs["room_name"] == "treacherous swamp"
    assert obs["available_exits"].sum() == 9
    assert len(info["auto_command_chunks"]) == 4

    assert "You attack Matthew the necromancer." in info["text"]
    assert "ducks your dreadful, frontal attack" in info["text"]
    assert "========" not in info["text"]


def test_missing_marker_means_no_field_extraction(scripted_env_factory):
    """Without the marker, chunk positions cannot be trusted: fields default, chunks stay text."""
    env = scripted_env_factory(observation="parsed")
    info = {"last_command": "move jump"}

    obs = env.bytes_to_observation(DRAGONFLY_WINDOW_BYTES, info)

    assert obs["room_name"] == ""
    assert info["auto_command_chunks"] == []
    assert 'The place known as "fast-flowing river"' in info["text"]


def test_step_batches_end_with_fei_and_marker_is_stripped(scripted_env_factory):
    env = scripted_env_factory(observation="parsed")
    env.reset()
    obs, _, _, _, info = env.step("look")
    connection = env.unwrapped.session.connection

    assert connection.commands[-1] == "look,sql,fes,fex,fei"
    assert obs["available_exits"].sum() == 8
    assert "========" not in obs["text"]
    assert "========" not in info["text"]


def test_text_mode_ends_with_fes_and_extracts_points(scripted_env_factory):
    """The text preset declares fes as its batch ender, contributing just the points key."""
    env = scripted_env_factory(observation="text")
    env.reset()
    obs, _, _, _, info = env.step("look")
    connection = env.unwrapped.session.connection

    assert env.unwrapped.auto_commands == ["fes"]
    assert connection.commands[-1] == "look,fes"
    assert set(obs) == {"text", "points"}
    assert obs["points"] == 200
    assert "75 75" not in obs["text"]


def test_bare_env_defaults_to_a_marker_only_text_field():
    """A field-less MudEnv() defaults to a marker-only fes field: it steps, and the observation is text-only."""
    env = MudEnv(connection=ScriptedConnection())
    try:
        env.reset()
        obs, _, _, _, info = env.step("look")
        connection = env.session.connection

        assert env.auto_commands == ["fes"]
        assert env.final_command == "fes"
        assert connection.commands[-1] == "look,fes"

        assert set(obs) == {"text"}
        assert obs["text"]
        assert "75 75" not in obs["text"]
        assert "75 75" not in info["text"]
    finally:
        env.close()


def test_auto_commands_without_a_marker_tail_raise():
    """The batch must end with a marker-capable field command; nothing is appended silently."""
    with pytest.raises(ValueError, match="end_of_turn_marker"):
        make_scripted_env(observation="parsed", auto_commands=["fei", "sql", "fes", "fex"])


def test_custom_auto_command_order_drives_field_claiming():
    """Field claiming follows the batch order, not the field declaration order."""
    env = make_scripted_env(observation="parsed", auto_commands=["fex", "fes", "sql", "fei"])
    env.reset()
    obs, _, _, _, _ = env.step("look")

    assert env.unwrapped.auto_commands == ["fex", "fes", "sql", "fei"]
    assert obs["room_name"] == "dally lane"
    assert obs["available_exits"].sum() == 8


def test_fes_terminated_batch_extracts_against_the_live_game(live_env_factory):
    """The fes wire marker must match the real game's SGR-interleaved status line."""
    env = live_env_factory(auto_commands=["sql", "fex", "fei", "fes"])
    obs, info = env.reset()
    obs, _, _, truncated, info = env.step("look")

    assert not truncated
    assert len(info["auto_command_chunks"]) == 4
    assert obs["vitals"].sum() > 0


def test_fes_terminates_the_batch_when_listed_last():
    """A marker-capable field's command listed last becomes the end-of-turn marker: no fei appended."""
    env = make_scripted_env(observation="parsed", auto_commands=["sql", "fex", "fei", "fes"])
    env.reset()
    obs, _, _, _, info = env.step("look")
    connection = env.unwrapped.session.connection

    assert env.unwrapped.auto_commands == ["sql", "fex", "fei", "fes"]
    assert env.unwrapped.final_command == "fes"
    assert connection.commands[-1] == "look,sql,fex,fei,fes"
    assert obs["points"] == 200
    assert obs["vitals"][0] == 75
    assert obs["room_name"] == "dally lane"
    assert "========" not in info["text"]


def test_the_reserved_command_follows_the_batch_terminator():
    """Only the batch's actual final command is reserved; a mid-batch fei is just another command."""
    env = make_scripted_env(observation="parsed", auto_commands=["sql", "fex", "fei", "fes"])
    env.reset()

    with pytest.raises(ValueError, match="reserved"):
        env.step("fes")

    env.step("fei")


def test_actions_containing_the_marker_command_are_rejected(scripted_env_factory):
    """fei is reserved: a player-issued fei would emit the divider mid-batch and desync the window."""
    env = scripted_env_factory(observation="parsed")
    env.reset()

    for action in ["fei", "look,fei", "FEI"]:
        with pytest.raises(ValueError, match="reserved"):
            env.step(action)


def test_incomplete_step_defaults_fields_even_when_the_divider_arrived(scripted_env_factory):
    """A incomplete read window cannot be trusted, marker bytes or not.

    TIMEOUT can cut the window between the divider and its trailing prompt; the transport
    reports incomplete=True and the parse must not treat the tail as marker-bounded.
    """
    env = scripted_env_factory(observation="parsed")
    info = {"last_command": "move jump", "incomplete": True}

    obs = env.bytes_to_observation(DRAGONFLY_INTERLEAVED_BYTES, info)

    assert obs["room_name"] == ""
    assert info["auto_command_chunks"] == []
    assert 'The place known as "fast-flowing river"' in info["text"]


def test_a_longer_equals_run_is_not_mistaken_for_the_marker(scripted_env_factory):
    """Only fei's exact eight-equals divider closes the window; longer ==== rules are narrative."""
    env = scripted_env_factory(observation="parsed")
    info = {"last_command": "look"}
    prompt = b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    raw_bytes = b"look,sql,fes,fex,fei\r\nSome narrative.\r\n" + prompt + b"==================\r\n" + prompt

    obs = env.bytes_to_observation(raw_bytes, info)

    assert obs["room_name"] == ""
    assert info["auto_command_chunks"] == []
    assert "==================" in info["text"]


def test_a_mid_line_equals_run_is_not_mistaken_for_the_marker(scripted_env_factory):
    """fei's divider only ever follows a line start or an ANSI SGR sequence; eight equals
    embedded mid-line are narrative."""
    env = scripted_env_factory(observation="parsed")
    info = {"last_command": "look"}
    prompt = b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    raw_bytes = b"look,sql,fes,fex,fei\r\nThe sign reads ========\r\n" + prompt

    obs = env.bytes_to_observation(raw_bytes, info)

    assert obs["room_name"] == ""
    assert info["auto_command_chunks"] == []
    assert "The sign reads ========" in info["text"]


def test_mgcheats_terminates_the_batch_when_listed_last():
    """The mgcheats closing tag serves as the marker when mgcheats ends the batch."""
    env = make_scripted_env(observation="cheats", auto_commands=["fes", "fex", "fei", "mgcheats"])
    env.reset()
    obs, _, _, _, info = env.step("look")
    connection = env.unwrapped.session.connection

    assert env.unwrapped.final_command == "mgcheats"
    assert connection.commands[-1] == "look,fes,fex,fei,mgcheats"
    assert obs["room_id"] == "groad3"
    assert obs["ticks"] == 125
    assert "[/mgcheats]" not in info["text"]
