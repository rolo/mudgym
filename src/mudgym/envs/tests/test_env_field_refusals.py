from mudgym.db.index import UNKNOWN
from mudgym.envs.fields.tests.payloads import (
    BLIZZARD_BYTES,
    FORD_COLLAPSE_BYTES,
    LIT_ROOM_WAKE_BYTES,
    VAMPIRE_BLIND_BYTES,
    VAMPIRE_WAKE_BLIND_BYTES,
)


def test_bytes_to_observation_defaults_field_on_player_state_refusal(scripted_env_factory):
    """A player-state refusal keeps empty field defaults and remains visible in text."""
    env = scripted_env_factory(observation="parsed")
    obs, _, unmatched_fields = env.bytes_to_observation(
        FORD_COLLAPSE_BYTES,
        sent_lines=["move swampward", "sql,fes,fex,fei"],
        response_complete=True,
    )

    assert obs["room_name"] == UNKNOWN
    assert unmatched_fields == ()

    # the other observation commands answered normally and still extract
    assert obs["available_exits"].sum() == 13

    # the player saw the refusal and the collapse: both survive in the text observation
    assert "You can't wake yourself up yet!" in obs["text"]
    assert "You collapse, unconscious." in obs["text"]
    assert "\x1b" not in obs["text"]


def test_bytes_to_observation_defaults_sql_on_blind_refusal(scripted_env_factory):
    """A mid-fight blinding must not crash the parse (captured live: vampire casts blindness).

    sql answers with the blind refusal, fes still answers its score line, fex answers a blank line (same shape as a dark
    room), and fei shows only the blind ``--`` placeholder above the divider.
    """
    env = scripted_env_factory(observation="parsed")
    obs, _, unmatched_fields = env.bytes_to_observation(
        VAMPIRE_BLIND_BYTES,
        sent_lines=["west", "sql,fes,fex,fei"],
        response_complete=True,
    )

    # sql was refused: its keys stay at the empty defaults
    assert obs["room_name"] == UNKNOWN
    assert unmatched_fields == ()

    # fes answered normally and still extracts
    assert obs["vitals"][0] == 51

    # blind fex is a blank response: the practical all-exits default applies, as in the dark
    assert obs["available_exits"].all()

    # the blind '--' placeholder above the fei divider is not an item
    assert obs["portables"] == ()
    assert obs["inventory"] == ()

    # the player saw the blinding and the refusal: both survive in the text observation
    assert "You have suddenly and magically gone blind!" in obs["text"]
    assert "You can't see a thing, you're blind." in obs["text"]
    assert "\x1b" not in obs["text"]


def test_step_continues_when_sql_wakes_a_blind_player(scripted_env_factory):
    env = scripted_env_factory(observation="parsed", responses={"move southwest": VAMPIRE_WAKE_BLIND_BYTES})
    env.reset()

    obs, _, terminated, truncated, info = env.step("move southwest")

    assert obs["room_name"] == UNKNOWN
    assert obs["vitals"][0] == 32
    assert info["unmatched_fields"] == ("SuperQuickLookField",)
    assert "You wake up!\nYour stamina is 32." in obs["text"]
    assert terminated is False
    assert truncated is False


def test_blizzard_sql_reply_names_the_blizzard(scripted_env_factory):
    env = scripted_env_factory(observation="parsed")
    obs, _, unmatched_fields = env.bytes_to_observation(
        BLIZZARD_BYTES,
        sent_lines=["move in", "sql,fes,fex,fei,mgcheats"],
        response_complete=True,
    )

    assert obs["room_name"] == "blizzard"
    assert obs["vitals"][0] == 72
    assert unmatched_fields == ()


def test_sql_finds_its_room_line_after_wake_up_lines(scripted_env_factory):
    env = scripted_env_factory(observation="parsed")
    obs, render_bytes, _ = env.bytes_to_observation(
        LIT_ROOM_WAKE_BYTES,
        sent_lines=["sleep", "sql,fes,fex,fei,mgcheats"],
        response_complete=True,
    )

    assert obs["room_name"] == "under well"
    assert "You wake up!\nYour stamina is 21." in obs["text"]
    assert b"\x1b[1;37;40mYou wake up!\nYour stamina is \x1b[0;33;40m21\x1b[1;37;40m." in render_bytes
    assert "The place known as" not in obs["text"]
