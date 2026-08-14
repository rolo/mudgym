import pytest

FORD_COLLAPSE_BYTES = (
    b"\x1b[1;37;40mmove swampward\r\n"
    b"\x1b[32mFord across river\x1b[37m.\r\n"
    b"\x1b[0;32;40mYou are standing on a ford across a fast-flowing river. To the west is a badly-paved "
    b"road, which carries on into the distance. Northwest is a ramshackle old building, and southwest "
    b"is some sort of well. South lies a forest, and north is the west bank of the river you now cross. "
    b"The ford goes beneath the water level to the east, but you can still go that way if you so "
    b"desire. \x1b[1;37;40m\x1b[0;32;40mIt is raining. \x1b[1;37;40m\r\n"
    b"Rain has swollen the river to a raging torrent! You fight your way across, but are constantly "
    b"buffeted and pounded all the way, causing you major injury!\r\n"
    b"You feel unbearably giddy.\r\n"
    b"You collapse, unconscious.\r\n"
    b"Your stamina has fallen from \x1b[0;33;40m11\x1b[1;37;40m to \x1b[31m1\x1b[37m.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40msql,fes,fex,fei\r\n"
    b"You can't wake yourself up yet!\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"\x1b[1;31;40m1\x1b[0;37;40m \x1b[1;32;40m51\x1b[0;37;40m 33 47 39 52 0 51 075 N N N N 52 R\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"up in down out swampward southwest south southeast northeast northwest west east north\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"========\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
)


def test_bytes_to_observation_defaults_field_on_player_state_refusal(scripted_env_factory):
    """A player-state refusal in an observation-command slot must not crash the parse.

    The refusal chunk occupies the field's position in the tail split, so alignment holds;
    the field keeps its empty defaults for the turn, the refusal is recorded in the info
    dict, and the player-visible line stays in the text observation.
    """
    env = scripted_env_factory(observation="parsed")
    obs, _, field_refusals = env.bytes_to_observation(
        FORD_COLLAPSE_BYTES,
        wire_lines=["move swampward", "sql,fes,fex,fei"],
        response_complete=True,
    )

    # sql was refused: its keys stay at the empty defaults
    assert obs["room_name"] == ""

    # the other observation commands answered normally and still extract
    assert obs["available_exits"].sum() == 13

    # the refusal is recorded explicitly, verbatim
    assert field_refusals == {"SuperQuickLookField": b"You can't wake yourself up yet!\r\n"}

    # the player saw the refusal and the collapse: both survive in the text observation
    assert "You can't wake yourself up yet!" in obs["text"]
    assert "You collapse, unconscious." in obs["text"]
    assert "\x1b" not in obs["text"]


VAMPIRE_BLIND_BYTES = (
    b"\x1b[31mYou are wounded by the violence of a crafty, upward blow by the vampire.\r\n"
    b"Stamina=\x1b[33m51\x1b[31m/\x1b[32m73\x1b[31m.\x1b[37m\r\n"
    b"\x1b[31mYou ably graze the vampire with a punishing spurt.\r\nDamage: 8.\x1b[37m\r\n"
    b"\x1b[0;37;40mThe vampire looks strong.\r\n"
    b"\x1b[1;37;40m\x1b[36mA fabulous, gold ring with a fearful, bloodstone setting has been dropped here. \r\n"
    b"\x1b[37mThe vampire makes some magical gestures.\r\n"
    b"\x1b[31mYou have suddenly and magically gone blind!\x1b[37m\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
    b"west\r\n"
    b"You can't just leave in the middle of a fight! You have to flee!\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40msql,fes,fex,fei\r\n"
    b"You can't see a thing, you're blind.\r\n"
    b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"\x1b[1;33;40m51\x1b[0;37;40m \x1b[1;32;40m73\x1b[0;37;40m 55 55 21 52 0 73 0200 Y N N N 52 F\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m"
    b"--\r\n"
    b"========\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
)


def test_bytes_to_observation_defaults_sql_on_blind_refusal(scripted_env_factory):
    """A mid-fight blinding must not crash the parse (captured live: vampire casts blindness).

    sql answers with the blind refusal, fes still answers its score line, fex answers a
    blank line (same shape as a dark room), and fei shows only the blind ``--`` placeholder
    above the divider. Every slot resolves and the refusal keeps sql's empty defaults.
    """
    env = scripted_env_factory(observation="parsed")
    obs, _, field_refusals = env.bytes_to_observation(
        VAMPIRE_BLIND_BYTES,
        wire_lines=["west", "sql,fes,fex,fei"],
        response_complete=True,
    )

    # sql was refused: its keys stay at the empty defaults
    assert obs["room_name"] == ""
    assert field_refusals == {"SuperQuickLookField": b"You can't see a thing, you're blind.\r\n"}

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


def test_unknown_observation_command_chunk_still_fails_loudly(scripted_env_factory):
    """Only recognised player-state refusals default; anything else raises explicitly."""
    env = scripted_env_factory(observation="parsed")
    prompt = b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m"
    raw_bytes = (
        b"look\r\n"
        b"Some narrative.\r\n"
        + prompt
        + b"sql,fes,fex,fei\r\n"
        + b"Certainly not a super quick look.\r\n"
        + prompt
        + b"\x1b[33m11\x1b[37m \x1b[1;32;40m51\x1b[0;37;40m 38 47 43 52 0 51 075 N N N N 52 R\r\n"
        + prompt
        + b"west east\r\n"
        + prompt
        + b"========\r\n"
        + prompt
    )

    with pytest.raises(RuntimeError, match="SuperQuickLookField.*found no matching response"):
        env.bytes_to_observation(
            raw_bytes,
            wire_lines=["look", "sql,fes,fex,fei"],
            response_complete=True,
        )
