import pytest

from mudgym.envs.fields import FEInventoryField
from tests.scripted import FES_RESPONSE, TEAROOM_EXIT_TEXT, scripted_response


def test_reset_reports_the_quickscore_points_like_every_later_observation(scripted_env_factory):
    reset_bytes = scripted_response(["move north"], reset_step=True)
    fields = scripted_response(["sql,fes,fex,fei"]).replace(FES_RESPONSE, FES_RESPONSE.replace(b"0200", b"0250"))
    env = scripted_env_factory(observation="parsed", responses={"move north": reset_bytes, "sql,fes,fex,fei": fields})

    observation, _ = env.reset()

    assert env.unwrapped.points == 200
    assert observation["points"] == 200


@pytest.mark.parametrize("before_narration", [True, False])
def test_exit_points_events_establish_the_reset_score(scripted_env_factory, before_narration):
    event = b"(+50 = \x1b[0;32;40m250\x1b[1;37;40m).\r\n"
    reset_bytes = scripted_response(["move north"], reset_step=True)
    narration = event + TEAROOM_EXIT_TEXT if before_narration else TEAROOM_EXIT_TEXT + event
    reset_bytes = reset_bytes.replace(TEAROOM_EXIT_TEXT, narration)
    step_bytes = scripted_response(["look", "sql,fes,fex,fei"]).replace(
        b"look\r\n", b"look\r\n(+10 = \x1b[0;32;40m260\x1b[1;37;40m).\r\n"
    )
    env = scripted_env_factory(responses={"move north": reset_bytes, "look": step_bytes})

    observation, _ = env.reset()
    assert observation["points"] == 250

    observation, reward, _, _, _ = env.step("look")
    assert observation["points"] == 260
    assert reward == 10


def test_final_reset_points_and_rejections_are_in_the_initial_baseline(scripted_env_factory):
    entry = scripted_response(["move north"], reset_step=True)
    fields = b"(+50 = \x1b[0;32;40m250\x1b[1;37;40m).\r\n" + scripted_response(["sql,fes,fex,fei"])
    env = scripted_env_factory(
        responses={
            "move north": (entry, False, False, {"marker_arrived": True, "rejected": True}),
            "sql,fes,fex,fei": fields,
        }
    )

    observation, info = env.reset()
    assert observation["points"] == 250
    assert info["step"] == 0
    assert info["action_rejected"] and info["transport"]["rejected"]
    assert info["transport"]["bytes_length"] == len(info["raw_bytes"])
    _, reward, terminated, truncated, _ = env.step("look")
    assert reward == 0
    assert not terminated and not truncated


@pytest.mark.parametrize(
    ("command", "body", "terminated", "incomplete"),
    [
        ("qs", b"quickscore failed", False, True),
        ("move north", b"entry failed", True, False),
        ("move north", b"entry timed out", False, True),
        ("move north", b"no exit narration", False, False),
        ("sql,fes,fex,fei", b"You have died.", True, False),
        ("sql,fes,fex,fei", b"fields timed out", False, True),
        ("sql,fes,fex,fei", b"(+0 = \x1b[0;32;40m204,800\x1b[1;37;40m).\r\n", False, False),
    ],
)
def test_reset_failure_requires_a_new_reset_without_printing(
    scripted_env_factory, capsys, command, body, terminated, incomplete
):
    env = scripted_env_factory(
        render_mode="human",
        responses={command: (body, terminated, incomplete, {"marker_arrived": not incomplete})},
    )
    with pytest.raises((RuntimeError, ValueError)) as raised:
        env.reset()
    assert repr(body)[2:-1] in str(raised.value)
    assert capsys.readouterr().out == ""
    assert env.session.connection.invalidated
    with pytest.raises(RuntimeError, match="reset"):
        env.step("look")
    with pytest.raises(RuntimeError, match="reset"):
        env.observe()

    env.session.connection.responses.clear()
    observation, _ = env.reset()
    assert observation["points"] == 200
    assert capsys.readouterr().out.count("Dally Lane") == 1


def test_reset_sends_exit_separately_and_only_keeps_final_echoes(scripted_env_factory):
    env = scripted_env_factory(field_parsers=[FEInventoryField])

    observation, info = env.reset()

    assert observation["portables"] == ("necklace0",)
    assert env.observation_space.contains(observation)
    assert env.session.connection.sent_lines == [["qs"], ["move north"], ["fei"]]
    assert info["transport"]["sent_lines"] == ["fei"]
    assert b"move north" not in info["raw_bytes"]
    assert observation["text"].count("Dally Lane") == 1


def test_tearoom_exit_ignores_an_earlier_ellipsis_line(scripted_env_factory):
    env = scripted_env_factory()
    raw_bytes = (
        b"move north\r\n"
        b"An unrelated message ends here...\r\n"
        b"The mist gradually disperse away to nothingness...\r\n"
        b"Dally Lane.\r\n"
    )

    cleaned = env.unwrapped.clean_tearoom_exit(raw_bytes)

    assert cleaned == b"Dally Lane.\r\n"


def test_tearoom_exit_recognises_the_sorcerer_narration(scripted_env_factory):
    env = scripted_env_factory()
    raw_bytes = (
        b"move north\r\n"
        b"fes\r\n"
        b"Greens and browns whirl around you in seemingly aimless patterns until they suddenly slide into shape...\r\n"
        b"Dense forest.\r\n"
    )

    cleaned = env.unwrapped.clean_tearoom_exit(raw_bytes)

    assert cleaned == b"Dense forest.\r\n"


def test_separate_entry_and_fields_preserve_a_live_capture(scripted_env_factory):
    # Received bytes from DockerRunConnection on v0.3.3 (7bd9635c7a28), with a 5 ms send delay.
    raw_bytes = (
        b"move north\r\nAs you step through the opening, you become swathed in a fine, gossamer mist. "
        b"The Elizabethan tearoom fades hazily away, and vague, new shapes begin to form around you."
        b" Their outlines become more defined, their colours grow stronger, and the mist thins out i"
        b"nto pale wisps, which gradually disperse away to nothingness...\r\n\x1b[32mNarrow road between "
        b"lands\x1b[37m.\r\n\x1b[0;32;40mYou are stood on a narrow road between The Land and whence you came"
        b". To the north and south are the small foothills of a pair of majestic mountains, with a l"
        b"arge wall running round. To the west the road continues, where in the distance you can see"
        b" a thatched cottage opposite an ancient cemetery. The way out is to the east, where a shro"
        b"ud of mist covers the secret pass by which you entered The Land. \x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1"
        b";34;40m*\x1b[0;34;40m\x1b[1;37;40mfei\r\n\x1b[0;37;40m========\r\n\x1b[1;37;40m"
        b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34"
        b";40m\x1b[1;37;40m"
    )
    entry_bytes, _, fields = raw_bytes.partition(b"fei\r\n")
    env = scripted_env_factory(
        field_parsers=[FEInventoryField], responses={"move north": entry_bytes, "fei": b"fei\r\n" + fields}
    )

    observation, _ = env.reset()

    assert observation["portables"] == ()
    assert observation["inventory"] == ()
    assert env.observation_space.contains(observation)
