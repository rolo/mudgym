import pytest

from mudgym.envs.fields import FEInventoryField
from tests.scripted import FES_RESPONSE, TEAROOM_EXIT_TEXT, scripted_response


def test_reset_reports_the_quickscore_points_like_every_later_observation(scripted_env_factory):
    reset_bytes = scripted_response(["move north", "sql,fes,fex,fei"], reset_step=True)
    reset_bytes = reset_bytes.replace(FES_RESPONSE, FES_RESPONSE.replace(b"0200", b"0250"))
    env = scripted_env_factory(observation="parsed", responses={"move north": reset_bytes})

    observation, _ = env.reset()

    assert env.unwrapped.points == 200
    assert observation["points"] == 200


@pytest.mark.parametrize("before_narration", [True, False])
def test_exit_points_events_establish_the_reset_score(scripted_env_factory, before_narration):
    event = b"(+50 = \x1b[0;32;40m250\x1b[1;37;40m).\r\n"
    reset_bytes = scripted_response(["move north", "sql,fes,fex,fei"], reset_step=True)
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


@pytest.mark.parametrize("echo_before_narration", [True, False])
def test_exit_removes_the_observation_echo_without_losing_inventory(scripted_env_factory, echo_before_narration):
    reset_bytes = scripted_response(["move north", "fei"], reset_step=True)
    if echo_before_narration:
        reset_bytes = reset_bytes.replace(b"fei\r\n", b"").replace(b"move north\r\n", b"move north\r\nfei\r\n")
    env = scripted_env_factory(field_parsers=[FEInventoryField], responses={"move north": reset_bytes})

    observation, _ = env.reset()

    assert observation["portables"] == ("necklace0",)
    assert env.observation_space.contains(observation)


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
