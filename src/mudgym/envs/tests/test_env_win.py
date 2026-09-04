import pytest

from mudgym.db.levels import WIZARD_POINTS
from tests.scripted import ROOM_TEXT, scripted_response


@pytest.mark.parametrize("observation_mode", ["bytes", "text", "parsed", "cheats"])
@pytest.mark.parametrize("points", [WIZARD_POINTS, WIZARD_POINTS + 12_800, 3_000_000_000])
def test_winning_score_ends_the_episode_and_clamps_reward(scripted_env_factory, observation_mode, points):
    raw_bytes = f"look\r\n(+12,800 = \x1b[0;32;40m{points:,}\x1b[1;37;40m).\r\n".encode("ascii")
    env = scripted_env_factory(
        observation=observation_mode,
        responses={
            "look": (raw_bytes, False, True, {"marker_arrived": False, "matched_prompt": "OPTION"}),
        },
    )
    initial_observation, _ = env.reset()

    observation, reward, terminated, truncated, info = env.step("look")

    assert terminated is True
    assert truncated is False
    assert observation["points"] == info["points"] == WIZARD_POINTS
    assert reward == WIZARD_POINTS - initial_observation["points"]
    assert env.observation_space.contains(observation)
    assert info["raw_bytes"] == raw_bytes
    assert info["transport"]["incomplete"] is True
    assert info["transport"]["marker_arrived"] is False
    assert info["transport"]["matched_prompt"] == "OPTION"

    reset_observation, _ = env.reset()
    assert env.observation_space.contains(reset_observation)
    assert reset_observation["points"] == initial_observation["points"]


def test_winning_total_does_not_require_a_nonzero_delta(scripted_env_factory):
    raw_bytes = b"look\r\n(+0 = \x1b[0;32;40m204,800\x1b[1;37;40m).\r\n"
    env = scripted_env_factory(responses={"look": (raw_bytes, False, True, {"marker_arrived": False})})
    env.reset()

    observation, _, terminated, truncated, _ = env.step("look")

    assert (terminated, truncated) == (True, False)
    assert observation["points"] == WIZARD_POINTS


def test_a_win_with_a_complete_observation_still_closes_the_session(scripted_env_factory):
    event = b"(+12,800 = \x1b[0;32;40m204,800\x1b[1;37;40m).\r\n"
    raw_bytes = scripted_response(["look", "sql,fes,fex,fei"]).replace(ROOM_TEXT, event + ROOM_TEXT)
    env = scripted_env_factory(responses={"look": raw_bytes})
    env.reset()

    observation, _, terminated, truncated, info = env.step("look")

    assert (terminated, truncated) == (True, False)
    assert observation["points"] == WIZARD_POINTS
    assert env.unwrapped.session.connection.invalidated is True
    assert info["transport"]["incomplete"] is False


def test_an_incomplete_response_below_wizard_score_stays_truncated(scripted_env_factory):
    raw_bytes = b"look\r\n(+12,800 = \x1b[0;32;40m204,799\x1b[1;37;40m).\r\n"
    env = scripted_env_factory(responses={"look": (raw_bytes, False, True, {"marker_arrived": False})})
    env.reset()

    observation, _, terminated, truncated, _ = env.step("look")

    assert (terminated, truncated) == (False, True)
    assert observation["points"] == WIZARD_POINTS - 1
