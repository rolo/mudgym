"""Reward accounting for tempdeath and permadeath."""

import pytest

from mudgym.envs.fields import FEScoreField
from tests.scripted import PROMPT, ROOM_TEXT

OBSERVATION_MODES = ("bytes", "text", "parsed", "cheats")

# Captured permadeath with no numeric points event.
COMBAT_DEATH_BYTES = (
    b"\x1b[0;30;41mYou feel your very existence severed from you...\r\n"
    b"You have been killed by the vampire.\x1b[1;37;40m\r\n"
    b"\x1b[0;31;40mNot updating persona.\x1b[1;37;40m\r\n"
)
# Captured tempdeath with its numeric points event.
SEAGULL_DEATH_BYTES = (
    b"You are splattered over a very large area, or at least most of you is.\r\n"
    b"You have changed experience level from protector to novice.\r\n"
    b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"
    b"Overall, you scored 189 points this game.\r\n"
)
QUIT_BYTES = b"\x1b[0;33;40mCheerio!\x1b[1;37;40m\r\n"


def terminal_step(action: str, body: bytes):
    """Build a terminal response without fes."""
    return action.encode("latin-1") + b"\r\n" + body, True, False, {"marker_arrived": False}


def scoring_step(action: str, *, delta: int, points: int) -> bytes:
    """Build a response with matching event and fes totals."""
    event = f"(+{delta:,} = \x1b[0;32;40m{points:,}\x1b[1;37;40m).\r\n".encode("latin-1")
    status_line = f"75 75 52 52 53 53 0 75 {points:04d} N N N N 53 R\r\n".encode("latin-1")
    return action.encode("latin-1") + b"\r\n" + event + ROOM_TEXT + PROMPT + b"fes\r\n" + status_line + PROMPT


@pytest.mark.parametrize("observation_mode", OBSERVATION_MODES)
def test_permadeath_charges_the_reset_score_in_every_observation_mode(scripted_env_factory, observation_mode):
    # the canned reset starts on 200 points
    env = scripted_env_factory(
        observation=observation_mode,
        responses={"kill vampire": terminal_step("kill vampire", COMBAT_DEATH_BYTES)},
    )
    env.reset()

    obs, reward, terminated, truncated, info = env.step("kill vampire")

    assert terminated is True
    assert truncated is False
    assert reward == -200.0
    assert info["points"] == 0
    assert obs["points"] == 0


def test_permadeath_charges_the_score_held_when_the_step_began(scripted_env_factory):
    body = b"(+100 = \x1b[0;32;40m3,100\x1b[1;37;40m).\r\n" + COMBAT_DEATH_BYTES
    env = scripted_env_factory(
        observation="text",
        responses={
            "look": scoring_step("look", delta=2800, points=3000),
            "kill vampire": terminal_step("kill vampire", body),
        },
    )
    env.reset()
    env.step("look")

    observation, reward, _, _, info = env.step("kill vampire")

    assert reward == -3000.0
    assert observation["points"] == 0
    assert info["points"] == 0


def test_permadeath_uses_the_most_recent_points_event(scripted_env_factory):
    env = scripted_env_factory(
        observation="text",
        responses={
            "look": scoring_step("look", delta=2800, points=3000),
            "kill vampire": terminal_step("kill vampire", COMBAT_DEATH_BYTES),
        },
    )
    env.reset()
    observation, _, _, _, _ = env.step("look")
    assert observation["points"] == 3000

    observation, reward, _, _, _ = env.step("kill vampire")

    assert reward == -3000.0
    assert observation["points"] == 0


def test_tempdeath_keeps_the_reward_its_own_event_states(scripted_env_factory):
    env = scripted_env_factory(observation="text", responses={"jump": terminal_step("jump", SEAGULL_DEATH_BYTES)})
    env.reset()

    observation, reward, terminated, _, info = env.step("jump")

    assert terminated is True
    assert reward == -11.0
    assert observation["points"] == 189
    assert info["points"] == 189


def test_quitting_costs_nothing(scripted_env_factory):
    env = scripted_env_factory(observation="text", responses={"quit": terminal_step("quit", QUIT_BYTES)})
    env.reset()

    observation, reward, terminated, _, info = env.step("quit")

    assert terminated is True
    assert reward == 0.0
    assert "points" not in info
    # quitting preserves the score
    assert observation["points"] == 200


def test_hidden_score_field_still_supports_first_step_permadeath_reward(scripted_env_factory):
    env = scripted_env_factory(
        field_parsers=(FEScoreField(include_keys=()),),
        responses={"fod me": terminal_step("fod me", COMBAT_DEATH_BYTES)},
    )
    initial_observation, _ = env.reset()

    observation, reward, terminated, truncated, info = env.step("fod me")

    assert set(initial_observation) == {"text"}
    assert set(observation) == {"text"}
    assert terminated is True
    assert truncated is False
    assert reward == -200.0
    assert info["points"] == 0
