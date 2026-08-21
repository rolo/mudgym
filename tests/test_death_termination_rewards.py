import pytest

from mudgym import make_env

OBSERVATION_MODES = ("bytes", "text", "parsed", "cheats")
SORCERER_POINTS = 13_000


@pytest.fixture(params=OBSERVATION_MODES)
def sorcerer_episode(request):
    env = make_env(observation=request.param, tearoom_commands="mgsorcerise")
    try:
        obs, _ = env.reset()
        yield env, obs
    finally:
        env.close()


def test_tempdeath_reward(sorcerer_episode):
    env, obs = sorcerer_episode
    start_points = int(obs["points"])

    observation, reward, terminated, truncated, info = env.step("fuck")
    end_points = int(observation["points"])

    assert start_points == SORCERER_POINTS
    assert terminated is True
    assert truncated is False
    assert end_points == info["points"]
    assert end_points < start_points
    assert reward == end_points - start_points


def test_permadeath_reward(sorcerer_episode):
    env, obs = sorcerer_episode
    start_points = int(obs["points"])

    observation, reward, terminated, truncated, info = env.step("fod me")
    end_points = int(observation["points"])

    assert start_points == SORCERER_POINTS
    assert terminated is True
    assert truncated is False
    assert end_points == 0
    assert end_points == info["points"]
    assert reward == -start_points
