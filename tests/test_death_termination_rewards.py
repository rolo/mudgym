import pytest

from mudgym import make_env

OBSERVATION_MODES = ("bytes", "text", "parsed", "cheats")


@pytest.fixture(params=OBSERVATION_MODES)
def sorcerer_episode(request):
    env = make_env(observation=request.param, tearoom_commands="mgsorcerise")
    try:
        obs, _ = env.reset()
        assert obs["points"] > 0
        yield env, obs
    finally:
        env.close()


def test_tempdeath_reward(sorcerer_episode):
    env, obs = sorcerer_episode
    start_points = int(obs["points"])

    obs, reward, terminated, truncated, _ = env.step("fuck")
    end_points = int(obs["points"])

    assert terminated is True
    assert truncated is False
    assert end_points < start_points
    assert reward == end_points - start_points


def test_permadeath_reward(sorcerer_episode):
    env, obs = sorcerer_episode
    start_points = int(obs["points"])

    obs, reward, terminated, truncated, _ = env.step("fod me")
    end_points = int(obs["points"])

    assert terminated is True
    assert truncated is False
    assert end_points == 0
    assert reward == -start_points
