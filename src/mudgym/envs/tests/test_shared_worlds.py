from gymnasium.vector import AutoresetMode

from mudgym.connections.provider import DockerExecProvider


def test_shared_world_remains_playable_after_one_child_autoresets(live_vector_env_factory):
    env = live_vector_env_factory(2, provider=DockerExecProvider(worlds=1), autoreset_mode=AutoresetMode.NEXT_STEP)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)

    observation, _, terminated, truncated, _ = env.step(["shout mudgymcoherence", "look"])
    assert "mudgymcoherence" in observation["text"][1].lower()
    assert not terminated.any() and not truncated.any()

    _, _, terminated, _, _ = env.step(["quit", "look"])
    assert terminated.tolist() == [True, False]

    observation, reward, terminated, truncated, info = env.step(["quit", "look"])
    assert env.observation_space.contains(observation)
    assert reward[0] == 0
    assert info["step"].tolist() == [0, 3]
    assert not terminated.any() and not truncated.any()

    _, _, terminated, truncated, _ = env.step(["look", "look"])
    assert not terminated.any() and not truncated.any()


def test_parallel_players_hear_each_other_and_continue_after_one_quits(live_parallel_env_factory):
    env = live_parallel_env_factory(agents=2)
    env.reset()

    observations, _, terminated, truncated, _ = env.step({"player_0": "shout mudgymcoherence", "player_1": "look"})
    assert "mudgymcoherence" in observations["player_1"]["text"].lower()
    assert not any(terminated.values()) and not any(truncated.values())

    env.step({"player_0": "quit", "player_1": "look"})
    assert env.agents == ["player_1"]
    observations, _, terminated, truncated, _ = env.step({"player_1": "look"})
    assert env.observation_space("player_1").contains(observations["player_1"])
    assert not any(terminated.values()) and not any(truncated.values())
