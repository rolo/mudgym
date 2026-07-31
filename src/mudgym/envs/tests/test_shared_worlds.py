def test_shared_world_vector_env_players_can_see_each_other(live_vector_env_factory):
    """
    Check that both players can see each other in a shared game instance.
    """
    env = live_vector_env_factory(envs=2, worlds=1)
    env.reset()
    env.step(["mgtransport banshee me", "mgtransport banshee me"])
    obs, *_ = env.step(["look,dance,wave", "look,bow,wave"])
    assert len(obs["players"][0]) == 2
    assert len(obs["players"][1]) == 2


def test_individual_world_vector_env_players_cannot_see_each_other(live_vector_env_factory):
    """
    Check that both players cannot see each other when each has their own game world instance.
    """
    env = live_vector_env_factory(envs=2, worlds=2)
    env.reset()
    env.step(["mgtransport banshee me", "mgtransport banshee me"])
    obs, *_ = env.step(["look,dance,wave", "look,bow,wave"])
    assert len(obs["players"][0]) == 1
    assert len(obs["players"][1]) == 1


def test_shared_world_marl_env(live_parallel_env_factory):
    """
    Check that all players can see each other in a PettingZoo MARL ParallelEnv environment.
    """
    env = live_parallel_env_factory(agents=4)
    env.reset()

    action = "mgtransport banshee me"
    actions = {f"player_{i}": action for i in range(env.num_agents)}
    env.step(actions)

    action = "bow,dance,macarena"
    actions = {f"player_{i}": action for i in range(env.num_agents)}
    obs, *_ = env.step(actions)

    assert all(len(o["players"]) == env.num_agents for o in obs.values())
