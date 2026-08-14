from mudgym.connections.provider import DockerExecProvider


def test_shared_world_vector_env_players_see_each_other_but_not_themselves(live_vector_env_factory):
    """
    Check that in a shared game instance each player's players field lists the other, never itself.
    """
    env = live_vector_env_factory(envs=2, provider=DockerExecProvider(worlds=1))
    env.reset()
    env.step(["mgtransport banshee me", "mgtransport banshee me"])
    obs, *_ = env.step(["look,dance,wave", "look,bow,wave"])
    assert len(obs["players"][0]) == 1
    assert len(obs["players"][1]) == 1


def test_individual_world_vector_env_players_see_no_one(live_vector_env_factory):
    """
    Check that with a game world each, players see no other players (and never themselves).
    """
    env = live_vector_env_factory(envs=2, provider=DockerExecProvider(worlds=2))
    env.reset()
    env.step(["mgtransport banshee me", "mgtransport banshee me"])
    obs, *_ = env.step(["look,dance,wave", "look,bow,wave"])
    assert len(obs["players"][0]) == 0
    assert len(obs["players"][1]) == 0


def test_shared_world_marl_env(live_parallel_env_factory):
    """
    Check that in a PettingZoo MARL ParallelEnv every player sees the others.
    """
    env = live_parallel_env_factory(agents=4)
    env.reset()

    action = "mgtransport banshee me"
    actions = {f"player_{i}": action for i in range(env.num_agents)}
    env.step(actions)

    action = "bow,dance,macarena"
    actions = {f"player_{i}": action for i in range(env.num_agents)}
    obs, *_ = env.step(actions)

    assert all(len(o["players"]) == env.num_agents - 1 for o in obs.values())
