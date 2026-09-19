"""Gymnasium vectorisation keeps scalar MudGym worlds independent."""

from contextlib import closing

import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from mudgym import make_env


def test_selective_vector_reset_preserves_the_continuing_world(wasm_runtime):
    with closing(
        SyncVectorEnv(
            [lambda: make_env(connection_kwargs={"runtime": wasm_runtime}) for index in range(2)],
            autoreset_mode=AutoresetMode.DISABLED,
        )
    ) as envs:
        envs.reset(seed=123)
        providers = [child.session.connection.provider for child in envs.envs]
        finished_world, continuing_world = [provider._ordered_worlds[0] for provider in providers]

        obs, _rewards, terminates, truncates, _infos = envs.step(("mgquit", "look"))
        np.testing.assert_array_equal(terminates, [True, False])
        np.testing.assert_array_equal(truncates, [False, False])
        assert continuing_world.current_tick() == 1

        reset_obs, _reset_infos = envs.reset(options={"reset_mask": terminates | truncates})

        assert providers[0]._ordered_worlds[0] is not finished_world
        assert providers[1]._ordered_worlds[0] is continuing_world
        assert continuing_world.current_tick() == 1
        assert [child.step_count for child in envs.envs] == [0, 1]
        for key in obs:
            np.testing.assert_equal(reset_obs[key][1], obs[key][1])

        obs, _rewards, terminates, truncates, infos = envs.step(("look", "look"))
        assert not terminates.any() and not truncates.any()
        assert [provider._ordered_worlds[0].current_tick() for provider in providers] == [1, 2]
        assert [child.step_count for child in envs.envs] == [1, 2]
        assert envs.observation_space.contains(obs)
        assert all(infos["raw_bytes"])
