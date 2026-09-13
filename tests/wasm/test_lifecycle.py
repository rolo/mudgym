"""The joint command boundary is eager while observation delivery remains incremental."""

from contextlib import closing

import pytest

from mudgym.envs.factory import make_parallel_env
from mudgym.envs.lifecycle import step_players


def test_joint_actions_and_tick_precede_lazy_observation_delivery():
    with closing(make_parallel_env(2, observation="text")) as environment:
        environment.reset(seed=7)
        provider = environment.envs["player_0"].session.connection.provider
        frames = [child.last_render_bytes for child in environment.envs.values()]

        results = step_players(environment.envs, {"player_0": "look", "player_1": "inventory"}, provider.tick_for_step)

        assert provider.advance_worlds(0) == {0: 1}
        assert [child.step_count for child in environment.envs.values()] == [1, 1]
        assert [child.last_render_bytes for child in environment.envs.values()] == frames
        player, transition = next(results)
        assert player == "player_0"
        assert b"look\r\n" in transition[-1]["raw_bytes"]
        assert environment.envs["player_1"].last_render_bytes == frames[1]

        # A real closed transport fails on the later read without erasing the completed first result.
        environment.envs["player_1"].session.connection.close()
        with pytest.raises(RuntimeError, match="not open"):
            next(results)
        assert transition[-1]["raw_bytes"]
        assert provider.advance_worlds(0) == {0: 1}
