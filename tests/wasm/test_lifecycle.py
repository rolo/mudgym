"""Player lifecycle coordination and cleanup against the live game."""

from contextlib import closing

import gymnasium as gym
import pytest

from mudgym.connections.wasm import WasmtimeProvider
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


def test_parallel_close_attempts_every_player_then_provider_and_preserves_errors(wasm_runtime):
    closed = []
    player_error = RuntimeError("player close failed")
    provider_error = OSError("provider close failed")

    class ClosingProvider(WasmtimeProvider):
        def close(self):
            closed.append(self)
            super().close()
            raise provider_error

    class ClosingEnvironment(gym.Wrapper):
        def close(self):
            closed.append(self.env)
            super().close()
            if self.env is children[0]:
                raise player_error

    provider = ClosingProvider(runtime=wasm_runtime, worlds=1)
    environment = make_parallel_env(2, provider=provider)
    try:
        environment.reset(seed=123)
        children = list(environment.envs.values())
        environment.envs = {player: ClosingEnvironment(child) for player, child in environment.envs.items()}

        with pytest.raises(ExceptionGroup, match="Player cleanup failed") as raised:
            environment.close()

        assert closed == [*children, provider]
        assert raised.value.exceptions == (player_error, provider_error)
    finally:
        WasmtimeProvider.close(provider)
