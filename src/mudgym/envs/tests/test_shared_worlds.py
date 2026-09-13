from contextlib import ExitStack, closing

import pytest

from mudgym import make_env
from mudgym.connections.wasm import WasmtimeProvider


def test_parallel_players_hear_each_other_and_continue_after_one_quits(live_parallel_env_factory):
    env = live_parallel_env_factory(agents=2)
    env.reset()

    obs, _, terminates, truncates, _ = env.step({"player_0": "shout mudgymcoherence", "player_1": "look"})
    assert "mudgymcoherence" in obs["player_1"]["text"].lower()
    assert not any(terminates.values()) and not any(truncates.values())

    env.step({"player_0": "quit", "player_1": "look"})
    assert env.agents == ["player_1"]
    obs, _, terminates, truncates, _ = env.step({"player_1": "look"})
    assert env.observation_space("player_1").contains(obs["player_1"])
    assert not any(terminates.values()) and not any(truncates.values())


@pytest.mark.parametrize("worlds", [1, 2])
def test_players_only_hear_shouts_from_their_own_world(wasm_runtime, worlds):
    with ExitStack() as stack:
        provider = stack.enter_context(closing(WasmtimeProvider(worlds=worlds, runtime=wasm_runtime)))
        first, second = [
            stack.enter_context(make_env(connection=connection)) for connection in provider.create_connections(2)
        ]
        first.reset()
        second.reset()
        first.act("shout mudgymisolation")
        second.act("look")
        provider.tick_for_step()
        first_observation, _, first_terminated, first_truncated, _ = first.observe()
        second_observation, _, second_terminated, second_truncated, _ = second.observe()

        assert not any((first_terminated, first_truncated, second_terminated, second_truncated))
        assert "mudgymisolation" in first_observation["text"].lower()
        assert ("mudgymisolation" in second_observation["text"].lower()) is (worlds == 1)
