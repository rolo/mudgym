from contextlib import ExitStack, closing

import pytest

from mudgym import make_env
from mudgym.connections.wasm import WasmProvider


@pytest.mark.parametrize("worlds", [1, 2])
def test_players_only_hear_shouts_from_their_own_world(wasm_runtime, worlds):
    with ExitStack() as stack:
        provider = stack.enter_context(closing(WasmProvider(worlds=worlds, runtime=wasm_runtime)))
        first, second = [
            stack.enter_context(make_env(connection=connection)) for connection in provider.create_connections(2)
        ]
        first.reset()
        second.reset()
        first.act("shout mudgymisolation")
        second.act("look")
        provider.tick_for_step()
        first_obs, _, first_terminated, first_truncated, _ = first.observe()
        second_obs, _, second_terminated, second_truncated, _ = second.observe()

        assert not any((first_terminated, first_truncated, second_terminated, second_truncated))
        assert "mudgymisolation" in first_obs["text"].lower()
        assert ("mudgymisolation" in second_obs["text"].lower()) is (worlds == 1)
