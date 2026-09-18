from contextlib import closing

import pytest

from mudgym import make_env
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.connections.wasm.engine_contract import MAX_SAFE_INTEGER


@pytest.mark.parametrize("seed_options", [{}, {"seed": None}])
def test_omitted_seed_varies_between_envs_and_can_be_replayed(wasm_runtime, seed_options):
    seeds = set()
    for attempt in range(4):
        with closing(make_env(connection_kwargs={"runtime": wasm_runtime, **seed_options})) as env:
            obs, info = env.reset()
            seed = info["transport"]["world_seed"]
            assert 0 <= seed <= MAX_SAFE_INTEGER
            seeds.add(seed)
            repeated_obs, repeated_info = env.reset()
            assert repeated_info["transport"]["world_seed"] == seed
            assert repeated_info["transport"]["persona"] == info["transport"]["persona"]
            assert repeated_obs["text"] == obs["text"]
    assert len(seeds) > 1

    with closing(make_env(connection_kwargs={"runtime": wasm_runtime, "seed": seed})) as replay:
        replay_obs, replay_info = replay.reset()
        assert replay_info["transport"]["world_seed"] == seed
        assert replay_info["transport"]["persona"] == info["transport"]["persona"]
        assert replay_obs["text"] == obs["text"]


@pytest.mark.parametrize("seed", [None, 0, MAX_SAFE_INTEGER - 1])
def test_provider_offsets_each_world_from_its_resolved_seed(wasm_runtime, seed):
    with closing(WasmtimeProvider(runtime=wasm_runtime, worlds=2, seed=seed)) as provider:
        connections = provider.create_connections(2)
        if seed is not None:
            assert provider.seed == seed
        provider.reset()
        for index, connection in enumerate(connections):
            connection.reset()
            connection.send_line("score")
            raw_bytes, terminated, incomplete, info = connection.read_response()
            assert not terminated and not incomplete
            assert info["world_seed"] == provider.seed + index
            assert info["world_seed"] <= MAX_SAFE_INTEGER


@pytest.mark.parametrize("seed", [-1, True, 1.5, MAX_SAFE_INTEGER + 1])
def test_provider_rejects_invalid_explicit_seeds(wasm_runtime, seed):
    with pytest.raises(ValueError, match="safe integer range"):
        WasmtimeProvider(runtime=wasm_runtime, seed=seed)
