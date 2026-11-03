from contextlib import closing

import pytest

from mudgym import make_env
from mudgym.connections.wasm import WasmProvider, WasmtimeRuntime
from mudgym.connections.wasm.engine_contract import MAX_SAFE_INTEGER


class RefusingRuntime(WasmtimeRuntime):
    """The real runtime, refusing to build the next world once ``refuse_next`` is set."""

    refuse_next = False

    def create_world(self, *, seed, **options):
        if self.refuse_next:
            self.refuse_next = False
            raise RuntimeError(f"world construction refused for seed {seed}")
        return super().create_world(seed=seed, **options)


def test_omitted_seed_varies_between_envs_and_its_first_world_can_be_replayed(wasm_runtime):
    seeds = set()
    for attempt in range(4):
        with closing(make_env(connection_kwargs={"runtime": wasm_runtime})) as env:
            obs, info = env.reset()
            seed = info["transport"]["world_seed"]
            assert 0 <= seed <= MAX_SAFE_INTEGER
            seeds.add(seed)
            assert env.reset()[1]["transport"]["world_seed"] != seed
    assert len(seeds) > 1

    with closing(make_env(connection_kwargs={"runtime": wasm_runtime, "seed": seed})) as replay:
        replay_obs, replay_info = replay.reset()
        assert replay_info["transport"]["world_seed"] == seed
        assert replay_info["transport"]["persona"] == info["transport"]["persona"]
        assert replay_obs["text"] == obs["text"]


@pytest.mark.parametrize("seed", [None, 0, MAX_SAFE_INTEGER - 1])
def test_provider_offsets_each_world_from_its_resolved_seed(wasm_runtime, seed):
    with closing(WasmProvider(runtime=wasm_runtime, worlds=2, seed=seed)) as provider:
        connections = provider.create_connections(2)
        if seed is not None:
            assert provider.seed == seed
        provider.reset()
        for index, connection in enumerate(connections):
            connection.reset()
            connection.send_line("score")
            _raw_bytes, terminated, incomplete, info = connection.read_response()
            assert not terminated and not incomplete
            assert info["world_seed"] == provider.seed + index
            assert info["world_seed"] <= MAX_SAFE_INTEGER


@pytest.mark.parametrize("seed", [-1, True, 1.5, MAX_SAFE_INTEGER + 1])
def test_provider_rejects_invalid_explicit_seeds(wasm_runtime, seed):
    with pytest.raises(ValueError, match="safe integer range"):
        WasmProvider(runtime=wasm_runtime, seed=seed)


def test_unseeded_resets_move_to_new_worlds_that_the_first_seed_reproduces(wasm_runtime):
    runs = []
    for attempt in range(2):
        with closing(make_env(connection_kwargs={"runtime": wasm_runtime})) as env:
            seeds = [env.reset(seed=123)[1]["transport"]["world_seed"]]
            seeds += [env.reset()[1]["transport"]["world_seed"] for reset in range(2)]
            runs.append(seeds)
    assert runs[0][0] == 123
    assert len(set(runs[0])) == 3
    assert runs[0] == runs[1]


@pytest.mark.parametrize("failed_seed", [456, None])
def test_a_failed_reset_leaves_the_worlds_later_unseeded_resets_draw_unchanged(failed_seed):
    runtime = RefusingRuntime()
    seeds = []
    for with_failed_reset in (False, True):
        with closing(make_env(connection_kwargs={"runtime": runtime})) as env:
            assert env.reset(seed=123)[1]["transport"]["world_seed"] == 123
            if with_failed_reset:
                runtime.refuse_next = True
                with pytest.raises(RuntimeError, match="refused"):
                    env.reset(seed=failed_seed)
            seeds.append(env.reset()[1]["transport"]["world_seed"])
    assert seeds[0] != 123
    assert seeds[0] == seeds[1]


@pytest.mark.parametrize("failed_seed", [456, None])
def test_a_failed_reset_of_one_connection_leaves_the_worlds_it_draws_later_unchanged(failed_seed):
    runtime = RefusingRuntime()
    seeds = []
    for with_failed_reset in (False, True):
        with closing(WasmProvider(runtime=runtime, worlds=1)) as provider:
            (connection,) = provider.create_connections(1)
            provider.reset(seed=123)
            connection.reset()
            if with_failed_reset:
                runtime.refuse_next = True
                with pytest.raises(RuntimeError, match="refused"):
                    connection.reset(seed=failed_seed)
            connection.reset()
            connection.send_line("look")
            seeds.append(connection.read_response()[3]["world_seed"])
    assert seeds[0] != 123
    assert seeds[0] == seeds[1]
