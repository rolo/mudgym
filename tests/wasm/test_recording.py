"""Recording must preserve real WASM reset, clock and response semantics."""

import numpy as np
import pytest

from mudgym import make_env, make_parallel_env, make_vector_env
from mudgym.connections.recording import RecordingConnection, RecordingProvider
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.connections.wasm.wasmtime_provider import create_connection


def make_episode(runtime, mode, preset, capture_directory=None):
    if mode == "scalar":
        connection = create_connection(runtime=runtime, seed=123)
        provider = connection.provider
        if capture_directory is not None:
            connection = RecordingConnection(connection, capture_directory / "scalar.jsonl")
        return make_env(connection=connection, observation=preset), provider
    provider = WasmtimeProvider(runtime=runtime, worlds=1 if mode == "parallel" else None, seed=123)
    wrapped = provider
    if capture_directory is not None:
        wrapped = RecordingProvider(provider, lambda index: capture_directory / f"player{index}.jsonl")
    factory = make_parallel_env if mode == "parallel" else make_vector_env
    return factory(2, provider=wrapped, observation=preset), provider


@pytest.mark.parametrize("preset", ["text", "parsed"])
@pytest.mark.parametrize("mode", ["scalar", "vector", "parallel"])
def test_recording_preserves_seeded_observations_rewards_and_world_ticks(wasm_runtime, mode, preset, tmp_path):
    direct, direct_provider = make_episode(wasm_runtime, mode, preset)
    recorded, recorded_provider = make_episode(wasm_runtime, mode, preset, tmp_path)
    world_count = 2 if mode == "vector" else 1
    try:
        for seed in (123, 456, 123):
            expected_initial, _ = direct.reset(seed=seed)
            actual_initial, _ = recorded.reset(seed=seed)
            np.testing.assert_equal(actual_initial, expected_initial)
            assert recorded_provider._world_seeds == direct_provider._world_seeds
            assert recorded_provider.advance_worlds(0) == dict.fromkeys(range(world_count), 0)
            for tick, command in enumerate(("look", "north", "look", "mgquit"), start=1):
                if mode == "scalar":
                    action = command
                elif mode == "vector":
                    action = [command] * 2
                else:
                    action = dict.fromkeys(("player_0", "player_1"), command)
                expected = direct.step(action)
                actual = recorded.step(action)
                assert direct_provider.advance_worlds(0) == dict.fromkeys(range(world_count), tick)
                assert recorded_provider.advance_worlds(0) == direct_provider.advance_worlds(0)
                np.testing.assert_equal(actual[:4], expected[:4])
    finally:
        recorded.close()
        direct.close()
