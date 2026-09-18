"""Recording must preserve real WASM reset, clock and response semantics."""

from contextlib import closing

import numpy as np
import pytest

from mudgym import make_env, make_parallel_env
from mudgym.connections.recording import RecordingConnection, RecordingProvider, ReplayConnection
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.connections.wasm.wasmtime_provider import create_connection


def make_episode(runtime, mode, preset, capture_directory=None):
    if mode == "scalar":
        connection = create_connection(runtime=runtime, seed=123)
        provider = connection.provider
        if capture_directory is not None:
            connection = RecordingConnection(connection, capture_directory / "scalar.jsonl")
        return make_env(connection=connection, observation=preset), provider
    provider = WasmtimeProvider(runtime=runtime, worlds=1, seed=123)
    wrapped = provider
    if capture_directory is not None:
        wrapped = RecordingProvider(provider, lambda index: capture_directory / f"player{index}.jsonl")
    return make_parallel_env(2, provider=wrapped, observation=preset), provider


@pytest.mark.parametrize("preset", ["text", "parsed"])
@pytest.mark.parametrize("mode", ["scalar", "parallel"])
def test_recording_preserves_seeded_observations_rewards_and_world_ticks(wasm_runtime, mode, preset, tmp_path):
    direct, direct_provider = make_episode(wasm_runtime, mode, preset)
    recorded, recorded_provider = make_episode(wasm_runtime, mode, preset, tmp_path)
    try:
        for seed in (123, 456, 123):
            expected_initial, _ = direct.reset(seed=seed)
            actual_initial, _ = recorded.reset(seed=seed)
            np.testing.assert_equal(actual_initial, expected_initial)
            assert recorded_provider._world_seeds == direct_provider._world_seeds
            assert recorded_provider.advance_worlds(0) == {0: 0}
            for tick, command in enumerate(("look", "north", "look", "mgquit"), start=1):
                action = command if mode == "scalar" else dict.fromkeys(("player_0", "player_1"), command)
                expected = direct.step(action)
                actual = recorded.step(action)
                assert direct_provider.advance_worlds(0) == {0: tick}
                assert recorded_provider.advance_worlds(0) == direct_provider.advance_worlds(0)
                np.testing.assert_equal(actual[:4], expected[:4])
    finally:
        recorded.close()
        direct.close()


@pytest.mark.parametrize("preset", ["parsed", "bytes"])
def test_replay_preserves_persona_and_response_metadata(wasm_runtime, tmp_path, preset):
    path = tmp_path / "persona.jsonl"
    expected_persona = {"name": "Zavren", "sex": "female"}
    connection = create_connection(runtime=wasm_runtime, persona="Zavren", sex="f")
    with closing(make_env(connection=RecordingConnection(connection, path), observation=preset)) as env:
        expected = [env.reset(seed=123), env.step("look")]
        assert expected[1][-1]["transport"]["sent_lines"] == (
            ["look", "sql,fes,fex,fei"] if preset == "parsed" else ["look"]
        )
    replay = ReplayConnection(path)
    with closing(make_env(connection=replay, observation=preset)) as env:
        actual = [env.reset(seed=123), env.step("look")]
        replay.assert_exhausted()
    for original, replayed in zip(expected, actual, strict=True):
        np.testing.assert_equal(replayed[:-1], original[:-1])
        original_info, replayed_info = original[-1], replayed[-1]
        assert original_info["persona"] == replayed_info["persona"] == "Zavren"
        assert original_info["persona_sex"] == replayed_info["persona_sex"] == "female"
        assert original_info["transport"]["persona"] == replayed_info["transport"]["persona"] == expected_persona
        assert original_info["transport"]["world_seed"] == replayed_info["transport"]["world_seed"] == 123
        for key in ("raw_bytes", "render_bytes"):
            assert replayed_info[key] == original_info[key]
        assert replayed_info["transport"]["sent_lines"] == original_info["transport"]["sent_lines"]
