"""WASM completion comes from engine calls without marker-only probes."""

from contextlib import closing

import pytest

from mudgym import make_env, make_parallel_env, make_vector_env
from mudgym.connections.recording import RecordingConnection, ReplayConnection
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.envs.fields import FEInventoryField, SuperQuickLookField


@pytest.mark.parametrize("command", ["say café", "say €100"])
def test_non_ascii_commands_are_rejected_without_losing_pending_output(wasm_runtime, command):
    with closing(WasmtimeProvider(runtime=wasm_runtime, worlds=1, seed=123)) as provider:
        player, peer = provider.create_connections(2)
        player.reset()
        peer.reset()
        peer.send_line('tell "pending message" to Ada')
        peer.read_response(None)

        with pytest.raises(ValueError, match="outside ASCII"):
            player.send_line(command)

        player.send_line("look")
        raw, terminated, incomplete, info = player.read_response(None)
        assert b"pending message" in raw
        assert b"Elizabethan tearoom" in raw
        assert not terminated and not incomplete
        assert info["sent_lines"] == ["look"]
        assert player.read_response(None)[0] == b""


@pytest.mark.parametrize("preset", ["text", "bytes"])
@pytest.mark.parametrize("mode", ["scalar", "vector", "parallel"])
def test_text_and_bytes_presets_send_only_the_player_action(wasm_runtime, mode, preset):
    if mode == "scalar":
        environment = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}, observation=preset)
        action = "look"
    elif mode == "vector":
        environment = make_vector_env(2, provider=WasmtimeProvider(runtime=wasm_runtime), observation=preset)
        action = ["look", "look"]
    else:
        environment = make_parallel_env(
            2, provider=WasmtimeProvider(runtime=wasm_runtime, worlds=1), observation=preset
        )
        action = {"player_0": "look", "player_1": "look"}
    try:
        environment.reset(seed=123)
        observations, _, terminated, truncated, info = environment.step(action)
        if mode == "scalar":
            pairs = [(observations, info)]
            assert not terminated and not truncated
        elif mode == "vector":
            pairs = [
                (
                    {key: value[index] for key, value in observations.items()},
                    {
                        "raw_bytes": info["raw_bytes"][index],
                        "transport": {key: values[index] for key, values in info["transport"].items()},
                    },
                )
                for index in range(2)
            ]
            assert not terminated.any() and not truncated.any()
        else:
            pairs = [(observations[key], info[key]) for key in observations]
            assert not any(terminated.values()) and not any(truncated.values())
        for observation, details in pairs:
            assert observation["text"] and observation["points"] == 200
            assert details["transport"]["sent_lines"] == ["look"]
            assert b"fes\r\n" not in details["raw_bytes"]
    finally:
        environment.close()


def test_wasm_accepts_an_observation_field_without_an_end_marker(wasm_runtime):
    environment = make_env(
        connection="wasm", connection_kwargs={"runtime": wasm_runtime}, field_parsers=[SuperQuickLookField]
    )
    try:
        initial, _ = environment.reset(seed=123)
        observation, _, terminated, truncated, info = environment.step("look")
        assert not terminated and not truncated
        assert observation["room_name"] == initial["room_name"]
        assert info["transport"]["sent_lines"] == ["look", "sql"]
    finally:
        environment.close()


def test_recording_and_replay_retain_marker_free_presets(wasm_runtime, tmp_path):
    provider = WasmtimeProvider(runtime=wasm_runtime, seed=123)
    connection = provider.create_connections(1)[0]
    capture = tmp_path / "wasm.jsonl"
    recording = RecordingConnection(connection, capture)
    environment = make_env(connection=recording, observation="bytes", world_ticker=provider.tick_for_step)
    try:
        initial, _ = environment.reset()
        result, reward, terminated, truncated, info = environment.step("look")
        assert info["transport"]["sent_lines"] == ["look"]
    finally:
        environment.close()
        provider.close()
    replay = ReplayConnection(capture)
    environment = make_env(connection=replay, observation="bytes")
    try:
        replay_initial, _ = environment.reset()
        replay_result, replay_reward, replay_terminated, replay_truncated, replay_info = environment.step("look")
        assert replay_initial["text"] == initial["text"]
        assert replay_result["text"] == result["text"]
        assert (replay_reward, replay_terminated, replay_truncated) == (reward, terminated, truncated)
        assert replay_info["raw_bytes"] == info["raw_bytes"]
        replay.assert_exhausted()
    finally:
        environment.close()


@pytest.mark.parametrize(
    "observation_options",
    [
        {"observation": "text"},
        {"observation": "bytes"},
        {"field_parsers": ()},
        {"field_parsers": (FEInventoryField,)},
    ],
    ids=["text", "bytes", "no-fields", "inventory-only"],
)
def test_points_track_combat_gains_flee_losses_and_terminal_score_without_fes(wasm_runtime, observation_options):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1, seed=123)
    player, opponent = provider.create_connections(2)
    environment = make_env(connection=player, **observation_options)

    def step(command):
        result = environment.step(command)
        assert environment.observation_space.contains(result[0])
        expected_lines = [command]
        if not result[2] and environment.session.observation_line:
            expected_lines.append(environment.session.observation_line)
        assert result[4]["transport"]["sent_lines"] == expected_lines
        assert b"fes\r\n" not in result[4]["raw_bytes"]
        return result

    try:
        initial, _ = environment.reset()
        assert initial["points"] == 200
        opponent.reset()
        opponent.send_line("north")
        opponent.read_response(None)
        for command in ("west", "kill Alba"):
            observation, reward, terminated, truncated, _ = step(command)
            assert observation["points"] == 200 and reward == 0
            assert not terminated and not truncated

        observation, reward, terminated, truncated, info = step("flee east")
        assert b"Persona saved on -51 = \x1b[0;31;40m149" in info["raw_bytes"]
        assert observation["points"] == info["points"] == 149 and reward == -51
        assert not terminated and not truncated

        for command in ("west", "kill Alba"):
            observation, reward, terminated, truncated, _ = step(command)
            assert observation["points"] == 149 and reward == 0
            assert not terminated and not truncated
        for _ in range(40):
            observation, reward, terminated, truncated, info = step("look")
            assert not terminated and not truncated
            if b"You have killed Alba" in info["raw_bytes"]:
                break
            assert observation["points"] == 149 and reward == 0
        else:
            pytest.fail("Seeded combat did not produce Alba's defeat within 40 steps")
        assert b"Persona saved on +120 = \x1b[0;32;40m269" in info["raw_bytes"]
        assert observation["points"] == info["points"] == 269 and reward == 120

        observation, reward, terminated, truncated, info = step("jump")
        assert b"Persona saved on -12 = \x1b[0;31;40m257" in info["raw_bytes"]
        assert b"Overall, you scored 257 points this game." in info["raw_bytes"]
        assert observation["points"] == info["points"] == 257 and reward == -12
        assert terminated and not truncated

        provider.reset(seed=123)
        observation, _ = environment.reset()
        assert observation["points"] == initial["points"]
        assert environment.observation_space.contains(observation)
    finally:
        environment.close()
        provider.close()
