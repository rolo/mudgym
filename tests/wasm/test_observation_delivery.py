"""Initial and trailing game output survives coordinated observation boundaries."""

import json
from contextlib import closing

import numpy as np
import pytest

from mudgym import make_env, make_parallel_env
from mudgym.connections.recording import RecordingProvider, ReplayProvider
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.envs.fields import SuperQuickLookField


@pytest.mark.parametrize("preset", ["text", "bytes", "parsed"])
def test_shared_reset_keeps_each_arrival_and_the_final_peer_state(wasm_runtime, preset, tmp_path):

    def capture_path(index):
        return tmp_path / f"player{index}.jsonl"

    provider = RecordingProvider(WasmtimeProvider(runtime=wasm_runtime, worlds=1), capture_path)
    with closing(make_parallel_env(2, provider=provider, observation=preset, render_mode="ansi")) as environment:
        # With all players prepared first, seed 10 places Ada and Alba in the same room.
        observations, infos = environment.reset(seed=10)
        children = list(environment.envs.values())
        for index, child in enumerate(children):
            observation = observations[f"player_{index}"]
            raw = infos[f"player_{index}"]["raw_bytes"]
            rendered = infos[f"player_{index}"]["render_bytes"]
            assert observation["text"].count("Badly-paved road") == 1
            assert rendered.count(b"Badly-paved road") == 1
            assert b"move north\r\n" not in raw
            calls = [json.loads(line) for line in capture_path(index).read_text().splitlines()[1:]]
            sent = [call["line"] for call in calls if call["call"] == "send_line"]
            assert sent == ["qs", "move north", *([child.session.observation_line] if preset == "parsed" else [])]
            reads = [call for call in calls if call["call"] == "read_response"]
            assert [read["sent_lines"] for read in reads] == [
                ["qs"],
                ["move north"],
                [child.session.observation_line] if preset == "parsed" else [],
            ]
            if preset == "parsed":
                np.testing.assert_array_equal(
                    observation["players"], ["Alba the protector" if index == 0 else "Ada the protector"]
                )
            if preset == "bytes":
                assert observation["raw_bytes"].tobytes()[: len(raw)] == raw

        actions = {"player_0": "dance", "player_1": "dance"}
        following, _, terminated, truncated, _ = environment.step(actions)
        assert not any(terminated.values()) and not any(truncated.values())
        texts = [observation["text"] for observation in following.values()]
        assert all("Badly-paved road" not in text for text in texts)


def test_real_peer_output_after_a_field_response_survives_parsing(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1, seed=123)
    connection, peer_connection = provider.create_connections(2)
    environment = make_env(connection=connection, field_parsers=[SuperQuickLookField])
    peer = make_env(connection=peer_connection, observation="text")
    try:
        environment.reset()
        peer.reset()
        connection.read_response(None)
        connection.send_line("sql")
        peer.step("shout trailing observation")
        raw, terminated, incomplete, info = connection.read_response(None)
        assert not terminated and not incomplete
        assert b"trailing observation" in raw
        observation, rendered, _ = environment.bytes_to_observation(
            raw, sent_lines=info["sent_lines"], response_complete=True
        )
        assert observation["room_name"]
        assert "trailing observation" in observation["text"]
        assert b"trailing observation" in rendered
        assert connection.read_response(None)[0] == b""
    finally:
        environment.close()
        peer.close()
        provider.close()


@pytest.mark.parametrize("preset", ["text", "bytes", "parsed"])
def test_recording_replays_complete_coordinated_resets(wasm_runtime, preset, tmp_path):

    def capture_path(index):
        return tmp_path / f"player{index}.jsonl"

    provider = RecordingProvider(WasmtimeProvider(runtime=wasm_runtime, worlds=1), capture_path)
    expected = []
    action = {"player_0": "look", "player_1": "look"}
    with closing(make_parallel_env(2, provider=provider, observation=preset)) as recorded:
        for seed in (3, 123):
            expected.append((recorded.reset(seed=seed), recorded.step(action)))
    with closing(make_parallel_env(2, provider=ReplayProvider(capture_path), observation=preset)) as replay:
        for seed, (initial, following) in zip((3, 123), expected, strict=True):
            for actual, recorded in ((replay.reset(seed=seed), initial), (replay.step(action), following)):
                np.testing.assert_equal(actual[:-1], recorded[:-1])
                actual_infos = actual[-1].values()
                recorded_infos = recorded[-1].values()
                for actual_info, recorded_info in zip(actual_infos, recorded_infos, strict=True):
                    np.testing.assert_equal(
                        {key: value for key, value in actual_info.items() if key != "transport"},
                        {key: value for key, value in recorded_info.items() if key != "transport"},
                    )
                    for key in ("sent_lines", "rejected", "marker_arrived", "incomplete"):
                        np.testing.assert_equal(actual_info["transport"][key], recorded_info["transport"][key])
        children = replay.envs.values()
        for child in children:
            child.session.connection.assert_exhausted()
