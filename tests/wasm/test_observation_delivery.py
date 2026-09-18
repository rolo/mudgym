"""Initial and trailing game output survives coordinated observation boundaries."""

import json
from contextlib import closing

import numpy as np
import pytest

from mudgym import make_env, make_parallel_env
from mudgym.connections.recording import RecordingConnection, RecordingProvider, ReplayConnection, ReplayProvider
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.envs.fields import SuperQuickLookField


@pytest.mark.parametrize("preset", ["text", "bytes", "parsed"])
def test_shared_reset_keeps_each_arrival_and_the_final_peer_state(wasm_runtime, preset, tmp_path):

    def capture_path(index):
        return tmp_path / f"player{index}.jsonl"

    provider = RecordingProvider(
        WasmtimeProvider(
            runtime=wasm_runtime,
            worlds=1,
            personas=(("Aaron", "male"), ("Abbie", "male")),
        ),
        capture_path,
    )
    with closing(make_parallel_env(2, provider=provider, observation=preset, render_mode="ansi")) as env:
        # With all players prepared first, seed 10 places Aaron and Abbie in the same room.
        obs, infos = env.reset(seed=10)
        children = list(env.envs.values())
        for index, child in enumerate(children):
            player_obs = obs[f"player_{index}"]
            raw = infos[f"player_{index}"]["raw_bytes"]
            rendered = infos[f"player_{index}"]["render_bytes"]
            assert player_obs["text"].count("Badly-paved road") == 1
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
                    player_obs["players"], ["Abbie the protector" if index == 0 else "Aaron the protector"]
                )
            if preset == "bytes":
                assert player_obs["raw_bytes"].tobytes()[: len(raw)] == raw

        actions = {"player_0": "dance", "player_1": "dance"}
        obs, _, terminates, truncates, _ = env.step(actions)
        assert not any(terminates.values()) and not any(truncates.values())
        texts = [player_obs["text"] for player_obs in obs.values()]
        assert all("Badly-paved road" not in text for text in texts)


def test_real_peer_output_after_a_field_response_survives_parsing(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1, seed=123)
    connection, peer_connection = provider.create_connections(2)
    env = make_env(connection=connection, field_parsers=[SuperQuickLookField])
    peer = make_env(connection=peer_connection, observation="text")
    try:
        env.reset()
        peer.reset()
        connection.read_response()
        connection.send_line("sql")
        peer.step("shout trailing observation")
        raw, terminated, incomplete, info = connection.read_response()
        assert not terminated and not incomplete
        assert b"trailing observation" in raw
        obs, rendered, _ = env.bytes_to_observation(raw, sent_lines=info["sent_lines"], response_complete=True)
        assert obs["room_name"]
        assert "trailing observation" in obs["text"]
        assert b"trailing observation" in rendered
        assert connection.read_response()[0] == b""
    finally:
        env.close()
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


@pytest.mark.parametrize("phase", ["action", "observation"])
@pytest.mark.parametrize("preset", ["parsed", "text", "bytes"])
def test_pending_combat_death_survives_the_next_command_and_reset(wasm_runtime, phase, preset, tmp_path):
    provider = WasmtimeProvider(
        runtime=wasm_runtime,
        worlds=1,
        seed=123,
        personas=(("Aaron", "male"), ("Abbie", "male")),
    )
    victim, survivor = provider.create_connections(2)
    capture = tmp_path / "departure.jsonl"
    env = make_env(connection=RecordingConnection(victim, capture), observation=preset)
    try:
        env.reset()
        survivor.reset()
        survivor.send_line("north")
        survivor.read_response()
        victim.send_line("west")
        assert b"Beaten track near cliff" in victim.read_response()[0]
        victim.send_line("kill Abbie")
        assert b"You attack Abbie" in victim.read_response()[0]
        if phase == "observation":
            env.unwrapped.act("look")

        world = victim._session.world
        old_session = victim._session
        for _ in range(40):
            provider.advance_worlds(1)
            if b"You have killed Aaron" in survivor._session.receive():
                break
        else:
            pytest.fail("Seeded combat did not kill Aaron within 40 ticks")
        assert world.is_alive() and not old_session.departed

        result = env.step("look") if phase == "action" else env.unwrapped.observe()
        obs, reward, terminated, truncated, info = result
        final_result = result
        final_message = b"You have been killed by Abbie"
        assert terminated and not truncated
        assert reward == -200 and obs["points"] == 0
        assert info["raw_bytes"].count(final_message) == 1
        assert b"sql,fes,fex,fei\r\n" not in info["raw_bytes"]
        if phase == "action":
            assert b"look\r\n" not in info["raw_bytes"]
        assert info["transport"]["sent_lines"] == (["look"] if phase == "observation" else [])
        assert "look" not in obs["text"].splitlines()
        assert b"look" not in info["render_bytes"].splitlines()
        assert old_session.departed and world.is_alive()
        # Unrelated uses of the retired handle must still fail loudly.
        with pytest.raises(RuntimeError, match=r"session-gone code=-7"):
            old_session.send("look", 5000)

        obs, reset_info = env.reset()
        assert obs["points"] == 200
        assert final_message not in reset_info["raw_bytes"]
        assert victim._session.world is world
        assert victim._session.generation > old_session.generation
        _, _, terminated, truncated, info = env.step("look")
        assert not terminated and not truncated
        assert final_message not in info["raw_bytes"]
    finally:
        env.close()
        provider.close()

    replay = ReplayConnection(capture)
    replay_env = make_env(connection=replay, observation=preset)
    try:
        replay_env.reset()
        if phase == "observation":
            replay_env.unwrapped.act("look")
            replay_result = replay_env.unwrapped.observe()
        else:
            replay_result = replay_env.step("look")
        replay_obs, replay_reward, replay_terminated, replay_truncated, replay_info = replay_result
        assert replay_obs["text"] == final_result[0]["text"]
        assert (replay_reward, replay_terminated, replay_truncated) == final_result[1:4]
        for key in ("raw_bytes", "render_bytes"):
            assert replay_info[key] == final_result[4][key]
        assert replay_info["transport"]["sent_lines"] == final_result[4]["transport"]["sent_lines"]
        replay_env.reset()
        replay_env.step("look")
        replay.assert_exhausted()
    finally:
        replay_env.close()
