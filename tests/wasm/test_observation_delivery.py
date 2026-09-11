"""Initial and trailing game output survives coordinated observation boundaries."""

import json
from contextlib import closing

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from mudgym import make_env, make_parallel_env, make_vector_env
from mudgym.connections.recording import RecordingProvider, ReplayProvider
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.envs.fields import SuperQuickLookField


@pytest.mark.parametrize("preset", ["text", "bytes"])
def test_single_slot_vector_reset_retains_the_scalar_initial_observation(wasm_runtime, preset):
    with make_env(connection="wasm", observation=preset, render_mode="ansi") as scalar:
        expected, expected_info = scalar.reset(seed=123)
    with closing(
        make_vector_env(1, provider=WasmtimeProvider(runtime=wasm_runtime), observation=preset, render_mode="ansi")
    ) as vector:
        actual, info = vector.reset(seed=123)
        assert actual["text"][0] == expected["text"]
        assert info["raw_bytes"][0] == expected_info["raw_bytes"]
        assert info["render_bytes"][0] == expected_info["render_bytes"]
        assert vector.render()[0] == expected_info["render_bytes"].decode("latin-1")
        if preset == "bytes":
            np.testing.assert_array_equal(actual["raw_bytes"][0], expected["raw_bytes"])
        observation, _, _, _, details = vector.envs[0].observe()
        assert observation["text"] == "" and details["raw_bytes"] == b""


@pytest.mark.parametrize("mode", ["vector", "parallel"])
@pytest.mark.parametrize("preset", ["text", "bytes", "parsed"])
def test_shared_reset_keeps_each_arrival_and_the_final_peer_state(wasm_runtime, mode, preset, tmp_path):
    factory = make_vector_env if mode == "vector" else make_parallel_env

    def capture_path(index):
        return tmp_path / f"player{index}.jsonl"

    provider = RecordingProvider(WasmtimeProvider(runtime=wasm_runtime, worlds=1), capture_path)
    with closing(factory(2, provider=provider, observation=preset, render_mode="ansi")) as environment:
        # With all players prepared first, seed 10 places Ada and Alba in the same room.
        observations, infos = environment.reset(seed=10)
        children = environment.envs if mode == "vector" else list(environment.envs.values())
        for index, child in enumerate(children):
            observation = (
                {key: value[index] for key, value in observations.items()}
                if mode == "vector"
                else observations[f"player_{index}"]
            )
            raw = infos["raw_bytes"][index] if mode == "vector" else infos[f"player_{index}"]["raw_bytes"]
            rendered = infos["render_bytes"][index] if mode == "vector" else infos[f"player_{index}"]["render_bytes"]
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

        actions = ["dance", "dance"] if mode == "vector" else {"player_0": "dance", "player_1": "dance"}
        following, _, terminated, truncated, _ = environment.step(actions)
        if mode == "vector":
            assert not terminated.any() and not truncated.any()
            texts = following["text"]
        else:
            assert not any(terminated.values()) and not any(truncated.values())
            texts = [observation["text"] for observation in following.values()]
        assert all("Badly-paved road" not in text for text in texts)


@pytest.mark.parametrize(("reset_kind", "seed"), [("masked", 5), ("next_step", 1)])
@pytest.mark.parametrize("preset", ["text", "bytes"])
def test_partial_resets_deliver_the_new_episode_output_once(wasm_runtime, reset_kind, seed, preset):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1)
    with closing(
        make_vector_env(3, provider=provider, observation=preset, autoreset_mode=AutoresetMode.NEXT_STEP)
    ) as vector:
        # These seeds make Alba's relogin visible to the unselected player.
        vector.reset(seed=seed)
        world = provider._ordered_worlds[0]
        sessions = [child.session.connection._session for child in vector.envs]
        cached = vector.observations[2]
        if reset_kind == "next_step":
            _, _, terminated, _, _ = vector.step(["quit", "quit", "look"])
            assert terminated.tolist() == [True, True, False]
            observations, rewards, terminated, truncated, infos = vector.step(["ignored", "ignored", "look"])
            assert rewards[:2].tolist() == [0, 0]
            assert not terminated.any() and not truncated.any()
            assert b"has just arrived" not in infos["raw_bytes"][2]
            assert world.current_tick() == 2
        else:
            observations, infos = vector.reset(options={"reset_mask": np.array([True, True, False])})
            assert vector.observations[2] is cached
            assert world.current_tick() == 0
        assert provider._ordered_worlds[0] is world
        assert vector.envs[2].session.connection._session is sessions[2]
        for index in (0, 1):
            assert vector.envs[index].session.connection._session is not sessions[index]
            assert observations["text"][index]
            assert infos["step"][index] == 0
            assert vector.envs[index].observe()[4]["raw_bytes"] == b""
        _, _, terminated, truncated, later_info = vector.envs[2].observe()
        assert not terminated and not truncated
        assert later_info["raw_bytes"].count(b"Alba the protector has just arrived") == 1


def test_failed_partial_entry_can_retry_without_resetting_the_unselected_player(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1)
    with closing(make_vector_env(3, provider=provider, observation="text")) as vector:
        vector.reset(seed=1)
        world = provider._ordered_worlds[0]
        survivor = vector.envs[2].session.connection._session
        cached = vector.observations[2]
        # Entering during setup makes the later exit command lack the required narration.
        vector.envs[1].tearoom_commands = "north"
        mask = {"reset_mask": np.array([True, True, False])}
        with pytest.raises(ValueError, match="tearoom exit marker") as raised:
            vector.reset(options=mask)
        assert any("entry" in note and "child 1" in note for note in raised.value.__notes__)
        assert vector.observations[2] is cached
        assert provider._ordered_worlds[0] is world
        assert vector.envs[2].session.connection._session is survivor
        with pytest.raises(RuntimeError, match="reset"):
            vector.step(["look", "look", "look"])

        vector.envs[1].tearoom_commands = None
        vector.reset(options=mask)
        assert provider._ordered_worlds[0] is world
        assert vector.envs[2].session.connection._session is survivor
        _, _, terminated, truncated, _ = vector.step(["look", "look", "look"])
        assert not terminated.any() and not truncated.any()


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


@pytest.mark.parametrize("mode", ["vector", "parallel"])
@pytest.mark.parametrize("preset", ["text", "bytes", "parsed"])
def test_recording_replays_complete_coordinated_resets(wasm_runtime, mode, preset, tmp_path):
    factory = make_vector_env if mode == "vector" else make_parallel_env

    def capture_path(index):
        return tmp_path / f"player{index}.jsonl"

    provider = RecordingProvider(WasmtimeProvider(runtime=wasm_runtime, worlds=1), capture_path)
    expected = []
    action = ["look", "look"] if mode == "vector" else {"player_0": "look", "player_1": "look"}
    with closing(factory(2, provider=provider, observation=preset)) as recorded:
        for seed in (3, 123):
            expected.append((recorded.reset(seed=seed), recorded.step(action)))
    with closing(factory(2, provider=ReplayProvider(capture_path), observation=preset)) as replay:
        for seed, (initial, following) in zip((3, 123), expected, strict=True):
            for actual, recorded in ((replay.reset(seed=seed), initial), (replay.step(action), following)):
                np.testing.assert_equal(actual[:-1], recorded[:-1])
                actual_infos = [actual[-1]] if mode == "vector" else actual[-1].values()
                recorded_infos = [recorded[-1]] if mode == "vector" else recorded[-1].values()
                for actual_info, recorded_info in zip(actual_infos, recorded_infos, strict=True):
                    np.testing.assert_equal(
                        {key: value for key, value in actual_info.items() if key != "transport"},
                        {key: value for key, value in recorded_info.items() if key != "transport"},
                    )
                    for key in ("sent_lines", "rejected", "marker_arrived", "incomplete"):
                        np.testing.assert_equal(actual_info["transport"][key], recorded_info["transport"][key])
        children = replay.envs if mode == "vector" else replay.envs.values()
        for child in children:
            child.session.connection.assert_exhausted()


@pytest.mark.parametrize("reset_kind", ["full", "masked", "next_step"])
def test_vector_human_reset_renders_each_initial_payload_once(wasm_runtime, capsys, reset_kind):
    with closing(
        make_vector_env(
            2,
            provider=WasmtimeProvider(runtime=wasm_runtime, worlds=1),
            observation="text",
            render_mode="human",
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )
    ) as vector:
        _, infos = vector.reset(seed=3)
        assert capsys.readouterr().out == b"".join(infos["render_bytes"]).decode("latin-1")

        _, _, _, _, infos = vector.step(["look", "look"])
        assert capsys.readouterr().out == b"".join(infos["render_bytes"]).decode("latin-1")

        if reset_kind == "full":
            _, infos = vector.reset(seed=3)
            rendered_indices = [0, 1]
        elif reset_kind == "masked":
            _, infos = vector.reset(options={"reset_mask": np.array([False, True])})
            rendered_indices = [1]
        else:
            vector.step(["mgquit", "look"])
            capsys.readouterr()
            _, _, _, _, infos = vector.step(["ignored", "look"])
            rendered_indices = [1, 0]
        assert capsys.readouterr().out == b"".join(infos["render_bytes"][index] for index in rendered_indices).decode(
            "latin-1"
        )

        expected_frames = "".join(child.render() for child in vector.envs)
        assert vector.render() == (None, None)
        assert capsys.readouterr().out == expected_frames
