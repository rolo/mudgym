import re

import numpy as np
import pytest

from mudgym.connections import registry
from mudgym.connections.recording import (
    CaptureWriter,
    RecordingConnection,
    RecordingProvider,
    ReplayConnection,
    ReplayProvider,
    StaleCaptureError,
    read_capture,
)
from mudgym.envs.factory import make_env
from tests.scripted import ScriptedConnection


def drive_conversation(connection):
    """Run a small fixed conversation and return the (bytes, terminated, incomplete) of each step."""
    connection.reset()
    results = [connection.send_command("look,sql,fes,fex,fei")[:3]]
    results.append(connection.send_command(["say hello", "sql,fes,fex,fei"])[:3])
    return results


def test_capture_round_trips_every_byte_value(tmp_path):
    path = tmp_path / "bytes.jsonl"
    writer = CaptureWriter(path, {"purpose": "test"})
    writer.record_reset()
    writer.record_step(["look"], bytes(range(256)), terminated=False, incomplete=True)
    writer.close()

    header, events = read_capture(path)
    assert header["purpose"] == "test"
    assert events[0] == {"event": "reset"}
    assert events[1]["lines"] == ["look"]
    assert events[1]["raw_bytes"] == bytes(range(256))
    assert events[1]["incomplete"] is True
    assert "rejected" not in events[1]


def test_recorded_conversation_replays_identically(tmp_path):
    path = tmp_path / "capture.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    live_results = drive_conversation(recording)
    recording.close()

    replay = ReplayConnection(path)
    assert replay.header["connection"] == "ScriptedConnection"
    assert drive_conversation(replay) == live_results
    assert replay.remaining_events() == 0


def test_replay_preserves_rejected_outcome_and_completed_marker(tmp_path):
    path = tmp_path / "capture.jsonl"
    response = (b"rejected\r\n", False, False, {"rejected": True, "marker_arrived": True})
    recording = RecordingConnection(ScriptedConnection(responses={"xyzzyfrobnicate": response}), path)
    recording.reset()
    recording.send_command("xyzzyfrobnicate")
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    _, _, _, debug_info = replay.send_command("xyzzyfrobnicate")

    assert debug_info["rejected"] is True
    assert debug_info["marker_arrived"] is True


def test_replay_rejects_diverged_wire_lines(tmp_path):
    path = tmp_path / "capture.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    drive_conversation(recording)
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    with pytest.raises(StaleCaptureError, match="diverged"):
        replay.send_command("dance,sql,fes,fex,fei")


def test_replay_rejects_out_of_order_and_exhausted_use(tmp_path):
    path = tmp_path / "capture.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    drive_conversation(recording)
    recording.close()

    with pytest.raises(StaleCaptureError, match="out of order"):
        ReplayConnection(path).send_command("look,sql,fes,fex,fei")

    replay = ReplayConnection(path)
    drive_conversation(replay)
    with pytest.raises(StaleCaptureError, match="exhausted"):
        replay.send_command("look,sql,fes,fex,fei")


def test_end_of_turn_marker_reaches_wrapped_connection(tmp_path):
    inner = ScriptedConnection()
    recording = RecordingConnection(inner, tmp_path / "capture.jsonl")
    marker = re.compile(rb"custom-marker")
    recording.end_of_turn_marker = marker
    assert inner.end_of_turn_marker is marker
    assert recording.end_of_turn_marker is marker
    recording.close()


def test_env_over_replay_reproduces_the_recorded_episode(tmp_path):
    path = tmp_path / "episode.jsonl"

    env = make_env(observation="parsed", connection=RecordingConnection(ScriptedConnection(), path))
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step("look")
    env.close()

    replay_env = make_env(observation="parsed", connection=ReplayConnection(path))
    replay_observation, replay_info = replay_env.reset()
    replay_observation, replay_reward, replay_terminated, replay_truncated, replay_info = replay_env.step("look")
    replay_env.close()

    assert replay_reward == reward
    assert replay_terminated == terminated
    assert replay_truncated == truncated
    assert replay_info["render_bytes"] == info["render_bytes"]
    assert set(replay_observation) == set(observation)
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(replay_observation[key], value), key
        else:
            assert replay_observation[key] == value, key


def test_factory_resolves_registry_default_connection_at_call_time(monkeypatch):
    monkeypatch.setattr(registry, "default_connection", ScriptedConnection)
    env = make_env(observation="parsed")
    observation, info = env.reset()
    env.close()
    assert observation["room_name"]


def test_providers_record_and_replay_per_index(tmp_path):
    class ScriptedProvider:
        def __init__(self):
            self.closed = False

        def create_connection(self, env_index):
            return ScriptedConnection()

        def close(self):
            self.closed = True

    def capture_path(env_index):
        return tmp_path / f"agent{env_index}.jsonl"

    inner = ScriptedProvider()
    provider = RecordingProvider(inner, capture_path)
    live_results = [drive_conversation(provider.create_connection(index)) for index in range(2)]
    provider.close()
    assert inner.closed

    replay_provider = ReplayProvider(capture_path)
    for index in range(2):
        assert drive_conversation(replay_provider.create_connection(index)) == live_results[index]
    assert [connection.remaining_events() for connection in replay_provider.connections] == [0, 0]
