import re
from pathlib import Path

import numpy as np
import pytest

from mudgym.connections import recording as recording_module
from mudgym.connections import registry
from mudgym.connections.recording import (
    CAPTURE_VERSION,
    CaptureWriter,
    RecordingConnection,
    RecordingProvider,
    ReplayConnection,
    ReplayProvider,
    StaleCaptureError,
    read_capture,
)
from mudgym.envs.factory import make_env
from mudgym.envs.fields.feinventory import FEInventoryField
from tests.scripted import ScriptedConnection


def drive_conversation(connection):
    """Run a small fixed conversation and return the (bytes, terminated, incomplete) of each step."""
    connection.reset()
    results = [send_and_read(connection, ["look", "sql,fes,fex,fei"])[:3]]
    results.append(send_and_read(connection, ["say hello", "sql,fes,fex,fei"])[:3])
    return results


def send_and_read(connection, lines):
    for line in lines:
        connection.send_line(line)
    return connection.read_response(lines, FEInventoryField.end_of_turn_marker)


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
    assert events[1]["rejected"] is False
    assert events[1]["marker_arrived"] is False


def test_capture_version_invalidates_the_old_wire_contract(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"format":"mudgym-session-capture","version":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=f"expected {CAPTURE_VERSION}"):
        read_capture(path)


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
    send_and_read(recording, ["xyzzyfrobnicate"])
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    _, _, _, debug_info = send_and_read(replay, ["xyzzyfrobnicate"])

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
        send_and_read(replay, ["dance", "sql,fes,fex,fei"])


def test_replay_rejects_out_of_order_and_exhausted_use(tmp_path):
    path = tmp_path / "capture.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    drive_conversation(recording)
    recording.close()

    with pytest.raises(StaleCaptureError, match="out of order"):
        send_and_read(ReplayConnection(path), ["look", "sql,fes,fex,fei"])

    replay = ReplayConnection(path)
    drive_conversation(replay)
    with pytest.raises(StaleCaptureError, match="exhausted"):
        send_and_read(replay, ["look", "sql,fes,fex,fei"])


def test_end_of_turn_marker_reaches_wrapped_connection(tmp_path):
    inner = ScriptedConnection()
    recording = RecordingConnection(inner, tmp_path / "capture.jsonl")
    marker = re.compile(rb"custom-marker")
    recording.reset()
    recording.send_line("look")
    recording.read_response(["look"], marker)
    assert inner.read_markers == [marker]
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
            self.reset_seeds = []

        def create_connections(self, count):
            return [ScriptedConnection() for _ in range(count)]

        def reset(self, *, seed=None):
            self.reset_seeds.append(seed)

        def close(self):
            self.closed = True

    def capture_path(env_index):
        return tmp_path / f"agent{env_index}.jsonl"

    inner = ScriptedProvider()
    provider = RecordingProvider(inner, capture_path)
    provider.reset(seed=17)
    live_results = [drive_conversation(connection) for connection in provider.create_connections(2)]
    provider.close()
    assert inner.closed
    assert inner.reset_seeds == [17]

    replay_provider = ReplayProvider(capture_path)
    for index, connection in enumerate(replay_provider.create_connections(2)):
        assert drive_conversation(connection) == live_results[index]
    assert replay_provider.remaining_events() == [0, 0]


def test_recording_provider_closes_the_underlying_batch_if_wrapping_fails(tmp_path, monkeypatch):
    class TrackingProvider:
        def __init__(self):
            self.connections = []
            self.closed = False

        def create_connections(self, count):
            self.connections = [ScriptedConnection() for _ in range(count)]
            return list(self.connections)

        def reset(self, *, seed=None):
            pass

        def close(self):
            self.closed = True

    real_recording_connection = RecordingConnection

    def failing_recording_connection(connection, path, metadata):
        if path.name == "agent1.jsonl":
            raise RuntimeError("wrapping failed")
        return real_recording_connection(connection, path, metadata)

    monkeypatch.setattr(recording_module, "RecordingConnection", failing_recording_connection)
    inner = TrackingProvider()
    provider = RecordingProvider(inner, lambda index: tmp_path / f"agent{index}.jsonl")

    with pytest.raises(RuntimeError, match="wrapping failed"):
        provider.create_connections(3)

    assert all(connection.closed for connection in inner.connections)
    assert inner.closed is True


def test_replay_provider_closes_partial_batch_if_connection_creation_fails(monkeypatch):
    created_connections = []

    class TrackingReplayConnection:
        def __init__(self, path):
            if path.name == "agent1.jsonl":
                raise RuntimeError("replay creation failed")
            self.closed = False
            created_connections.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(recording_module, "ReplayConnection", TrackingReplayConnection)
    provider = ReplayProvider(lambda index: Path(f"agent{index}.jsonl"))

    with pytest.raises(RuntimeError, match="replay creation failed"):
        provider.create_connections(3)

    assert all(connection.closed for connection in created_connections)
    assert provider._connections == []
