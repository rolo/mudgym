import json
import re

import numpy as np
import pytest

from mudgym.connections import recording as recording_module
from mudgym.connections.errors import ConnectionClosedError
from mudgym.connections.recording import (
    CAPTURE_FORMAT,
    CAPTURE_VERSION,
    RecordingConnection,
    RecordingProvider,
    ReplayConnection,
    ReplayMismatchError,
    ReplayProvider,
)
from mudgym.envs.factory import make_env
from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.session import MudSession
from tests.scripted import ScriptedConnection, ScriptedProvider

END_OF_TURN_MARKER = FEInventoryField.end_of_turn_marker
OBSERVATION_LINE = "sql,fes,fex,fei"


def send_and_read(connection, lines):
    for line in lines:
        connection.send_line(line)
    return connection.read_response(END_OF_TURN_MARKER)


def drive_conversation(connection):
    """Run a small fixed connection transcript and return each response outcome."""
    connection.reset()
    results = [send_and_read(connection, ["look", OBSERVATION_LINE])[:3]]
    results.append(send_and_read(connection, ["say hello", OBSERVATION_LINE])[:3])
    return results


def test_connection_capture_round_trips_every_byte_value(tmp_path):
    path = tmp_path / "bytes.jsonl"
    response = (bytes(range(256)), False, False, {"rejected": False, "marker_arrived": True})
    recording = RecordingConnection(ScriptedConnection(responses={"look": response}), path, {"purpose": "test"})

    recording.reset()
    recorded_result = send_and_read(recording, ["look"])
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    replayed_result = send_and_read(replay, ["look"])
    replay.assert_exhausted()

    assert replay.header["format"] == CAPTURE_FORMAT
    assert replay.header["version"] == CAPTURE_VERSION
    assert replay.header["purpose"] == "test"
    assert replayed_result[:3] == recorded_result[:3]
    assert replayed_result[0] == bytes(range(256))
    # normal json.dumps keeps awkward C1 controls out of the JSONL itself
    assert "\\u0085" in path.read_text(encoding="utf-8")


def test_capture_version_deliberately_rejects_v2(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"format":"mudgym-session-capture","version":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=repr(CAPTURE_FORMAT)):
        ReplayConnection(path)


def test_recorded_connection_transcript_replays_identically(tmp_path):
    path = tmp_path / "capture.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    live_results = drive_conversation(recording)
    recording.close()

    replay = ReplayConnection(path)
    assert replay.header["connection"] == "ScriptedConnection"
    assert drive_conversation(replay) == live_results
    replay.assert_exhausted()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record.get("call") for record in records[1:]] == [
        "reset",
        "send_line",
        "send_line",
        "read_response",
        "send_line",
        "send_line",
        "read_response",
    ]


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
    replay.assert_exhausted()

    assert debug_info["rejected"] is True
    assert debug_info["marker_arrived"] is True


def test_replay_verifies_each_sent_line_immediately(tmp_path):
    path = tmp_path / "calls.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    drive_conversation(recording)
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    with pytest.raises(ReplayMismatchError, match="sent 'dance'.*recorded 'look'"):
        replay.send_line("dance")


def test_replay_rejects_read_response_before_recorded_send_lines(tmp_path):
    path = tmp_path / "calls.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    drive_conversation(recording)
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    with pytest.raises(ReplayMismatchError, match="expected 'read_response'.*has 'send_line'"):
        replay.read_response(END_OF_TURN_MARKER)


def test_replay_rejects_exhausted_use_and_unconsumed_calls(tmp_path):
    path = tmp_path / "calls.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    drive_conversation(recording)
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    with pytest.raises(ReplayMismatchError, match="unconsumed calls"):
        replay.assert_exhausted()

    send_and_read(replay, ["look", OBSERVATION_LINE])
    send_and_read(replay, ["say hello", OBSERVATION_LINE])
    replay.assert_exhausted()
    with pytest.raises(ReplayMismatchError, match="exhausted"):
        replay.send_line("look")


def make_session(connection) -> MudSession:
    return MudSession(
        connection,
        observation_line=OBSERVATION_LINE,
        end_of_turn_marker=END_OF_TURN_MARKER,
    )


def test_recorded_connection_closed_send_replays_before_action_response_is_drained(tmp_path):
    path = tmp_path / "closed-send.jsonl"
    response = (
        b"buffered death output",
        True,
        False,
        {"rejected": False, "marker_arrived": False},
    )
    live_connection = ScriptedConnection(
        responses={"quit": response},
        send_errors={OBSERVATION_LINE: ConnectionClosedError("closed after action")},
    )
    recording = RecordingConnection(live_connection, path)
    live_session = make_session(recording)

    live_session.send("quit")
    live_result = live_session.receive()
    recording.close()

    replay = ReplayConnection(path)
    replay_session = make_session(replay)
    replay_session.send("quit")
    replay_result = replay_session.receive()
    replay.assert_exhausted()

    assert live_result[:3] == (b"buffered death output", True, False)
    assert replay_result[:3] == live_result[:3]
    assert live_result[3]["wire_lines"] == ["quit"]
    assert replay_result[3]["wire_lines"] == ["quit"]
    assert live_connection.invalidated is True


def test_replay_requires_recorded_invalidation(tmp_path):
    path = tmp_path / "invalidate.jsonl"
    recording = RecordingConnection(ScriptedConnection(), path)
    recording.reset()
    recording.invalidate()
    recording.close()

    replay = ReplayConnection(path)
    replay.reset()
    with pytest.raises(ReplayMismatchError, match="unconsumed calls"):
        replay.assert_exhausted()
    replay.invalidate()
    replay.assert_exhausted()


def test_end_of_turn_marker_reaches_wrapped_connection(tmp_path):
    inner = ScriptedConnection()
    recording = RecordingConnection(inner, tmp_path / "capture.jsonl")
    marker = re.compile(rb"custom-marker")
    recording.reset()
    recording.send_line("look")
    recording.read_response(marker)
    assert inner.read_markers == [marker]
    recording.close()


def test_env_over_replay_reproduces_the_recorded_episode(tmp_path):
    path = tmp_path / "episode.jsonl"

    env = make_env(observation="parsed", connection=RecordingConnection(ScriptedConnection(), path))
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step("look")
    env.close()

    replay = ReplayConnection(path)
    replay_env = make_env(observation="parsed", connection=replay)
    replay_observation, replay_info = replay_env.reset()
    replay_observation, replay_reward, replay_terminated, replay_truncated, replay_info = replay_env.step("look")
    replay_env.close()
    replay.assert_exhausted()

    assert replay_reward == reward
    assert replay_terminated == terminated
    assert replay_truncated == truncated
    assert set(replay_observation) == set(observation)
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(replay_observation[key], value), key
        else:
            assert replay_observation[key] == value, key


def test_providers_record_and_replay_per_index(tmp_path):
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
    replay_connections = replay_provider.create_connections(2)
    for index, connection in enumerate(replay_connections):
        assert drive_conversation(connection) == live_results[index]
        connection.assert_exhausted()


def test_recording_provider_closes_the_underlying_batch_if_wrapping_fails(tmp_path, monkeypatch):
    real_recording_connection = RecordingConnection

    def failing_recording_connection(connection, path, metadata):
        if path.name == "agent1.jsonl":
            raise RuntimeError("wrapping failed")
        return real_recording_connection(connection, path, metadata)

    monkeypatch.setattr(recording_module, "RecordingConnection", failing_recording_connection)
    inner = ScriptedProvider()
    provider = RecordingProvider(inner, lambda index: tmp_path / f"agent{index}.jsonl")

    with pytest.raises(RuntimeError, match="wrapping failed"):
        provider.create_connections(3)

    assert all(connection.closed for connection in inner.connections)
    assert inner.closed is True
