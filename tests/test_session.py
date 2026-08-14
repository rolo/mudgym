import json

import pytest

from mudgym.connections.recording import RecordingConnection
from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.session import MudSession
from tests.scripted import ScriptedConnection

OBSERVATION_LINE = "sql,fes,fex,fei"


def make_session(connection=None) -> MudSession:
    return MudSession(
        connection or ScriptedConnection(),
        observation_line=OBSERVATION_LINE,
        end_of_turn_marker=FEInventoryField.end_of_turn_marker,
    )


def recorded_calls(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return records[1:]


@pytest.mark.parametrize("command", ["look", "say hello", '"hello', "get sword,say hi"])
def test_player_command_and_observation_commands_are_always_separate_wire_calls(command, tmp_path):
    capture_path = tmp_path / "session.jsonl"
    connection = RecordingConnection(ScriptedConnection(), capture_path)
    session = make_session(connection)

    session.send(command)
    assert session.pending_command == command

    session.receive()
    connection.close()

    calls = recorded_calls(capture_path)
    assert [call["call"] for call in calls] == ["send_line", "send_line", "read_response"]
    assert [call["line"] for call in calls[:2]] == [command, OBSERVATION_LINE]
    assert calls[2]["terminated"] is False
    assert calls[2]["incomplete"] is False
    assert session.pending_command is None


def test_command_is_send_followed_by_receive():
    session = make_session()

    result = session.command("look")

    assert result[3]["wire_lines"] == ["look", "sql,fes,fex,fei"]


def test_receive_without_a_pending_command_refreshes_observation_only(tmp_path):
    capture_path = tmp_path / "session.jsonl"
    connection = RecordingConnection(ScriptedConnection(), capture_path)
    session = make_session(connection)

    session.receive()
    connection.close()

    calls = recorded_calls(capture_path)
    assert [call["call"] for call in calls] == ["send_line", "read_response"]
    assert calls[0]["line"] == OBSERVATION_LINE


def test_send_rejects_a_second_command_while_one_is_pending():
    session = make_session()
    session.send("look")

    with pytest.raises(RuntimeError, match="still waiting"):
        session.send("dance")


def test_reset_uses_the_same_split_protocol_for_the_persona_probe(tmp_path):
    capture_path = tmp_path / "session.jsonl"
    connection = RecordingConnection(ScriptedConnection(), capture_path)
    session = make_session(connection)

    session.reset()
    connection.close()

    assert session.persona == "Alexander"
    calls = recorded_calls(capture_path)
    assert [call["call"] for call in calls] == ["reset", "send_line", "send_line", "read_response"]
    assert [call["line"] for call in calls if call["call"] == "send_line"] == ["qs", OBSERVATION_LINE]


@pytest.mark.parametrize(("terminated", "incomplete"), [(True, False), (False, True)])
def test_terminal_or_incomplete_response_invalidates_the_connection(terminated, incomplete):
    response = b"response", terminated, incomplete, {"marker_arrived": not incomplete}
    connection = ScriptedConnection(responses={"look": response})
    session = make_session(connection)

    result = session.command("look")

    assert result[1:3] == (terminated, incomplete)
    assert connection.invalidated is True
