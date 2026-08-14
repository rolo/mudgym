import pytest

from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.session import MudSession
from tests.scripted import ScriptedConnection

OBSERVATION_LINE = "sql,fes,fex,fei"


def make_session(connection: ScriptedConnection | None = None) -> MudSession:
    return MudSession(
        connection or ScriptedConnection(),
        observation_line=OBSERVATION_LINE,
        end_of_turn_marker=FEInventoryField.end_of_turn_marker,
    )


@pytest.mark.parametrize("command", ["look", "say hello", '"hello', "get sword,say hi"])
def test_player_command_and_observation_commands_are_always_separate_wire_lines(command):
    session = make_session()

    session.send(command)
    assert session.pending_command == command
    assert session.connection.pending_lines == [command]

    _, _, _, info = session.receive()

    assert info["wire_lines"] == [command, "sql,fes,fex,fei"]
    assert session.pending_command is None


def test_command_is_send_followed_by_receive():
    session = make_session()

    result = session.command("look")

    assert result[3]["wire_lines"] == ["look", "sql,fes,fex,fei"]


def test_receive_without_a_pending_command_refreshes_observation_only():
    session = make_session()

    _, _, _, info = session.receive()

    assert info["wire_lines"] == ["sql,fes,fex,fei"]


def test_send_rejects_a_second_command_while_one_is_pending():
    session = make_session()
    session.send("look")

    with pytest.raises(RuntimeError, match="still waiting"):
        session.send("dance")


def test_reset_uses_the_same_split_protocol_for_the_persona_probe():
    connection = ScriptedConnection()
    session = make_session(connection)

    session.reset()

    assert session.persona == "Alexander"
    assert connection.sent_lines == [["qs", "sql,fes,fex,fei"]]
