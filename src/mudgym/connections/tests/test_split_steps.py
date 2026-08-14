import socket
from types import SimpleNamespace

import pytest
from pexpect.fdpexpect import fdspawn

from mudgym.connections.connection import MudConnection
from mudgym.connections.errors import ConnectionClosedError
from mudgym.connections.prompts import INVALID_COMMAND_PROMPTS, Prompt, State
from mudgym.connections.state_machine import ConnectionState
from mudgym.envs.fields.feinventory import FEInventoryField

END_OF_TURN_MARKER = FEInventoryField.end_of_turn_marker
GAME_PROMPT = b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"


def test_connection_reads_only_lines_which_were_successfully_sent():
    sent_lines = []
    read_lines = []

    def send(line: str) -> None:
        if line == "sql,fes,fex,fei":
            raise ConnectionClosedError("closed after action")
        sent_lines.append(line)

    def read(lines, end_of_turn_marker):
        read_lines.append(list(lines))
        return b"buffered action response", True, False, {}

    connection = MudConnection()
    state_machine = SimpleNamespace(send=send, read=read)
    connection.sm = state_machine

    connection.send_line("quit")
    with pytest.raises(ConnectionClosedError):
        connection.send_line("sql,fes,fex,fei")

    result = connection.read_response(END_OF_TURN_MARKER)

    assert result[:3] == (b"buffered action response", True, False)
    assert read_lines == [["quit"]]

    with pytest.raises(RuntimeError, match="No command lines"):
        connection.read_response(END_OF_TURN_MARKER)


def close_during_send(data: bytes) -> None:
    raise OSError("child closed during write")


@pytest.mark.parametrize(
    "child",
    [
        SimpleNamespace(isalive=lambda: False),
        SimpleNamespace(isalive=lambda: True, send=close_during_send),
    ],
)
def test_state_machine_send_reports_a_specific_closed_connection_error(child):
    state_machine = object.__new__(ConnectionState)
    state_machine.child = child
    state_machine.state = State.GAME

    with pytest.raises(ConnectionClosedError):
        state_machine.send("look")


@pytest.mark.parametrize("quit_fails", [False, True])
def test_invalidating_a_connection_discards_its_state_machine_even_if_quit_fails(quit_fails):
    closed = []

    def quit() -> None:
        raise RuntimeError("cannot cleanly leave this state")

    def close(*, force: bool = True) -> None:
        closed.append(force)

    connection = MudConnection()
    state_machine = SimpleNamespace(
        isalive=lambda: quit_fails,
        quit=quit,
        close=close,
    )
    connection.sm = state_machine

    connection.invalidate()

    assert closed == [True]
    assert connection.sm is None


def test_final_line_rejection_waits_for_timeout_before_marking_the_read_window_incomplete():
    child = SimpleNamespace(before=b"", after=b"")
    state_machine = object.__new__(ConnectionState)
    state_machine.child = child
    state_machine.state = State.GAME
    calls = 0

    def expect(patterns, *, raise_on_eof_timeout):
        nonlocal calls
        if calls == 0:
            child.after = b"fei\r\n"
            index = 0
        elif calls == 1:
            child.after = b"I made no sense of that:\r\n*"
            index = patterns.index(INVALID_COMMAND_PROMPTS[0])
        else:
            child.after = Prompt.TIMEOUT.value
            index = patterns.index(Prompt.TIMEOUT.value)
        calls += 1
        prompt = Prompt.TIMEOUT if child.after is Prompt.TIMEOUT.value else None
        return index, prompt

    state_machine.expect = expect

    _, terminated, incomplete, debug_info = state_machine.read(["fei"], END_OF_TURN_MARKER)

    assert terminated is False
    assert incomplete is True
    assert debug_info["rejected"] is True
    assert debug_info["marker_arrived"] is False


def test_action_rejection_after_observation_echo_waits_for_marker():
    output = (
        b"Not updating persona.\r\n"
        b"fei\r\n"
        b'I made no sense of that: "not updating persona", but it\'s an excluding preposition.\r\n'
        + GAME_PROMPT
        + b"\x1b[0;37;40m========\r\ntea\r\n\x1b[1;37;40m"
        + GAME_PROMPT
    )
    connection_socket, game_socket = socket.socketpair()
    child = fdspawn(connection_socket.detach(), encoding=None)
    game_socket.sendall(GAME_PROMPT + output)
    state_machine = ConnectionState(child=child, initial_prompt=Prompt.GAME)
    state_machine.state = State.GAME
    try:
        raw_bytes, terminated, incomplete, debug_info = state_machine.read(
            ["Not updating persona.", "fei"],
            END_OF_TURN_MARKER,
        )
    finally:
        child.close()
        game_socket.close()

    assert b"========" in raw_bytes
    assert terminated is False
    assert incomplete is False
    assert debug_info["rejected"] is True
    assert debug_info["marker_arrived"] is True
