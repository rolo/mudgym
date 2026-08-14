from collections.abc import Sequence
from typing import Any

import pytest

from mudgym.connections.connection import MudConnection
from mudgym.connections.errors import ConnectionClosedError
from mudgym.connections.prompts import INVALID_COMMAND_PROMPTS, State
from mudgym.connections.state_machine import ConnectionState
from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.session import MudSession

END_OF_TURN_MARKER = FEInventoryField.end_of_turn_marker


class DiesBeforeObservationConnection(MudConnection):
    def __init__(self):
        super().__init__()
        self.sent_lines: list[str] = []
        self.read_lines: list[str] | None = None
        self.invalidated = False

    def send_line(self, line: str) -> None:
        self.sent_lines.append(line)
        if len(self.sent_lines) == 2:
            raise ConnectionClosedError("closed after action")

    def read_response(self, lines: Sequence[str], end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        self.read_lines = list(lines)
        assert end_of_turn_marker is END_OF_TURN_MARKER
        return b"buffered death output", True, False, {}

    def invalidate(self) -> None:
        self.invalidated = True


class ResultConnection(MudConnection):
    def __init__(
        self,
        *,
        terminated: bool,
        incomplete: bool,
        rejected: bool = False,
        marker_arrived: bool = False,
    ):
        super().__init__()
        self.terminated = terminated
        self.incomplete = incomplete
        self.rejected = rejected
        self.marker_arrived = marker_arrived
        self.invalidated = False

    def send_line(self, line: str) -> None:
        pass

    def read_response(self, lines: Sequence[str], end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        assert end_of_turn_marker is END_OF_TURN_MARKER
        return (
            b"response",
            self.terminated,
            self.incomplete,
            {
                "rejected": self.rejected,
                "marker_arrived": self.marker_arrived,
            },
        )

    def invalidate(self) -> None:
        self.invalidated = True


def make_session(connection: MudConnection) -> MudSession:
    return MudSession(
        connection,
        observation_line="sql,fes,fex,fei",
        end_of_turn_marker=END_OF_TURN_MARKER,
    )


def test_receive_drains_the_action_response_when_the_auto_send_finds_a_dead_connection():
    connection = DiesBeforeObservationConnection()
    session = make_session(connection)

    session.send("quit")
    raw_bytes, terminated, incomplete, info = session.receive()

    assert raw_bytes == b"buffered death output"
    assert terminated is True
    assert incomplete is False
    assert connection.read_lines == ["quit"]
    assert info["wire_lines"] == ["quit"]
    assert connection.invalidated is True


@pytest.mark.parametrize(("terminated", "incomplete"), [(True, False), (False, True)])
def test_terminal_or_incomplete_step_invalidates_the_connection(terminated, incomplete):
    connection = ResultConnection(terminated=terminated, incomplete=incomplete)
    session = make_session(connection)

    session.send("look")
    session.receive()

    assert connection.invalidated is True


@pytest.mark.parametrize("incomplete", [False, True])
def test_session_preserves_connection_read_completeness(incomplete):
    connection = ResultConnection(
        terminated=False,
        incomplete=incomplete,
        rejected=True,
        marker_arrived=not incomplete,
    )
    session = make_session(connection)

    session.send("invalid")
    _, _, result_incomplete, _ = session.receive()

    assert result_incomplete is incomplete
    assert connection.invalidated is incomplete


class ClosedChild:
    def isalive(self) -> bool:
        return False


class ChildClosingDuringSend:
    def isalive(self) -> bool:
        return True

    def send(self, data: bytes) -> None:
        raise OSError("child closed during write")


@pytest.mark.parametrize("child", [ClosedChild(), ChildClosingDuringSend()])
def test_state_machine_send_reports_a_specific_closed_connection_error(child):
    state_machine = object.__new__(ConnectionState)
    state_machine.child = child
    state_machine.state = State.GAME

    with pytest.raises(ConnectionClosedError):
        state_machine.send("look")


class DisposableStateMachine:
    def __init__(self):
        self.closed = False

    def isalive(self) -> bool:
        return False

    def close(self, *, force: bool = True) -> None:
        self.closed = True


class UncleanStateMachine(DisposableStateMachine):
    def isalive(self) -> bool:
        return True

    def quit(self) -> None:
        raise RuntimeError("cannot cleanly leave this state")


def test_invalidating_a_connection_discards_its_state_machine():
    connection = MudConnection()
    state_machine = DisposableStateMachine()
    connection.sm = state_machine

    connection.invalidate()

    assert state_machine.closed is True
    assert connection.sm is None


def test_cleanup_failure_does_not_escape_connection_invalidation():
    connection = MudConnection()
    state_machine = UncleanStateMachine()
    connection.sm = state_machine

    connection.invalidate()

    assert state_machine.closed is True
    assert connection.sm is None


def test_final_line_rejection_marks_the_read_window_incomplete():
    class Child:
        before = b""
        after = b""

    child = Child()
    state_machine = object.__new__(ConnectionState)
    state_machine.child = child
    state_machine.state = State.GAME
    calls = 0

    def expect(patterns, *, raise_on_eof_timeout):
        nonlocal calls
        if calls == 0:
            child.after = b"fei\r\n"
            index = 0
        else:
            child.after = b"I made no sense of that:\r\n*"
            index = patterns.index(INVALID_COMMAND_PROMPTS[0])
        calls += 1
        return index, None

    state_machine.expect = expect

    _, terminated, incomplete, debug_info = state_machine.read(["fei"], END_OF_TURN_MARKER)

    assert terminated is False
    assert incomplete is True
    assert debug_info["rejected"] is True
    assert debug_info["marker_arrived"] is False
