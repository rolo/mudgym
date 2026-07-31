"""The connection command API rejects empty batches before any transport activity."""

import pytest

from mudgym.connections.connection import MudConnection


class RecordingStateMachine:
    """Just the surface MudConnection.send_command dispatches to."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def send_command(self, command):
        self.batches.append(list(command))
        return b"", False, False, {}


def test_an_empty_batch_raises_without_touching_the_transport():
    connection = MudConnection()
    connection.sm = RecordingStateMachine()

    with pytest.raises(ValueError, match="at least one command line"):
        connection.send_command([])

    assert connection.sm.batches == []


def test_a_single_string_dispatches_as_a_one_line_batch():
    connection = MudConnection()
    connection.sm = RecordingStateMachine()

    connection.send_command("look")

    assert connection.sm.batches == [["look"]]


def test_a_one_element_sequence_dispatches_unchanged():
    connection = MudConnection()
    connection.sm = RecordingStateMachine()

    connection.send_command(["look,sql,fes,fex,fei"])

    assert connection.sm.batches == [["look,sql,fes,fex,fei"]]


def test_a_multi_line_speech_batch_dispatches_unchanged():
    connection = MudConnection()
    connection.sm = RecordingStateMachine()

    connection.send_command(["say hello", "sql,fes,fex,fei"])

    assert connection.sm.batches == [["say hello", "sql,fes,fex,fei"]]
