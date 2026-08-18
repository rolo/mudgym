"""The Option-menu answer gate: redrawn prompts must not be answered twice.

The menu redraws its Option prompt around interstitials (the MAIL screens after mgquit) before
consuming an answer already sent. The game queues typed input, so a second answer becomes a stray
line that the persona dialogue swallows later, desynchronising the relogin.
"""

import pexpect
import pytest

from mudgym.connections.prompts import EXPECT_LIST, State
from mudgym.connections.transitions import menu_answer_was_echoed, send_db_slot

# the wire bytes the menu emits between answering an Option prompt and its redraw, as captured
# after mgquit
MAIL_INTERSTITIAL = b" mgquit\r\n\x1b[f\x1b[2J[MAIL unavailable]\r\n[MAIL exit]\r\n"


class FakeStateMachine:
    """Just the surface the transition actions touch."""

    def __init__(self, buffer: bytes = b"", db_slot: int = 0):
        self.default_db_slot = db_slot
        self.pending_menu_answer: bytes | None = None
        self.output_since_menu_answer = b""
        self.menu_answer_echoed = False
        self.state = State.OPTION
        self.sent: list[str] = []
        self._buffer = buffer

    def get_buffer(self) -> bytes:
        return self._buffer

    def send(self, data: str) -> None:
        self.sent.append(data)


def test_a_fresh_option_prompt_is_answered():
    state_machine = FakeStateMachine()

    send_db_slot(state_machine)

    assert state_machine.sent == [b"p0"]
    assert state_machine.pending_menu_answer == b"p0"


def test_the_mail_interstitial_matches_no_recognised_prompt():
    """The gate holds through the MAIL screens because no prompt fires between the two Options.

    The state machine clears pending_menu_answer whenever it matches a non-Option prompt, so the
    gate rests on this invariant: nothing the menu emits between answering an Option prompt and
    its redraw matches EXPECT_LIST (the ``[MAIL unavailable]`` status line is deliberately absent - see
    test_termination). If a pattern ever starts matching this capture, the pending answer would
    be cleared mid-interstitial and the redraw answered a second time.
    """
    for pattern in EXPECT_LIST:
        if pattern in (pexpect.EOF, pexpect.TIMEOUT):
            continue
        if isinstance(pattern, bytes):
            assert pattern not in MAIL_INTERSTITIAL
        else:
            assert pattern.search(MAIL_INTERSTITIAL) is None


def test_a_redrawn_option_prompt_is_not_answered_again():
    # captured shape: mgquit echo and the MAIL screens arrive between the two prompts, but the
    # earlier p0 answer has not been echoed back, so it is still queued
    state_machine = FakeStateMachine(buffer=MAIL_INTERSTITIAL)
    send_db_slot(state_machine)
    state_machine.sent.clear()

    send_db_slot(state_machine)

    assert state_machine.sent == []
    assert state_machine.pending_menu_answer == b"p0"


def test_a_reask_after_the_echo_is_answered_again():
    # captured shape: during boot the menu re-asks after rejecting the answer, and the echo shows
    # the earlier answer was consumed
    state_machine = FakeStateMachine(buffer=b" p0\r\nDatabase 0 is not initialised.\r\n")
    send_db_slot(state_machine)
    state_machine.sent.clear()

    send_db_slot(state_machine)

    assert state_machine.sent == [b"p0"]


def test_the_configured_slot_is_used():
    state_machine = FakeStateMachine(db_slot=3)

    send_db_slot(state_machine)

    assert state_machine.sent == [b"p3"]
    assert state_machine.pending_menu_answer == b"p3"


# the two shapes the menu echoes an answer back in, both as captured
ECHO_FORMS = [b"\r\np0\r\n", b" p0\r\nDatabase 0 is not initialised.\r\n"]

# text carrying the answer's bytes without being an echo of it
NON_ECHO_FORMS = [
    b"An incidental status string containing p0 but not an echo.\r\n",
    MAIL_INTERSTITIAL,
    b"p0no\r\n",
]


@pytest.mark.parametrize("output", ECHO_FORMS)
def test_an_echoed_answer_is_recognised(output):
    assert menu_answer_was_echoed(output, b"p0")


@pytest.mark.parametrize("output", NON_ECHO_FORMS)
def test_the_answers_bytes_in_passing_are_not_an_echo(output):
    assert not menu_answer_was_echoed(output, b"p0")


def test_a_colour_wrapped_echo_is_recognised():
    assert menu_answer_was_echoed(b"\x1b[0;37;40mp0\x1b[1;37;40m\r\n", b"p0")


def test_incidental_output_does_not_reopen_the_gate():
    state_machine = FakeStateMachine()
    send_db_slot(state_machine)
    state_machine.sent.clear()
    state_machine._buffer = b"An incidental status string containing p0 but not an echo.\r\n"

    send_db_slot(state_machine)

    assert state_machine.sent == []
