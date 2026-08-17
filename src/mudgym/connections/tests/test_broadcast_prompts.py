"""Database broadcasts must not hijack the login dialogue."""

from collections import deque
from types import SimpleNamespace

import pytest

from mudgym.connections.prompts import Prompt, State
from mudgym.connections.state_machine import ConnectionState
from mudgym.featurizers.strings import encode_command_bytes

DIALOGUE_STATES = (
    State.INITIAL,
    State.LOGIN,
    State.OPTION,
    State.PERSONA_SELECT,
    State.PERSONA_NAME_INPUT,
    State.PERSONA_SEX_INPUT,
    State.RESETTING,
)

# everywhere the broadcast means nothing, which is everywhere but RESETTING
NON_RESETTING_STATES = tuple(state for state in State if state is not State.RESETTING)

# as captured from an 8 slot container
OTHER_SLOTS_BROADCAST = b"+- Database 7 has finished initialising -+"
OUR_SLOTS_BROADCAST = b"+- Database 0 has finished initialising -+"


class DialogueStateMachine(ConnectionState):
    """Records sends instead of writing to a pty, so the real transition table drives the test."""

    def __init__(self, state: State, db_slot: int = 0, matched: bytes = b"", buffer: bytes = b""):
        self.state = state
        self.sent: list[bytes] = []
        self.history: list[State] = []
        self.pending_menu_answer: str | None = None
        self.output_since_menu_answer = b""
        self.default_db_slot = db_slot
        self.child = SimpleNamespace(after=matched)
        self._buffer = buffer

    def send(self, data: str | bytes, add_cr: bool = True) -> None:
        self.sent.append(data if isinstance(data, bytes) else encode_command_bytes(data))

    def get_buffer(self) -> bytes:
        return self._buffer


@pytest.mark.parametrize("state", NON_RESETTING_STATES)
def test_a_broadcast_is_a_noop_outside_resetting(state):
    state_machine = DialogueStateMachine(state, matched=OTHER_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert state == state_machine.state
    assert [] == state_machine.sent


@pytest.mark.parametrize("state", NON_RESETTING_STATES)
def test_even_our_own_slots_broadcast_is_a_noop_outside_resetting(state):
    # answering it would feed the menu's p<slot> to whatever question is outstanding
    state_machine = DialogueStateMachine(state, matched=OUR_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert state == state_machine.state
    assert [] == state_machine.sent


def test_resetting_re_enters_the_database_when_our_slot_finishes():
    state_machine = DialogueStateMachine(State.RESETTING, matched=OUR_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    # stay put and let the next real prompt say where we are
    assert State.RESETTING == state_machine.state
    assert [b"p0"] == state_machine.sent


def test_a_broadcast_does_not_clear_an_outstanding_resetting_menu_answer():
    """RESETTING acts on the broadcast, but acting on it says nothing about our answer."""
    state_machine = DialogueStateMachine(State.RESETTING, matched=OUR_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)
    assert [b"p0"] == state_machine.sent
    assert "p0" == state_machine.pending_menu_answer

    # another slot finishes before our answer is echoed
    state_machine.child.after = OTHER_SLOTS_BROADCAST
    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert "p0" == state_machine.pending_menu_answer

    state_machine.maybe_apply_transition(Prompt.OPTION)

    assert [b"p0"] == state_machine.sent


def test_the_resetting_answer_is_repeated_once_it_has_been_echoed():
    state_machine = DialogueStateMachine(State.RESETTING, matched=OUR_SLOTS_BROADCAST)
    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    state_machine.output_since_menu_answer = b"p0\r\n"
    state_machine.child.after = OTHER_SLOTS_BROADCAST
    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    state_machine.maybe_apply_transition(Prompt.OPTION)

    assert [b"p0", b"p0"] == state_machine.sent


def test_resetting_keeps_waiting_when_another_slot_finishes():
    state_machine = DialogueStateMachine(State.RESETTING, matched=OTHER_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert State.RESETTING == state_machine.state
    assert [] == state_machine.sent


def test_a_broadcast_does_not_reopen_the_option_answer_gate():
    """A broadcast between an Option prompt and its redraw must not let it be answered twice."""
    state_machine = DialogueStateMachine(State.OPTION, matched=OTHER_SLOTS_BROADCAST)
    state_machine.pending_menu_answer = "p0"

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert State.OPTION == state_machine.state
    assert "p0" == state_machine.pending_menu_answer
    assert [] == state_machine.sent

    state_machine.maybe_apply_transition(Prompt.OPTION)

    # no p0 echo seen, so the earlier answer is still queued
    assert [] == state_machine.sent


def test_an_echo_the_broadcast_swallowed_still_counts_as_consumed():
    """The echo can land in the broadcast's chunk, and the redraw must still be answered."""
    state_machine = DialogueStateMachine(State.OPTION)
    state_machine.maybe_apply_transition(Prompt.OPTION)

    # as captured - the p0 echo lands inside the chunk the broadcast consumed
    state_machine.output_since_menu_answer = b" -+\r\np0\r\n\r\n+- Database 1 has finished initialising"
    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    # the redraw's own buffer no longer carries the echo
    state_machine._buffer = b" -+\r\n\x1b[f\x1b[2J\x1b[1;37;40m"
    state_machine.maybe_apply_transition(Prompt.OPTION)

    assert [b"p0", b"p0"] == state_machine.sent


def test_the_sex_question_is_answered_after_a_broadcast_interrupts():
    # the captured failure - two broadcasts landed after the name was answered
    state_machine = DialogueStateMachine(State.PERSONA_NAME_INPUT, matched=OTHER_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)
    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)
    state_machine.maybe_apply_transition(Prompt.PERSONA_SEX)

    assert State.PERSONA_SEX_INPUT == state_machine.state
    assert 1 == len(state_machine.sent)


@pytest.mark.parametrize("state", DIALOGUE_STATES)
def test_the_sex_question_is_answered_from_every_dialogue_state(state):
    state_machine = DialogueStateMachine(state)

    state_machine.maybe_apply_transition(Prompt.PERSONA_SEX)

    assert State.PERSONA_SEX_INPUT == state_machine.state
    assert 1 == len(state_machine.sent)


@pytest.mark.parametrize("state", DIALOGUE_STATES)
def test_the_tearoom_is_sipped_from_every_dialogue_state(state):
    state_machine = DialogueStateMachine(state)

    state_machine.maybe_apply_transition(Prompt.TEAROOM)

    assert State.TEAROOM == state_machine.state
    assert [b"sip t"] == state_machine.sent


@pytest.mark.parametrize("prompt", [Prompt.PERSONA_SEX, Prompt.TEAROOM])
def test_closing_does_not_answer_the_dialogues_questions(prompt):
    state_machine = DialogueStateMachine(State.CLOSING)

    state_machine.maybe_apply_transition(prompt)

    assert State.CLOSING == state_machine.state
    assert [] == state_machine.sent


class ScriptedChild:
    """Replays a fixed run of prompts in place of a pty. Entries are a Prompt or (Prompt, before)."""

    def __init__(self, prompts):
        self.prompts = [entry if isinstance(entry, tuple) else (entry, b"") for entry in prompts]
        self.before = b""
        self.after = b""

    def expect(self, patterns, timeout=None):
        prompt, before = self.prompts.pop(0)
        self.before = before
        return patterns.index(prompt.value)


class ScriptedStateMachine(ConnectionState):
    def __init__(self, state: State, prompts, db_slot: int = 0):
        self.state = state
        self.sent: list[bytes] = []
        self.history = deque(maxlen=10)
        self.chunk_history = deque(maxlen=10)
        self.last_prompt = None
        self.pending_menu_answer: str | None = None
        self.output_since_menu_answer = b""
        self.default_db_slot = db_slot
        self.max_transition_steps = 15
        self.max_continue_seconds = 300.0
        self.child = ScriptedChild(prompts)

    def send(self, data: str | bytes, add_cr: bool = True) -> None:
        self.sent.append(data if isinstance(data, bytes) else encode_command_bytes(data))

    def get_buffer(self) -> bytes:
        return self.child.before or b""


def test_a_broadcast_flood_does_not_exhaust_the_step_budget():
    # 64 slots broadcast far more times than the 15 step budget allows
    flood = [Prompt.DATABASE_FINISHED_INITIALIZING] * 64
    state_machine = ScriptedStateMachine(State.PERSONA_SEX_INPUT, [*flood, Prompt.TEAROOM, Prompt.TEA_SIPPED])

    state_machine.continue_until(State.TEA_SIPPED)

    assert State.TEA_SIPPED == state_machine.state


class EndlessBroadcastChild:
    """Never stops broadcasting, so only the elapsed bound can end the loop."""

    before = b""
    after = b""

    def expect(self, patterns, timeout=None):
        return patterns.index(Prompt.DATABASE_FINISHED_INITIALIZING.value)


def test_an_endless_broadcast_stream_still_gives_up():
    state_machine = ScriptedStateMachine(State.PERSONA_SEX_INPUT, [])
    state_machine.child = EndlessBroadcastChild()
    state_machine.max_continue_seconds = 0.2

    with pytest.raises(RuntimeError, match="elapsed"):
        state_machine.continue_until(State.TEA_SIPPED)


def test_an_echo_swallowed_by_an_intervening_match_is_accumulated():
    """Drives real expect() calls, so the accumulator is exercised rather than handed its answer."""
    state_machine = ScriptedStateMachine(
        State.OPTION,
        [
            Prompt.OPTION,
            (Prompt.DATABASE_FINISHED_INITIALIZING, b" -+\r\np0\r\n\r\n+- Database 1 has finished initialising"),
            (Prompt.OPTION, b" -+\r\n\x1b[f\x1b[2J\x1b[1;37;40m"),
        ],
    )

    state_machine.expect()
    state_machine.expect()
    state_machine.expect()

    assert [b"p0", b"p0"] == state_machine.sent
