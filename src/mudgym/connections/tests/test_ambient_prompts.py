"""Ambient broadcasts must not hijack the login dialogue."""

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

MENU_STATES = (State.INITIAL, State.LOGIN, State.OPTION, State.RESETTING)

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
        self.default_db_slot = db_slot
        self.child = SimpleNamespace(after=matched)
        self._buffer = buffer

    def send(self, data: str | bytes, add_cr: bool = True) -> None:
        self.sent.append(data if isinstance(data, bytes) else encode_command_bytes(data))

    def get_buffer(self) -> bytes:
        return self._buffer


@pytest.mark.parametrize("state", [State.PERSONA_SELECT, State.PERSONA_NAME_INPUT, State.PERSONA_SEX_INPUT])
def test_a_broadcast_leaves_the_persona_dialogue_where_it_was(state):
    state_machine = DialogueStateMachine(state, matched=OTHER_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert state == state_machine.state
    assert [] == state_machine.sent


def test_a_broadcast_for_our_own_slot_leaves_the_persona_dialogue_alone():
    # answering it would feed the menu's p<slot> to whatever question is outstanding
    state_machine = DialogueStateMachine(State.PERSONA_SELECT, matched=OUR_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert State.PERSONA_SELECT == state_machine.state
    assert [] == state_machine.sent


def test_the_sex_question_is_answered_after_a_broadcast_interrupts():
    # the captured failure: two broadcasts landed between the name answer and the sex question
    state_machine = DialogueStateMachine(State.PERSONA_SELECT, matched=OTHER_SLOTS_BROADCAST)

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


@pytest.mark.parametrize("state", MENU_STATES)
def test_a_broadcast_for_our_slot_still_enters_the_database_from_the_menus(state):
    state_machine = DialogueStateMachine(state, matched=OUR_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert State.OPTION == state_machine.state
    assert [b"p0"] == state_machine.sent


@pytest.mark.parametrize("state", MENU_STATES)
def test_another_slots_broadcast_is_not_answered_from_the_menus(state):
    state_machine = DialogueStateMachine(state, matched=OTHER_SLOTS_BROADCAST)

    state_machine.maybe_apply_transition(Prompt.DATABASE_FINISHED_INITIALIZING)

    assert [] == state_machine.sent


@pytest.mark.parametrize("prompt", [Prompt.PERSONA_SEX, Prompt.TEAROOM])
def test_closing_does_not_answer_the_dialogues_questions(prompt):
    state_machine = DialogueStateMachine(State.CLOSING)

    state_machine.maybe_apply_transition(prompt)

    assert State.CLOSING == state_machine.state
    assert [] == state_machine.sent
