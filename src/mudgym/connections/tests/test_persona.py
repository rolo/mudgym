"""Generated persona names must satisfy the game's rules."""

from mudgym.connections import persona
from mudgym.connections.prompts import State
from mudgym.connections.transitions import choose_or_create_persona


def test_generated_names_retry_until_faker_returns_an_acceptable_name(monkeypatch):
    candidates = iter(["Anne-Marie", "Renée", "Richard", "Alexandriaaa", "Alice"])
    monkeypatch.setattr(persona.faker, "first_name", lambda: next(candidates))

    assert persona.generate_persona_name() == "Alice"


def test_unused_persona_slots_are_not_treated_as_existing_personas():
    class FakeStateMachine:
        default_persona_slot = 2
        default_name_generator = None
        state = State.PERSONA_SELECT

        def __init__(self):
            self.sent = []

        def get_buffer(self):
            return b"(1) Alice, (2) **Unused**."

        def send(self, value):
            self.sent.append(value)

    state_machine = FakeStateMachine()

    choose_or_create_persona(state_machine)

    assert state_machine.sent == ["1"]
