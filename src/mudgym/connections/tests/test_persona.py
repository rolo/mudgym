"""Generated persona names must satisfy the game's rules."""

from mudgym.connections import persona


def test_generated_names_retry_until_faker_returns_an_acceptable_name(monkeypatch):
    candidates = iter(["Anne-Marie", "Renée", "Richard", "Alexandriaaa", "Alice"])
    monkeypatch.setattr(persona.faker, "first_name", lambda: next(candidates))

    assert persona.generate_persona_name() == "Alice"
