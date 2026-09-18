import random

import pytest

from mudgym.connections.persona import Persona, parse_persona, resolve_personas, validate_persona_pool


@pytest.mark.parametrize("sex, expected", [("male", "male"), ("M", "male"), ("female", "female"), ("F", "female")])
def test_persona_normalizes_sex_without_changing_name(sex, expected):
    persona = Persona("aLiCe", sex)
    assert persona.name == "aLiCe"
    assert persona.sex == expected


@pytest.mark.parametrize("name", ["", "ElevenChars", "two words", "Zoë", "Amy2", 42])
def test_invalid_name(name):
    with pytest.raises(ValueError, match="name"):
        Persona(name)


@pytest.mark.parametrize("sex", ["", "unknown", 42])
def test_invalid_sex(sex):
    with pytest.raises(ValueError, match="sex"):
        Persona("Alice", sex)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (("Dave",), Persona("Dave")),
        (("m",), Persona(sex="male")),
        (("F",), Persona(sex="female")),
        (("MaLe",), Persona(sex="male")),
        (("FeMaLe",), Persona(sex="female")),
        (("Fred",), Persona("Fred")),
        (("Male", None), Persona("Male")),
        ((None, None), Persona()),
    ],
)
def test_persona_shorthand(entry, expected):
    assert parse_persona(entry) == expected


@pytest.mark.parametrize("entry", [(), ("Dave", "m", "extra"), "mf"])
def test_invalid_persona_shorthand(entry):
    with pytest.raises(ValueError, match="persona"):
        parse_persona(entry)


@pytest.mark.parametrize("pool", [(Persona(),), (Persona("Ash"), Persona("ash"))])
def test_pool_requires_named_unique_entries(pool):
    with pytest.raises(ValueError):
        validate_persona_pool(pool)


@pytest.mark.parametrize(
    ("requests", "pool", "expected"),
    [
        (
            (Persona(), Persona("aaron", "m")),
            (Persona("Aaron", "m"), Persona("Ash", "m")),
            (Persona("Ash", "m"), Persona("aaron", "m")),
        ),
        (
            (Persona(), Persona(sex="m")),
            (Persona("Aaron", "m"), Persona("Amy", "f")),
            (Persona("Amy", "f"), Persona("Aaron", "m")),
        ),
        (
            (Persona(sex="m"), Persona(sex="f")),
            (Persona("Aaron", "m"), Persona("Ash")),
            (Persona("Aaron", "m"), Persona("Ash", "f")),
        ),
        (
            (Persona(sex="f"), Persona(sex="m")),
            (Persona("Aaron", "m"), Persona("Ash")),
            (Persona("Ash", "f"), Persona("Aaron", "m")),
        ),
        ((Persona("Custom", "f"),), (), (Persona("Custom", "f"),)),
    ],
)
def test_resolve_personas(requests, pool, expected):
    assert resolve_personas(requests, pool=pool, seed=0) == expected


@pytest.mark.parametrize(
    ("requests", "pool", "message"),
    [
        ((Persona("Ash"), Persona("ash")), (), "duplicate"),
        ((Persona(), Persona()), (Persona("Only", "m"),), "pool"),
        ((Persona(sex="m"),), (Persona("Alice", "f"),), "pool"),
    ],
)
def test_impossible_persona_requests(requests, pool, message):
    with pytest.raises(ValueError, match=message):
        resolve_personas(requests, pool=pool, seed=0)


def test_seeded_selection_is_reproducible_prefix_stable_and_private():
    before = random.getstate()
    pool = tuple(Persona(name) for name in ("Ada", "Bea", "Cia", "Dee", "Ema", "Fia"))
    short = resolve_personas((Persona(),) * 3, pool=pool, seed=-442)
    assert short == resolve_personas((Persona(),) * 3, pool=pool, seed=-442)
    assert short == resolve_personas((Persona(),) * 6, pool=pool, seed=-442)[:3]
    assert resolve_personas((Persona("Custom"),), pool=(), seed=3) == resolve_personas(
        (Persona("Custom"),), pool=pool, seed=3
    )
    assert random.getstate() == before


def test_default_pool_fills_a_large_mixed_persona_list():
    requests = tuple(Persona(sex="m" if index % 2 else "f") for index in range(60))
    resolved = resolve_personas(requests, seed=2026)
    assert len({persona.name.casefold() for persona in resolved}) == len(requests)
    assert tuple(persona.sex for persona in resolved) == tuple(persona.sex for persona in requests)
