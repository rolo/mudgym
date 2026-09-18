"""Stable persona name pools and deterministic persona selection."""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

SEX_ALIASES = {"m": "male", "male": "male", "f": "female", "female": "female"}


@dataclass(frozen=True)
class Persona:
    """A requested or resolved game persona."""

    name: str | None = None
    sex: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z]{1,10}", self.name)):
            raise ValueError("persona name must contain 1 to 10 ASCII letters")
        if self.sex is not None:
            sex = self.sex.casefold() if isinstance(self.sex, str) else ""
            if sex not in SEX_ALIASES:
                raise ValueError("persona sex must be male, female, m, f, or None")
            object.__setattr__(self, "sex", SEX_ALIASES[sex])


def parse_persona(entry: tuple[str] | tuple[str | None, str | None]) -> Persona:
    """Expand a persona entry, recognising sex aliases only in one-item tuples."""
    match entry:
        case (str(value),):
            if value.casefold() in SEX_ALIASES:
                return Persona(sex=value)
            return Persona(name=value)
        case (name, sex):
            return Persona(name, sex)
        case _:
            raise ValueError("persona entries must be (name,), (sex,), or (name, sex) tuples")


MALE_PERSONA_NAMES = (
    "Aaron",
    "Adam",
    "Albert",
    "Alexander",
    "Alfie",
    "Alfred",
    "Angus",
    "Anthony",
    "Archie",
    "Arlo",
    "Arthur",
    "Austin",
    "Bailey",
    "Ben",
    "Blake",
    "Bobby",
    "Bradley",
    "Brandon",
    "Caleb",
    "Cameron",
    "Carter",
    "Charles",
    "Ciaran",
    "Cody",
    "Cole",
    "Corey",
    "Craig",
    "Daniel",
    "Danny",
    "Declan",
    "Dexter",
    "Dylan",
    "Edward",
    "Elijah",
    "Elliot",
    "Ellis",
    "Ethan",
    "Ezra",
    "Felix",
    "Finn",
    "Frankie",
    "Fraser",
    "Freddie",
    "Gabriel",
    "George",
    "Harry",
    "Harvey",
    "Hayden",
    "Henry",
    "Hugo",
    "Hunter",
    "Isaac",
    "Jack",
    "Jacob",
    "Jake",
    "James",
    "Jason",
    "Jay",
    "Joel",
    "Jordan",
    "Josh",
    "Jude",
    "Kai",
    "Kane",
    "Kayden",
    "Kian",
    "Kieran",
    "Kyle",
    "Lee",
    "Leo",
    "Levi",
    "Lewis",
    "Liam",
    "Logan",
    "Lucas",
    "Luke",
    "Mark",
    "Mason",
    "Max",
    "Michael",
    "Miles",
    "Milo",
    "Mitchell",
    "Myles",
    "Niall",
    "Nicholas",
    "Noah",
    "Ollie",
    "Oscar",
    "Patrick",
    "Paul",
    "Peter",
    "Ralph",
    "Reece",
    "Reggie",
    "Rhys",
    "Riley",
    "Robbie",
    "Ronnie",
    "Rory",
    "Ross",
    "Rowan",
    "Ruben",
    "Ryan",
    "Samuel",
    "Scott",
    "Sean",
    "Seth",
    "Sonny",
    "Spencer",
    "Stanley",
    "Taylor",
    "Theo",
    "Thomas",
    "Tobias",
    "Toby",
    "Tyler",
    "William",
    "Yannis",
    "Zac",
    "Zebedee",
)

FEMALE_PERSONA_NAMES = (
    "Abbie",
    "Abigail",
    "Aimee",
    "Alana",
    "Alexandra",
    "Alice",
    "Amelie",
    "Amy",
    "Angel",
    "Anna",
    "Ayla",
    "Bella",
    "Beth",
    "Bonnie",
    "Caitlyn",
    "Cara",
    "Casey",
    "Catherine",
    "Cerys",
    "Charlotte",
    "Chloe",
    "Ciara",
    "Courtney",
    "Demi",
    "Ebony",
    "Eleanor",
    "Elena",
    "Elise",
    "Eliza",
    "Ellen",
    "Elsie",
    "Emilia",
    "Emma",
    "Erin",
    "Esme",
    "Esther",
    "Evelyn",
    "Evie",
    "Faith",
    "Faye",
    "Felicity",
    "Florence",
    "Freya",
    "Gemma",
    "Georgina",
    "Hannah",
    "Harper",
    "Heidi",
    "Hollie",
    "Imogen",
    "Isabelle",
    "Isla",
    "Jennifer",
    "Jessica",
    "Jodie",
    "Julia",
    "Katherine",
    "Kayla",
    "Keira",
    "Kelsey",
    "Kirsty",
    "Lacey",
    "Laura",
    "Lauren",
    "Libby",
    "Lillie",
    "Lois",
    "Lola",
    "Louise",
    "Lucy",
    "Luna",
    "Lydia",
    "Maddison",
    "Madeleine",
    "Maisy",
    "Maria",
    "Martha",
    "Mary",
    "Matilda",
    "Maya",
    "Megan",
    "Melissa",
    "Mia",
    "Millie",
    "Morgan",
    "Mya",
    "Nancy",
    "Naomi",
    "Natasha",
    "Nicole",
    "Nina",
    "Olivia",
    "Orla",
    "Paige",
    "Penelope",
    "Phoebe",
    "Rachel",
    "Rebecca",
    "Rhiannon",
    "Robyn",
    "Rosie",
    "Sadie",
    "Samantha",
    "Sarah",
    "Savannah",
    "Scarlett",
    "Seren",
    "Shannon",
    "Sienna",
    "Skye",
    "Sofia",
    "Sophia",
    "Stephanie",
    "Tegan",
    "Thea",
    "Tia",
    "Tilly",
    "Victoria",
    "Yasmin",
    "Zara",
    "Zoe",
)


def unique_persona_names(personas: Sequence[Persona]) -> set[str]:
    names = [persona.name.casefold() for persona in personas if persona.name is not None]
    unique = set(names)
    if len(names) != len(unique):
        raise ValueError("duplicate persona name")
    return unique


def validate_persona_pool(pool: Sequence[Persona]) -> tuple[Persona, ...]:
    """Snapshot a named persona pool and reject ambiguous names."""
    snapshot = tuple(pool)
    if any(persona.name is None for persona in snapshot):
        raise ValueError("persona pool entries require a name")
    unique_persona_names(snapshot)
    return snapshot


DEFAULT_PERSONA_POOL = validate_persona_pool(
    sorted(
        [Persona(name, "male") for name in MALE_PERSONA_NAMES]
        + [Persona(name, "female") for name in FEMALE_PERSONA_NAMES],
        key=lambda persona: persona.name,
    )
)


def resolve_personas(
    personas: Sequence[Persona],
    *,
    pool: Sequence[Persona] = DEFAULT_PERSONA_POOL,
    seed: int,
) -> tuple[Persona, ...]:
    """Resolve personas from a validated pool without consuming application random state."""
    requests = tuple(personas)
    reserved_names = unique_persona_names(requests)
    available = [persona for persona in pool if persona.name.casefold() not in reserved_names]
    pending = sorted(
        (index for index, persona in enumerate(requests) if persona.name is None),
        key=lambda index: requests[index].sex is None,
    )
    name_random = random.Random(seed)
    sex_random = random.Random(f"persona-sex:{seed}")
    resolved = list(requests)
    for position, index in enumerate(pending):
        sex = requests[index].sex
        candidates = available
        if sex is not None:
            other_sex = "female" if sex == "male" else "male"
            # Do not spend a neutral name that a later, constrained player needs.
            can_use_neutral = sum(persona.sex in (None, other_sex) for persona in available) > sum(
                requests[later].sex == other_sex for later in pending[position + 1 :]
            )
            candidates = [
                persona for persona in available if persona.sex == sex or (persona.sex is None and can_use_neutral)
            ]
        if not candidates:
            raise ValueError("persona pool cannot satisfy the requested personas")
        selected = name_random.choice(candidates)
        available.remove(selected)
        resolved[index] = Persona(selected.name, selected.sex or sex)
    return tuple(
        persona if persona.sex is not None else Persona(persona.name, sex_random.choice(("male", "female")))
        for persona in resolved
    )
