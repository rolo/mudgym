"""
Handles persona selection and creation logic.
"""

import os
import random
import re

from mudgym.featurizers.strings import decode_text_bytes

# Valid names shared by every connection. Stable order also gives seeded worlds repeatable identities.
PERSONA_NAMES = (
    "Ada",
    "Alba",
    "Alder",
    "Amos",
    "Anouk",
    "Arden",
    "Astrid",
    "Aurelia",
    "Bly",
    "Bram",
    "Bruno",
    "Caleb",
    "Calla",
    "Cassia",
    "Cedric",
    "Clio",
    "Cora",
    "Cyrus",
    "Dahlia",
    "Delia",
    "Dorian",
    "Edda",
    "Edwin",
    "Elba",
    "Elena",
    "Elias",
    "Esme",
    "Ewan",
    "Fabian",
    "Fenna",
    "Fern",
    "Finn",
    "Flora",
    "Freya",
    "Gideon",
    "Gita",
    "Greta",
    "Hal",
    "Hana",
    "Harlan",
    "Hazel",
    "Hester",
    "Hugo",
    "Ida",
    "Imogen",
    "Inez",
    "Ivo",
    "Jonas",
    "Juno",
    "Kaya",
)
UNUSED_PERSONA = "**Unused**"

# what the game takes at "What sex do you wish to be?"
PERSONA_SEXES = (b"m", b"f")

# set MUDGYM_PERSONA_SEX to any of these to pin it, otherwise each persona is drawn at random
PERSONA_SEX_BY_NAME = {"m": b"m", "male": b"m", "f": b"f", "female": b"f"}


def parse_persona_screen(text: bytes) -> dict[int, str]:
    """Extract persona names from persona selection text."""
    text_str = decode_text_bytes(text)
    # names are only ever max 10 characters and dont include punctuation, whitespace or non alpha chars
    # TODO: tighten up this regex to only include valid persona names and make it operate on bytestrings
    pattern = rf"\((\d+)\)\s+([A-Za-z][\w'-]{{0,9}}|{re.escape(UNUSED_PERSONA)})(?:,|\.|$)"
    matches = re.findall(pattern, text_str)
    return {int(num): name for num, name in matches}


def generate_persona_name() -> str:
    return random.choice(PERSONA_NAMES)


def generate_persona_sex() -> bytes:
    override = os.getenv("MUDGYM_PERSONA_SEX")
    if not override:
        return random.choice(PERSONA_SEXES)

    wanted = override.strip().lower()
    if wanted not in PERSONA_SEX_BY_NAME:
        raise ValueError(f"MUDGYM_PERSONA_SEX must be one of {', '.join(PERSONA_SEX_BY_NAME)}, got {override!r}")

    return PERSONA_SEX_BY_NAME[wanted]
