"""
Handles persona selection and creation logic.
"""

import os
import random
import re

from faker import Faker

from mudgym.featurizers.strings import decode_text_bytes

faker = Faker()

# the game rejects persona names longer than this or containing anything non-alphabetic
# ("Use names of 10 characters at most, please")
PERSONA_NAME_MAX_LENGTH = 10

PERSONA_NAME_BLACKLIST = [
    "richard",
]
UNUSED_PERSONA = "**Unused**"

# what the game takes at "What sex do you wish to be?"
PERSONA_SEXES = (b"m", b"f")

# set MUDGYM_PERSONA_SEX to any of these to create every persona that sex, otherwise each one is
# drawn at random
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
    # the game rejects names longer than PERSONA_NAME_MAX_LENGTH or containing anything
    # non-alphabetic, and faker first names can be hyphenated or accented ("Anne-Marie",
    # "Renée"), which would wedge persona creation at the name prompt
    name = ""
    while (
        not name.isascii()
        or not name.isalpha()
        or len(name) > PERSONA_NAME_MAX_LENGTH
        or name.lower() in PERSONA_NAME_BLACKLIST
    ):
        name = faker.first_name()
    return name


def generate_persona_sex() -> bytes:
    override = os.getenv("MUDGYM_PERSONA_SEX")
    if not override:
        return random.choice(PERSONA_SEXES)

    wanted = override.strip().lower()
    if wanted not in PERSONA_SEX_BY_NAME:
        raise ValueError(f"MUDGYM_PERSONA_SEX must be one of {', '.join(PERSONA_SEX_BY_NAME)}, got {override!r}")

    return PERSONA_SEX_BY_NAME[wanted]
