"""
Handles persona selection and creation logic.
"""

import re

from faker import Faker

from mudgym.featurizers.strings import decode_text_bytes
from mudgym.logs import get_logger

logger = get_logger(__name__)

faker = Faker()

# the game rejects persona names longer than this or containing anything non-alphabetic
# ("Use names of 10 characters at most, please")
PERSONA_NAME_MAX_LENGTH = 10

PERSONA_NAME_BLACKLIST = [
    "richard",
]


def parse_persona_screen(text: bytes) -> dict[int, str]:
    """Extract persona names from persona selection text."""
    text_str = decode_text_bytes(text)
    # names are only ever max 10 characters and dont include punctuation, whitespace or non alpha chars
    # TODO: tighten up this regex to only include valid persona names and make it operate on bytestrings
    pattern = r"\((\d+)\)\s+([A-Za-z][\w'-]{0,9}|\*\*Unused\*\*)(?:,|\.|$)"
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
