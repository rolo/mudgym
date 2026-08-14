import string

import numpy as np
from gymnasium import spaces

# general purpose
INDEX_DTYPE = np.int64
INT_DTYPE = np.int32
FLOAT_DTYPE = np.float32
BYTE_DTYPE = np.uint8
BIT_DTYPE = np.int8  # Gymnasium's MultiBinary space and discrete action masks use int8.

# game specific
# The game's text is 7-bit: printable ASCII plus ANSI escapes and line breaks (the entire muddle
# text database contains no byte above 0x7F). Wire bytes above 0x7F are protocol codes (fecodes,
# the unsupported client mode), never text, and the decode path rejects them loudly (see
# featurizers.strings.decode_text_bytes). Latin-1 remains the byte-to-text mapping throughout.
PRINTABLE_ASCII = string.ascii_letters + string.digits + string.punctuation + " "
TEXT_CHARSET = PRINTABLE_ASCII + "\n"

# Single-line game values: room names, weather, item descriptions. Same characters as text,
# never a line break.
SINGLE_LINE_CHARSET = PRINTABLE_ASCII

# An action is one logical game input line: the session sends its observation-command line separately,
# so a line break inside an action would smuggle extra wire commands. The game also
# transliterates input bytes above 0x7F (a typed 0xE9 'e-acute' echoes back as 'i'), so actions
# stay printable ASCII.
ACTION_CHARSET = PRINTABLE_ASCII

# Game identifiers are single tokens: room ids ("groad1", "((store))", "Limbo"; at most 9 chars,
# enumerated from the room db), object instance ids ("necklace0", "key22"; a vocabulary word of
# at most 14 chars - MAXIDLEN in the game's database compiler - plus an instance number), and
# direction words. Persona names are also identifiers (alphabetic, at most 10 chars) but their
# limit lives with the persona logic in connections.persona.
IDENTIFIER_CHARSET = string.ascii_letters + string.digits + "()-"
IDENTIFIER_MAX_LENGTH = 17

# a sequence of game entities, eg in an inventory or room contents
ITEM_MAX_LENGTH = 96
ITEM_SPACE = spaces.Sequence(
    spaces.Text(max_length=ITEM_MAX_LENGTH, min_length=0, charset=SINGLE_LINE_CHARSET),
    stack=False,
)

# a sequence of machine-readable game identifiers, eg the fei inventory ids ("brand39",
# "cloth-of-gold") and mgcheats' ``here`` values
IDENTIFIER_SPACE = spaces.Sequence(
    spaces.Text(max_length=IDENTIFIER_MAX_LENGTH, min_length=0, charset=IDENTIFIER_CHARSET),
    stack=False,
)
ROOM_ID_MAX_LENGTH = 9
ROOM_NAME_MAX_LENGTH = 60

ACTION_MAX_LENGTH = 64
TEXT_MAX_LENGTH = 4096
