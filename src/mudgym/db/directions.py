DIRECTIONS = [
    "north",
    "east",
    "south",
    "west",
    "northeast",
    "southeast",
    "southwest",
    "northwest",
    "up",
    "down",
    "in",
    "out",
    "over",
    "swampward",  # cannot be used at Mage / Sir level
]
"""The game's movement directions, in the order used by its `exits` command, which is also the discrete action index and the `available_exits` mask order."""

DIRECTION_ALIASES = {
    "n": "north",
    "e": "east",
    "s": "south",
    "w": "west",
    "ne": "northeast",
    "se": "southeast",
    "sw": "southwest",
    "nw": "northwest",
    "u": "up",
    "d": "down",
    "in": "in",
    "o": "out",
    "j": "over",
    "zw": "swampward",
    "jump": "over",
    "swamp": "swampward",
}

DIRECTION_INDEX_BY_NAME = {direction: index for index, direction in enumerate(DIRECTIONS)}
