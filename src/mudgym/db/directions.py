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

DIRECTION_INDEX_BY_NAME = {direction: index for index, direction in enumerate(DIRECTIONS)}
