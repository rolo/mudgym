"""Split the lists the game prints into the names they hold."""

# The door of the giant sitting room is "sturdy, giant door", the one name in the game that holds a comma.
NAMES_HOLDING_A_COMMA = ("sturdy, giant door",)

HELD_COMMA = "\0"


def split_listed_names(listed: str, separator: str) -> list[str]:
    """Split a printed list on its separator, keeping whole a name that holds a comma itself."""
    for name in NAMES_HOLDING_A_COMMA:
        listed = listed.replace(name, name.replace(",", HELD_COMMA))
    return [part.replace(HELD_COMMA, ",") for part in listed.split(separator)]
