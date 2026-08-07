"""
    Dood                            a novice with no prefix or postfix is just the bare name
    Dood the sorcerised sorcerer    <name> the [<prefix> ]<level>[ <postfix>]
    Sir Dood                        [<prefix> ]<level>[ <postfix>] <name>

The game issues prefixes of its own ("dragon-slaying" for killing the dragon, plus "dragonfly-slaying",
"doomed", "surrealist", "sorcerised", "touring", "special" and "ersatz"), its own postfix is
"(Player of the Month)", and wizzes can set either to anything. What is fixed is where the name
sits: first, unless the level is one that reverses the ordering, in which case last.

One further shape is out of scope: a wiz can INVERT a player, rendering them "Dood [sorcerer]".
Nothing mudgym drives can turn that on, and a prefix under inversion is ambiguous with the name.
"""

from mudgym.db.levels import PREFIX_FORMAT_LEVELS


def bare_persona_name(full_name: str) -> str:
    """Return the persona's own name from the full name the game renders for them."""
    words = full_name.split()
    if not words:
        raise ValueError(f"no persona name in: {full_name!r}")

    # "<name> the ..." always continues into a level, so the "the" of a two word name belongs to
    # a persona named "The" who is a Sir or a Brother, not to the usual ordering
    if len(words) > 2 and words[1].lower() == "the":
        return words[0]

    if any(word.lower() in PREFIX_FORMAT_LEVELS for word in words):
        return words[-1]

    return words[0]
