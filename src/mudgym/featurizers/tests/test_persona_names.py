import pytest

from mudgym.db.levels import LEVELS, PREFIX_FORMAT_LEVELS
from mudgym.featurizers.persona_names import bare_persona_name


def test_reverse_format_levels_are_real_levels():
    assert PREFIX_FORMAT_LEVELS <= set(LEVELS)


@pytest.mark.parametrize(
    "full_name, expected_name",
    [
        # a novice with no prefix or postfix is rendered as the bare name alone
        ("Dood", "Dood"),
        # the usual shape: "<name> the <level>"
        ("Dood the protector", "Dood"),
        ("Dood the mage", "Dood"),
        ("Keyser the archizard", "Keyser"),
        ("Dood the guest", "Dood"),
        # prefixes the game issues itself sit between the "the" and the level
        ("Dood the dragon-slaying warrior", "Dood"),
        ("Dood the sorcerised sorcerer", "Dood"),
        ("Dood the doomed protector", "Dood"),
        # a prefix on a novice makes the otherwise hidden level appear
        ("Crazpotlm the awkward novice", "Crazpotlm"),
        # postfixes are free text and follow the level
        ("Dood the mage with a fringe", "Dood"),
        ("Dood the sorcerer (Player of the Month)", "Dood"),
    ],
)
def test_bare_name_of_a_normally_ordered_full_name(full_name, expected_name):
    assert bare_persona_name(full_name) == expected_name


@pytest.mark.parametrize(
    "full_name, expected_name",
    [
        # Sir/Lady (unprotected non-magic-user, 102,400 points) and Brother/Sister (protected
        # magic-user, 6,400) put the level first, so the persona is the last word
        ("Sir Dood", "Dood"),
        ("Lady Dood", "Dood"),
        ("Brother Dood", "Dood"),
        ("Sister Dood", "Dood"),
        # a prefix is capitalised in front of the level here
        ("Awkward Lady Crazpotlm", "Crazpotlm"),
        # a postfix still follows the level, which reads oddly but is what the game emits
        ("Sir with a fringe Boop", "Boop"),
        ("Sir (Player of the Month) Boop", "Boop"),
    ],
)
def test_bare_name_of_a_reverse_ordered_full_name(full_name, expected_name):
    assert bare_persona_name(full_name) == expected_name


@pytest.mark.parametrize(
    "full_name, expected_name",
    [
        # persona names are alphabetic and unvetted against the level vocabulary, so a level word
        # can be the name itself: the ordering, not the word, decides which end to read
        ("Lady the protector", "Lady"),
        ("Sir the mage", "Sir"),
        ("Sir Sister", "Sister"),
        ("The the protector", "The"),
        ("Sir The", "The"),
    ],
)
def test_bare_name_when_the_persona_is_named_after_a_level(full_name, expected_name):
    assert bare_persona_name(full_name) == expected_name


@pytest.mark.parametrize("full_name", ["  Dood the protector  ", "Dood the protector\r", "\tSir Dood\t"])
def test_surrounding_whitespace_is_ignored(full_name):
    assert bare_persona_name(full_name) == "Dood"


@pytest.mark.parametrize("full_name", ["", "   "])
def test_an_empty_full_name_is_an_error(full_name):
    with pytest.raises(ValueError, match="no persona name"):
        bare_persona_name(full_name)
