import numpy as np
import pytest

from mudgym.db.index import UNKNOWN, room_name_to_index
from mudgym.db.rooms import ROOM_NAMES
from mudgym.envs.fields.superquicklook import SuperQuickLookField
from mudgym.envs.fields.tests.helpers import assert_valid_observation

COAL_BUNKER_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[1;36;40mthe coal\x1b[0;37;40m, \x1b[36mthe door\x1b[37m, \x1b[31mJuan the protector\x1b[37m "
    b"and \x1b[35m4 rats\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The large rat is carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The rat is carrying the following:\r\n"
    b"        nothing.\r\n"
)

KEEP_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mthird floor of keep\x1b[0;37;40m" contains '
    b"\x1b[1;36;40mthe manuscript\x1b[0;37;40m, \x1b[32mthe flickering haze\x1b[37m, "
    b"\x1b[32mthe wall\x1b[37m, \x1b[31mJuan the protector\x1b[37m and \x1b[36mthe mortar\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The mortar contains:\r\n"
    b"        powdered dragonblood.\r\n"
)

TWO_MORTALS_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[36mthe door\x1b[37m, \x1b[31mDavid the sorcerer\x1b[37m, "
    b"\x1b[31mJessica the protector\x1b[37m, \x1b[35m4 rats\x1b[37m and "
    b"\x1b[1;36;40mthe coal\x1b[0;37;40m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
)

ARCANE_FOREST_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40marcane forest\x1b[0;37;40m" contains '
    b"\x1b[1;35;40mthe dragon\x1b[0;37;40m, \x1b[1;36;40mthe amulet\x1b[0;37;40m, "
    b"\x1b[31mMark the protector\x1b[37m and \x1b[32mthe arcane tree\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
    b"The dragon is carrying the following:\r\n"
    b"        the emerald.\r\n"
)

WIZ_PLAYER_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[36mthe door\x1b[37m, \x1b[1;31;40mKeyser the wizard\x1b[0;37;40m and "
    b"\x1b[31mDumbo the protector\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
)

REVERSED_TITLES_CHUNK = (
    b'\x1b[0;37;40mThe place known as "\x1b[1;32;40mcoal bunker\x1b[0;37;40m" contains '
    b"\x1b[36mthe door\x1b[37m, \x1b[31mSir David\x1b[37m and \x1b[31mSister Jessica\x1b[37m.\r\n"
    b"You are carrying the following:\r\n"
    b"        nothing.\r\n"
)


# Captured from the WASM game with seed 123 and persona Aaron.
@pytest.mark.parametrize(
    ("raw", "room_name", "portable"),
    [
        pytest.param(
            b'sql\r\n\x1b[0;37;40mThe place known as "\x1b[1;32;40mback entrance to "Il Castellare"'
            b'\x1b[0;37;40m" contains \x1b[31mAaron the protector\x1b[37m and '
            b"\x1b[36mthe kitchen door\x1b[37m.\nYou are carrying the following:\n"
            b"        \x1b[1;36;40mthe ribbon\x1b[0;37;40m.\n",
            'back entrance to "il castellare"',
            "kitchen door",
            id="back-entrance",
        ),
        pytest.param(
            b'sql\r\n\x1b[0;37;40mThe place known as "\x1b[1;32;40moutside bedroom in "Il Castellare"'
            b'\x1b[0;37;40m" contains \x1b[31mAaron the protector\x1b[37m and '
            b"\x1b[36mthe door\x1b[37m.\nYou are carrying the following:\n"
            b"        \x1b[1;36;40mthe ribbon\x1b[0;37;40m.\n",
            'outside bedroom in "il castellare"',
            "door",
            id="outside-bedroom",
        ),
    ],
)
def test_quoted_room_names_preserve_contents_and_inventory(raw, room_name, portable):
    field = SuperQuickLookField()
    obs = field.extract([raw])

    np.testing.assert_equal(
        obs,
        {
            "room_name": room_name,
            "room_name_index": room_name_to_index(room_name),
            "here": ("Aaron the protector", portable),
            "portables": (portable,),
            "players": ("Aaron the protector",),
            "mobiles": (),
            "features": (),
            "inventory": ("ribbon",),
        },
    )
    assert_valid_observation(field, obs)


@pytest.mark.parametrize("line_break", ["\r", "\n", "\r\n"])
def test_room_names_cannot_span_lines(line_break):
    raw = f'The place known as "{line_break}coal bunker" contains nothing.\r\n'.encode()
    field = SuperQuickLookField()

    assert field.extract([raw]) == field.empty()


@pytest.mark.parametrize("room_name", ["unlisted room", "", "   "])
def test_unrecognised_or_empty_room_names_fail_loudly(room_name):
    raw = f'The place known as "{room_name}" contains nothing.\r\n'.encode()

    with pytest.raises(ValueError):
        SuperQuickLookField().extract([raw])


def test_orangery_colours_are_removed_without_losing_content_classification():
    raw = (
        b'\x1b[0;37;40mThe place known as "\x1b[1;32;40m\x1b[0;33;40morange'
        b'\x1b[1;32;40mry\x1b[0;37;40m" contains \x1b[31mAbbie the protector\x1b[37m '
        b"and \x1b[32mrain\x1b[37m.\nYou are carrying the following:\n        nothing.\n"
    )
    field = SuperQuickLookField()
    obs = field.extract([raw], persona="Abbie")

    assert obs["room_name"] == "orangery"
    assert obs["room_name_index"] == room_name_to_index("orangery") > 0
    assert obs["here"] == ("Abbie the protector", "rain")
    assert obs["features"] == ("rain",)
    assert obs["players"] == ()
    assert obs["inventory"] == ()
    assert_valid_observation(field, obs)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            COAL_BUNKER_CHUNK,
            {
                "room_name": "coal bunker",
                "here": ("coal", "door", "Juan the protector", "4 rats"),
                "portables": ("coal", "door"),
                "players": ("Juan the protector",),
                "mobiles": ("4 rats",),
                "features": (),
                "inventory": (),
            },
            id="coal-bunker",
        ),
        pytest.param(
            KEEP_CHUNK,
            {
                "room_name": "third floor of keep",
                "here": ("manuscript", "flickering haze", "wall", "Juan the protector", "mortar"),
                "portables": ("manuscript", "mortar"),
                "players": ("Juan the protector",),
                "mobiles": (),
                "features": ("flickering haze", "wall"),
                "inventory": (),
            },
            id="keep",
        ),
        pytest.param(
            TWO_MORTALS_CHUNK,
            {
                "room_name": "coal bunker",
                "here": ("door", "David the sorcerer", "Jessica the protector", "4 rats", "coal"),
                "portables": ("door", "coal"),
                "players": ("David the sorcerer", "Jessica the protector"),
                "mobiles": ("4 rats",),
                "features": (),
                "inventory": (),
            },
            id="two-mortals",
        ),
        pytest.param(
            ARCANE_FOREST_CHUNK,
            {
                "room_name": "arcane forest",
                "here": ("dragon", "amulet", "Mark the protector", "arcane tree"),
                "portables": ("amulet",),
                "players": ("Mark the protector",),
                "mobiles": ("dragon",),
                "features": ("arcane tree",),
                "inventory": (),
            },
            id="arcane-forest",
        ),
        pytest.param(
            WIZ_PLAYER_CHUNK,
            {
                "room_name": "coal bunker",
                "here": ("door", "Keyser the wizard", "Dumbo the protector"),
                "portables": ("door",),
                "players": ("Keyser the wizard", "Dumbo the protector"),
                "mobiles": (),
                "features": (),
                "inventory": (),
            },
            id="wizard-player",
        ),
    ],
)
def test_classifies_captured_room_contents_and_player_inventory(raw, expected):
    field = SuperQuickLookField()
    obs = field.extract([raw])
    expected = {**expected, "room_name_index": room_name_to_index(expected["room_name"])}

    np.testing.assert_equal(obs, expected)
    assert_valid_observation(field, obs)


@pytest.mark.parametrize(
    ("raw", "persona", "observer", "remaining"),
    [
        pytest.param(
            TWO_MORTALS_CHUNK, "David", "David the sorcerer", "Jessica the protector", id="ordinary-title-David"
        ),
        pytest.param(
            TWO_MORTALS_CHUNK, "Jessica", "Jessica the protector", "David the sorcerer", id="ordinary-title-Jessica"
        ),
        pytest.param(REVERSED_TITLES_CHUNK, "David", "Sir David", "Sister Jessica", id="title-first-David"),
        pytest.param(REVERSED_TITLES_CHUNK, "Jessica", "Sister Jessica", "Sir David", id="title-first-Jessica"),
    ],
)
def test_persona_exclusion_only_removes_the_observer_from_players(raw, persona, observer, remaining):
    obs = SuperQuickLookField().extract([raw], persona=persona)

    assert obs["players"] == (remaining,)
    assert observer in obs["here"]


@pytest.mark.parametrize("raw", [b"", b"It's too dark for you to see anything."])
def test_missing_room_view_returns_empty_defaults(raw):
    field = SuperQuickLookField()
    expected = {
        "room_name": UNKNOWN,
        "room_name_index": 0,
        "here": (),
        "inventory": (),
        "features": (),
        "portables": (),
        "mobiles": (),
        "players": (),
    }

    np.testing.assert_equal(field.empty(), expected)
    np.testing.assert_equal(field.extract([raw]), expected)


def test_last_room_name_uses_the_last_index_in_its_space():
    field = SuperQuickLookField()
    raw = f'The place known as "{ROOM_NAMES[-1]}" contains nothing.\r\n'.encode()
    obs = field.extract([raw])

    assert obs["room_name"] == ROOM_NAMES[-1]
    assert obs["room_name_index"] == len(ROOM_NAMES)
    assert_valid_observation(field, obs)
