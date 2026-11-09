import numpy as np
import pytest

from mudgym.db.index import UNKNOWN, room_id_to_index, room_name_to_index
from mudgym.db.rooms import ROOM_IDS, ROOM_NAMES
from mudgym.envs.fields.mgcheats import MGCheatsField
from mudgym.envs.fields.tests.helpers import assert_valid_observation

MGCHEATS_BEATEN_TRACK = (
    b"[mgcheats]room_id=mtrack2; room_name=beaten track; fighting=0; dark=0; glowing=0; "
    b"asleep=0; gifted=0; here=[necklace0, road]; players=[]; mobiles=[]; ticks=0; inventory=[][/mgcheats]\r\n"
)


@pytest.mark.parametrize(
    ("room_id", "room_name", "index_key", "expected_index"),
    [
        pytest.param("wthstp", "weathered steps", "room_id_index", len(ROOM_IDS), id="last-room-id"),
        pytest.param("mzroom", "zombie room", "room_name_index", len(ROOM_NAMES), id="last-room-name"),
    ],
)
def test_room_index_boundaries(room_id, room_name, index_key, expected_index):
    raw = MGCHEATS_BEATEN_TRACK.replace(b"mtrack2", room_id.encode())
    raw = raw.replace(b"beaten track", room_name.encode())
    field = MGCheatsField()

    obs = field.extract([raw])

    assert obs["room_id"] == room_id
    assert obs["room_name"] == room_name
    assert obs[index_key] == expected_index
    assert_valid_observation(field, obs)


@pytest.mark.parametrize("original", [b"mtrack2", b"beaten track"], ids=["room_id", "room_name"])
def test_unknown_or_empty_rooms_fail_loudly(original):
    raw = MGCHEATS_BEATEN_TRACK.replace(original, b"notknown")

    with pytest.raises(ValueError):
        MGCheatsField().extract([raw])


def test_missing_mgcheats_returns_empty_defaults():
    field = MGCheatsField()
    expected = {
        "room_id": UNKNOWN,
        "room_id_index": 0,
        "room_name": UNKNOWN,
        "room_name_index": 0,
        "fighting": 0,
        "dark": 0,
        "glowing": 0,
        "asleep": 0,
        "gifted": 0,
        "here": (),
        "players": (),
        "mobiles": (),
    }

    np.testing.assert_equal(field.empty(), expected)
    np.testing.assert_equal(field.extract([b""]), expected)


def test_extracts_mgcheats_chunk_with_or_without_ticks():
    field = MGCheatsField()
    obs = field.extract([MGCHEATS_BEATEN_TRACK])

    np.testing.assert_equal(
        obs,
        {
            "room_id": "mtrack2",
            "room_id_index": room_id_to_index("mtrack2"),
            "room_name": "beaten track",
            "room_name_index": room_name_to_index("beaten track"),
            "fighting": 0,
            "dark": 0,
            "glowing": 0,
            "asleep": 0,
            "gifted": 0,
            "here": ("necklace0", "road"),
            "players": (),
            "mobiles": (),
        },
    )
    assert_valid_observation(field, obs)


# Captured from mud2-embedded seed 1234 with Tester observing and Visitor beside them.
MGCHEATS_CELLAR = (
    b"[mgcheats]room_id=mcellar; room_name=cellar; fighting=0; dark=0; glowing=0; asleep=0; gifted=0; "
    b"here=[visitor, door, wall, rat16, rat17, rat18, brand40]; players=[visitor]; mobiles=[rat16, rat17, rat18]; "
    b"inventory=[][/mgcheats]\n"
)


def test_players_and_mobiles_repeat_the_creatures_in_here():
    field = MGCheatsField()
    obs = field.extract([MGCHEATS_CELLAR])

    assert obs["here"] == ("visitor", "door", "wall", "rat16", "rat17", "rat18", "brand40")
    assert obs["players"] == ("visitor",)
    assert obs["mobiles"] == ("rat16", "rat17", "rat18")
    assert_valid_observation(field, obs)


def test_printable_names_fit_the_mgcheats_here_space():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"necklace0, road", b"cloth-of-gold, ornate wall")
    field = MGCheatsField()
    obs = field.extract([raw])

    assert obs["here"] == ("cloth-of-gold", "ornate wall")
    assert_valid_observation(field, obs)


def test_a_flag_that_is_not_exactly_0_or_1_fails_loudly():
    # the game prints a flag as the literal 0 or 1, so anything else is a parse fault, not a number to coerce
    raw = MGCHEATS_BEATEN_TRACK.replace(b"fighting=0", b"fighting=01")
    with pytest.raises(ValueError, match="fighting"):
        MGCheatsField().extract([raw])


def test_flags_are_read_as_bits():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"fighting=0; dark=0", b"fighting=1; dark=1")
    field = MGCheatsField()
    obs = field.extract([raw])

    assert (obs["fighting"], obs["dark"], obs["glowing"], obs["asleep"], obs["gifted"]) == (1, 1, 0, 0, 0)
    assert_valid_observation(field, obs)


@pytest.mark.parametrize(
    ("listed", "bare"),
    [
        (b"here=[necklace0, road]", b"here=rain"),
        (b"players=[]", b"players=visitor"),
        (b"mobiles=[]", b"mobiles=rat16"),
    ],
    ids=["here", "players", "mobiles"],
)
def test_lists_must_be_bracketed(listed, bare):
    raw = MGCHEATS_BEATEN_TRACK.replace(listed, bare)
    with pytest.raises(ValueError, match=bare.split(b"=")[0].decode()):
        MGCheatsField().extract([raw])


def test_duplicate_keys_fail_loudly():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"room_id=mtrack2", b"room_id=mtrack2; room_id=groad1")
    with pytest.raises(ValueError, match="duplicate.*room_id"):
        MGCheatsField().extract([raw])


def test_a_block_missing_a_key_the_game_always_prints_fails_loudly():
    raw = MGCHEATS_BEATEN_TRACK.replace(b" gifted=0;", b"")
    with pytest.raises(KeyError, match="gifted"):
        MGCheatsField().extract([raw])


def test_room_names_are_lowercased_to_match_the_room_tables():
    # the game capitalises proper names ("Dally Lane"), the room tables are all lowercase
    raw = MGCHEATS_BEATEN_TRACK.replace(b"room_name=beaten track", b"room_name=Dally Lane")
    obs = MGCheatsField().extract([raw])

    assert obs["room_name"] == "dally lane"
    assert obs["room_name_index"] == room_name_to_index("dally lane") > 0


def test_orangery_colours_are_removed_before_room_name_lookup():
    raw = (
        b"[mgcheats]room_id=gorange; room_name=\x1b[0;33;40morange\x1b[1;37;40mry; fighting=0; "
        b"dark=0; glowing=0; asleep=0; gifted=0; here=[rain]; players=[]; mobiles=[]; inventory=[][/mgcheats]\n"
    )
    field = MGCheatsField()
    obs = field.extract([raw])

    assert obs["room_name"] == "orangery"
    assert obs["room_name_index"] == room_name_to_index("orangery") > 0
    assert obs["here"] == ("rain",)
    assert_valid_observation(field, obs)
