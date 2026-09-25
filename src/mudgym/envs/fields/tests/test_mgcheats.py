import numpy as np
import pytest

from mudgym.db.index import UNKNOWN, room_id_to_index, room_name_to_index
from mudgym.db.rooms import ROOM_IDS, ROOM_NAMES
from mudgym.envs.fields.mgcheats import MGCheatsField
from mudgym.envs.fields.tests.helpers import assert_valid_observation

MGCHEATS_BEATEN_TRACK = (
    b"[mgcheats]room_id=mtrack2; room_name=beaten track; fighting=0; dark=0; glowing=0; "
    b"asleep=0; gifted=0; here=[necklace0, road]; ticks=0; inventory=[][/mgcheats]\r\n"
)


@pytest.mark.parametrize(
    ("room_id", "room_name", "index_key", "expected_index"),
    [
        pytest.param("abcell", "abbot's cell", "room_id_index", 1, id="first-room-id"),
        pytest.param("monch", "abbatial church", "room_name_index", 1, id="first-room-name"),
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
@pytest.mark.parametrize("value", [b"notknown", b""], ids=["notknown", "empty"])
def test_unknown_or_empty_rooms_fail_loudly(original, value):
    raw = MGCHEATS_BEATEN_TRACK.replace(original, value)

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
    }

    np.testing.assert_equal(field.empty(), expected)
    np.testing.assert_equal(field.extract([b""]), expected)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(MGCHEATS_BEATEN_TRACK, id="ticks-present"),
        pytest.param(MGCHEATS_BEATEN_TRACK.replace(b"ticks=0; ", b""), id="ticks-absent"),
    ],
)
def test_extracts_mgcheats_chunk_with_or_without_ticks(raw):
    field = MGCheatsField()
    obs = field.extract([raw])

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
        },
    )
    assert_valid_observation(field, obs)


def test_printable_names_fit_the_mgcheats_here_space():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"necklace0, road", b"cloth-of-gold, ornate wall")
    field = MGCheatsField()
    obs = field.extract([raw])

    assert obs["here"] == ("cloth-of-gold", "ornate wall")
    assert_valid_observation(field, obs)


@pytest.mark.parametrize("value", [b"01", b"+1", b"1_0", b"007", b"x", b"200", b"true", b""])
def test_a_flag_that_is_not_exactly_0_or_1_fails_loudly(value):
    # the game prints a flag as the literal 0 or 1, so anything else is a parse fault, not a number to coerce
    raw = MGCHEATS_BEATEN_TRACK.replace(b"fighting=0", b"fighting=" + value)
    with pytest.raises(ValueError, match="fighting"):
        MGCheatsField().extract([raw])


def test_flags_are_read_as_bits():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"fighting=0; dark=0", b"fighting=1; dark=1")
    field = MGCheatsField()
    obs = field.extract([raw])

    assert (obs["fighting"], obs["dark"], obs["glowing"], obs["asleep"], obs["gifted"]) == (1, 1, 0, 0, 0)
    assert_valid_observation(field, obs)


def test_here_must_be_a_bracketed_list():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"here=[necklace0, road]", b"here=rain")
    with pytest.raises(ValueError, match="here"):
        MGCheatsField().extract([raw])


def test_duplicate_keys_fail_loudly():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"room_id=mtrack2", b"room_id=mtrack2; room_id=groad1")
    with pytest.raises(ValueError, match="duplicate.*room_id"):
        MGCheatsField().extract([raw])


def test_a_block_missing_a_key_the_game_always_prints_fails_loudly():
    raw = MGCHEATS_BEATEN_TRACK.replace(b" gifted=0;", b"")
    with pytest.raises(KeyError, match="gifted"):
        MGCheatsField().extract([raw])


def test_a_pair_without_an_equals_sign_fails_loudly():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"dark=0", b"dark")
    with pytest.raises(ValueError):
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
        b"dark=0; glowing=0; asleep=0; gifted=0; here=[rain]; inventory=[][/mgcheats]\n"
    )
    field = MGCheatsField()
    obs = field.extract([raw])

    assert obs["room_name"] == "orangery"
    assert obs["room_name_index"] == room_name_to_index("orangery") > 0
    assert obs["here"] == ("rain",)
    assert_valid_observation(field, obs)
