import pytest

from mudgym.db.index import room_name_to_index
from mudgym.envs.fields.mgcheats import MGCheatsField

MGCHEATS_BEATEN_TRACK = (
    b"[mgcheats]room_id=mtrack2; room_name=beaten track; fighting=0; dark=0; glowing=0; "
    b"asleep=0; gifted=0; here=[necklace0, road]; ticks=0; inventory=[][/mgcheats]\r\n"
)


def test_cheats_includes_room_id():
    raw = (
        b"[mgcheats]room_id=groad1; room_name=dally lane; fighting=0; dark=0; "
        b"glowing=0; asleep=0; gifted=0; here=[]; ticks=0[/mgcheats]"
    )

    out = MGCheatsField().extract([raw])

    assert out["room_name"] == "dally lane"
    assert out["room_id"] == "groad1"


def test_missing_mgcheats_returns_empty_defaults():
    """When mgcheats is missing (e.g., episode ended), return the empty defaults."""
    out = MGCheatsField().extract([b""])
    assert out["room_id"] == ""
    assert out["room_name"] == ""
    assert out["here"] == ()


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(MGCHEATS_BEATEN_TRACK, id="ticks-present"),
        pytest.param(MGCHEATS_BEATEN_TRACK.replace(b"ticks=0; ", b""), id="ticks-absent"),
    ],
)
def test_extracts_mgcheats_chunk_with_or_without_ticks(raw):
    field = MGCheatsField()
    out = field.extract([raw])

    assert out["room_id"] == "mtrack2"
    assert out["room_name"] == "beaten track"
    assert out["here"] == ("necklace0", "road")
    assert "ticks" not in out
    assert field.full_space()["here"].contains(out["here"])


def test_printable_names_fit_the_mgcheats_here_space():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"necklace0, road", b"cloth-of-gold, ornate wall")
    field = MGCheatsField()
    out = field.extract([raw])

    assert out["here"] == ("cloth-of-gold", "ornate wall")
    assert field.full_space()["here"].contains(out["here"])


def test_empty_returns_valid_defaults_cheats():
    defaults = MGCheatsField().empty()

    assert "room_name" in defaults
    assert "room_id" in defaults


def test_end_of_turn_marker_matches_the_closing_tag():
    """The [/mgcheats] closing tag can terminate a batch when mgcheats is the final command."""
    response = (
        b"[mgcheats]room_id=groad3; room_name=dally lane; fighting=0; dark=0; glowing=0; "
        b"asleep=0; gifted=0; here=[necklace0, weather, road]; ticks=125; inventory=[][/mgcheats]\r\n"
    )

    assert MGCheatsField.end_of_turn_marker.search(response)
    assert not MGCheatsField.end_of_turn_marker.search(b"You attack Matthew the necromancer.\r\n")


@pytest.mark.parametrize("value", [b"01", b"+1", b"1_0", b"007", b"x", b"200", b"true", b""])
def test_a_flag_that_is_not_exactly_0_or_1_fails_loudly(value):
    # the game prints a flag as the literal 0 or 1, so anything else is a parse fault, not a number to coerce
    raw = MGCHEATS_BEATEN_TRACK.replace(b"fighting=0", b"fighting=" + value)
    with pytest.raises(ValueError, match="fighting"):
        MGCheatsField().extract([raw])


def test_flags_are_read_as_bits():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"fighting=0; dark=0", b"fighting=1; dark=1")
    out = MGCheatsField().extract([raw])

    assert (out["fighting"], out["dark"], out["glowing"], out["asleep"], out["gifted"]) == (1, 1, 0, 0, 0)
    assert all(MGCheatsField().full_space()[key].contains(out[key]) for key in MGCheatsField.BIT_KEYS)


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
    out = MGCheatsField().extract([raw])

    assert out["room_name"] == "dally lane"
    assert out["room_name_index"] == room_name_to_index("dally lane") > 0
