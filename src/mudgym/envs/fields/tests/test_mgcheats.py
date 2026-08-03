import pytest

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


def test_hyphenated_identifiers_fit_the_mgcheats_space():
    raw = MGCHEATS_BEATEN_TRACK.replace(b"necklace0, road", b"cloth-of-gold")
    field = MGCheatsField()
    out = field.extract([raw])

    assert out["here"] == ("cloth-of-gold",)
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
