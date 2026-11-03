import pytest

from mudgym.featurizers.quickscore import parse_quickscore


def stamina_column(current: bytes, maximum: bytes) -> bytes:
    """The stamina column as the game colours it: bright values around a plain slash."""
    return b"sta \x1b[1;32;40m" + current + b"\x1b[0;37;40m/\x1b[1;32;40m" + maximum + b"\x1b[0;37;40m"


def quickscore_reply(name_line: bytes, stats_columns: list[bytes], column_gap: bytes) -> bytes:
    """A quickscore reply: the command echo, the coloured name line, then the stats columns."""
    return b"qs,sql,fes,fex,fei\r\n" + b"\x1b[0;37;40m" + name_line + b"\r\n" + column_gap.join(stats_columns) + b"\r\n"


FIGHTER_COLUMNS = [b"eff str 40", b"eff dex 45", stamina_column(b"40", b"40"), b"pts 200", b"gam 1"]
MAGIC_USER_COLUMNS = [
    b"eff str 68",
    b"eff dex 59",
    stamina_column(b"28", b"48"),
    b"mag 48",
    b"pts 140,000",
    b"gam 2",
]


@pytest.mark.parametrize("column_gap", [b"      ", b"\t"])
def test_parse_quickscore_accepts_terminal_and_embedded_column_gaps(column_gap):
    raw_bytes = quickscore_reply(b"Alpha the protector", FIGHTER_COLUMNS, column_gap)

    assert parse_quickscore(raw_bytes) == ("Alpha", 200)


@pytest.mark.parametrize(
    "name_line, stats_columns, expected_points",
    [
        # a novice's name line is the bare name alone
        (b"Alpha", FIGHTER_COLUMNS, 200),
        # prefixes the game issues sit inside the title, and a magic user gets the mag column before pts
        (b"Alpha the sorcerised sorcerer", MAGIC_USER_COLUMNS, 140_000),
        # Brother/Sister put the title first and are always magic-users, so both quirks land on one line
        (b"Brother Alpha", MAGIC_USER_COLUMNS, 140_000),
    ],
)
def test_parse_quickscore_reads_the_persona_out_of_its_full_name(name_line, stats_columns, expected_points):
    raw_bytes = quickscore_reply(name_line, stats_columns, b"      ")

    assert parse_quickscore(raw_bytes) == ("Alpha", expected_points)


def test_parse_quickscore_ignores_colour_inside_the_name_line():
    raw_bytes = quickscore_reply(b"Sir \x1b[1;37;40mAlpha\x1b[0;37;40m", FIGHTER_COLUMNS, b"      ")

    assert parse_quickscore(raw_bytes) == ("Alpha", 200)


def test_parse_quickscore_fails_without_a_points_column():
    raw_bytes = quickscore_reply(
        b"Alpha", [b"eff str 40", b"eff dex 45", stamina_column(b"40", b"40"), b"gam 1"], b"  "
    )

    with pytest.raises(ValueError, match="quickscore"):
        parse_quickscore(raw_bytes)
