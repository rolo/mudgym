import pytest

from mudgym.featurizers.quickscore import parse_quickscore_name


def stamina_column(current: bytes, maximum: bytes) -> bytes:
    """The stamina column as the game colours it: bright values around a plain slash."""
    return b"sta \x1b[1;32;40m" + current + b"\x1b[0;37;40m/\x1b[1;32;40m" + maximum + b"\x1b[0;37;40m"


def quickscore_reply(name_line: bytes, stats_columns: list[bytes], column_gap: bytes) -> bytes:
    """A quickscore reply: the command echo, the coloured name line, then the stats columns."""
    return b"qs,sql,fes,fex,fei\r\n" + b"\x1b[0;37;40m" + name_line + b"\r\n" + column_gap.join(stats_columns) + b"\r\n"


@pytest.mark.parametrize("column_gap", [b"      ", b"\t"])
def test_parse_quickscore_name_accepts_terminal_and_embedded_column_gaps(column_gap):
    raw_bytes = quickscore_reply(
        b"Alpha the protector",
        [b"eff str 40", b"eff dex 45", stamina_column(b"40", b"40"), b"pts 200", b"gam 1"],
        column_gap,
    )

    assert parse_quickscore_name(raw_bytes) == "Alpha"


FIGHTER_COLUMNS = [b"eff str 40", b"eff dex 45", stamina_column(b"40", b"40"), b"pts 200", b"gam 1"]
MAGIC_USER_COLUMNS = [
    b"eff str 68",
    b"eff dex 59",
    stamina_column(b"28", b"48"),
    b"mag 48",
    b"pts 140,000",
    b"gam 2",
]


@pytest.mark.parametrize(
    "name_line, stats_columns, expected_name",
    [
        # a novice's name line is the bare name alone
        (b"Alpha", FIGHTER_COLUMNS, "Alpha"),
        # prefixes the game issues sit inside the title, not around the name
        (b"Alpha the dragon-slaying warrior", FIGHTER_COLUMNS, "Alpha"),
        (b"Alpha the sorcerised sorcerer", MAGIC_USER_COLUMNS, "Alpha"),
        # Sir/Lady put the title first, and are always non-magic-users
        (b"Sir Alpha", FIGHTER_COLUMNS, "Alpha"),
        (b"Lady Alpha", FIGHTER_COLUMNS, "Alpha"),
        # Brother/Sister do too, and are always magic-users, so both quirks land on one line
        (b"Brother Alpha", MAGIC_USER_COLUMNS, "Alpha"),
        (b"Sister Alpha", MAGIC_USER_COLUMNS, "Alpha"),
        # a prefix on a reversed title is capitalised in front of it
        (b"Awkward Lady Alpha", FIGHTER_COLUMNS, "Alpha"),
    ],
)
def test_parse_quickscore_name_reads_the_persona_out_of_its_full_name(name_line, stats_columns, expected_name):
    raw_bytes = quickscore_reply(name_line, stats_columns, b"      ")

    assert parse_quickscore_name(raw_bytes) == expected_name


def test_parse_quickscore_name_ignores_colour_inside_the_name_line():
    raw_bytes = quickscore_reply(b"Sir \x1b[1;37;40mAlpha\x1b[0;37;40m", FIGHTER_COLUMNS, b"      ")

    assert parse_quickscore_name(raw_bytes) == "Alpha"


@pytest.mark.parametrize("column_gap", [b"      ", b"\t"])
def test_parse_quickscore_name_accepts_the_magic_user_mag_column(column_gap):
    # a magic user's stats line carries an extra mag column between stamina and points
    raw_bytes = quickscore_reply(
        b"Alpha the mage",
        [
            b"eff str 68",
            b"eff dex 59",
            stamina_column(b"28", b"48"),
            b"mag 48",
            b"pts 140,000",
            b"gam 2",
        ],
        column_gap,
    )

    assert parse_quickscore_name(raw_bytes) == "Alpha"
