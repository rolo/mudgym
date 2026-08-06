import pytest

from mudgym.featurizers.quickscore import parse_quickscore_name


@pytest.mark.parametrize("column_gap", [b"      ", b"\t"])
def test_parse_quickscore_name_accepts_terminal_and_embedded_column_gaps(column_gap):
    raw_bytes = (
        b"qs,sql,fes,fex,fei\r\n"
        b"\x1b[0;37;40mAlpha the protector\r\n"
        b"eff str 40"
        + column_gap
        + b"eff dex 45"
        + column_gap
        + b"sta \x1b[1;32;40m40\x1b[0;37;40m/\x1b[1;32;40m40\x1b[0;37;40m"
        + column_gap
        + b"pts 200"
        + column_gap
        + b"gam 1\r\n"
    )

    assert parse_quickscore_name(raw_bytes) == "Alpha"
