import pytest

from mudgym.featurizers.ansi import expand_tabs, strip_ansi


def test_strips_sgr_and_csi_sequences():
    assert strip_ansi(b"\x1b[1;32;40m64\x1b[0;37;40m \x1b[2Jclear") == b"64 clear"


def test_two_byte_escapes_are_stripped_and_nothing_after_them_is_claimed():
    # ESC ] is a two-byte escape here: the game never emits OSC, so a payload would stay in the text
    assert strip_ansi(b"\x1b]0;title\x07 text \x1bMsaved") == b"0;title\x07 text saved"


@pytest.mark.parametrize(
    ("raw", "expanded"),
    [
        # How the game laid out inventory, score and exits when it still expanded tabs itself.
        pytest.param(b"\tnothing.", b"        nothing.", id="indent"),
        pytest.param(b"name:\t\tTester", b"name:           Tester", id="columns"),
        pytest.param(
            b"\x1b[0;37;40mnorth:\t\t\x1b[1;32;40mMiddle",
            b"\x1b[0;37;40mnorth:          \x1b[1;32;40mMiddle",
            id="colours-take-no-columns",
        ),
        pytest.param(b"abcdefgh\tx\nab\tx", b"abcdefgh        x\nab      x", id="each-line-starts-at-column-0"),
    ],
)
def test_tabs_expand_to_the_next_eight_column_stop(raw, expanded):
    assert expand_tabs(raw) == expanded
