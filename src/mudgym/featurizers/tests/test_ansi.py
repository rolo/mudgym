from mudgym.featurizers.ansi import strip_ansi


def test_strips_sgr_and_csi_sequences():
    assert strip_ansi(b"\x1b[1;32;40m64\x1b[0;37;40m \x1b[2Jclear") == b"64 clear"


def test_two_byte_escapes_are_stripped_and_nothing_after_them_is_claimed():
    # ESC ] is a two-byte escape here: the game never emits OSC, so a payload would stay in the text
    assert strip_ansi(b"\x1b]0;title\x07 text \x1bMsaved") == b"0;title\x07 text saved"
