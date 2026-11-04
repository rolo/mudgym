"""Game text and player commands use strict ASCII."""

import pytest

from mudgym.featurizers.strings import (
    decode_text_bytes,
    decode_text_lines,
    encode_command_bytes,
)


def test_ansi_escapes_pass_through():
    assert decode_text_bytes(b"\x1b[1;37;40mDally Lane\x1b[0m") == "\x1b[1;37;40mDally Lane\x1b[0m"


def test_high_bytes_are_leaked_protocol_not_text():
    # eg the fecode marker bytes \x9b\xff, which decoding as text would mask
    with pytest.raises(ValueError, match="at byte 5"):
        decode_text_bytes(b"leak \x9b\xff here")


def test_non_ascii_commands_fail_before_the_wire():
    # the game transliterates high bytes rather than echoing them, so the exact-echo anchor would never match. Fail
    # clearly before sending instead.
    with pytest.raises(ValueError, match="outside ASCII"):
        encode_command_bytes("say caf\xe9")
    with pytest.raises(ValueError, match="outside ASCII"):
        encode_command_bytes("say €100")


def test_text_lines_split_on_newlines_and_strip_ascii_whitespace_only():
    # bytes.split cannot invent line breaks at separator controls, and bytes.strip leaves them intact
    assert decode_text_lines(b"a \r\n\tb\n\x0cc\x1cd\x1f\r\n") == ["a", "b", "c\x1cd\x1f", ""]
