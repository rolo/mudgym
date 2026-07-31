"""Text policy: Latin-1 is the byte-to-text mapping (never UTF-8), game text is 7-bit, and wire
bytes above 0x7F are protocol codes that must fail loudly in text paths while still rendering in
diagnostics."""

import pytest

from mudgym.featurizers.strings import decode_text_bytes, decode_wire_bytes, encode_command_bytes


def test_ascii_text_round_trips():
    text = "You are standing on a dusty road with rising ground both to the north and south."
    assert decode_text_bytes(text.encode("ascii")) == text


def test_ansi_escapes_pass_through():
    assert decode_text_bytes(b"\x1b[1;37;40mDally Lane\x1b[0m") == "\x1b[1;37;40mDally Lane\x1b[0m"


def test_high_bytes_are_leaked_protocol_not_text():
    # eg the fecode marker bytes \x9b\xff; decoding them as text would mask the leak
    with pytest.raises(ValueError, match="at byte 5"):
        decode_text_bytes(b"leak \x9b\xff here")


def test_every_high_byte_is_rejected_in_text_paths():
    # a lone \xff (the fecode pair's second byte, split from its \x9b) must not decode as text
    with pytest.raises(ValueError):
        decode_text_bytes(b"caf\xe9")
    with pytest.raises(ValueError):
        decode_text_bytes(b"\xff")


def test_wire_decode_is_total_for_diagnostics():
    # error paths must always be able to render the wire without raising themselves
    assert decode_wire_bytes(b"leak \x9b\xff here") == "leak \x9b\xff here"
    assert decode_wire_bytes(bytes(range(256))) == "".join(chr(code) for code in range(256))


def test_ascii_commands_encode_unchanged():
    assert encode_command_bytes("say cafe") == b"say cafe"


def test_non_ascii_commands_fail_before_the_wire():
    # the game transliterates high bytes rather than echoing them, so the exact-echo anchor
    # would never match; fail clearly before sending instead
    with pytest.raises(ValueError, match="outside ASCII"):
        encode_command_bytes("say caf\xe9")
    with pytest.raises(ValueError, match="outside ASCII"):
        encode_command_bytes("say €100")
