import time

import pytest

from mudgym.featurizers.responses import contains_echo, split_on_echo, split_on_prompt


@pytest.fixture
def raw_bytes():
    return b"dance,fes,fex,fei\r\n\x1b[0;33;40mOK, Janet the protector \x1b[1;33;40mdances.\x1b[0;33;40m\x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m\x1b[1;32;40m60\x1b[0;37;40m \x1b[1;32;40m60\x1b[0;37;40m 53 61 56 59 0 60 0377 N N N N 46 F\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mup down out swampward south southeast east north\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mcoracle\r\nvial1\r\nring0\r\nbrand39\r\ncoronet\r\n========\r\nkey50\r\ncloth-of-gold\r\nbroadsword\r\n\x1b[1;37;40m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"


@pytest.fixture
def raw_bytes_berk():
    return b'\x1b[1;37;40mmove south,sql,fes,fex,fei\r\n\x1b[32mDally Lane\x1b[37m.\r\n\x1b[0;32;40mIt is raining. \x1b[1;37;40m\x1b[36mA splendid necklace lies on the ground. \x1b[37m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mThe place known as "\x1b[1;32;40mDally Lane\x1b[0;37;40m" contains \x1b[1;36;40mthe necklace\x1b[0;37;40m, \x1b[32mrain\x1b[37m, \x1b[31mAlexander the protector\x1b[37m and \x1b[32mthe road\x1b[37m.\r\nYou are carrying the following:\r\n        nothing.\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m\x1b[1;32;40m75\x1b[0;37;40m \x1b[1;32;40m75\x1b[0;37;40m 52 52 53 53 0 75 0200 N N N N 53 R\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mup out swampward southwest south west east north\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40mnecklace0\r\n========\r\n\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m'


def test_split_response_chunks(raw_bytes):
    expected = [
        b"dance,fes,fex,fei\r\n\x1b[0;33;40mOK, Janet the protector \x1b[1;33;40mdances.\x1b[0;33;40m\x1b[1;37;40m\r\n",
        b"60\x1b[0;37;40m \x1b[1;32;40m60\x1b[0;37;40m 53 61 56 59 0 60 0377 N N N N 46 F\r\n",
        b"up down out swampward south southeast east north\r\n",
        b"coracle\r\nvial1\r\nring0\r\nbrand39\r\ncoronet\r\n========\r\nkey50\r\ncloth-of-gold\r\nbroadsword\r\n\x1b[1;37;40m\r\n",
    ]
    assert [chunk for chunk in split_on_prompt(raw_bytes) if chunk] == expected


def test_split_on_prompt_keeps_empty_command_slots():
    raw = b"*score\r\n*"

    assert split_on_prompt(raw) == [b"", b"score\r\n"]
    assert [chunk for chunk in split_on_prompt(raw) if chunk] == [b"score\r\n"]


def test_split_on_echo_preserves_pre_echo_output_before_prompt_marker():
    prompt = b"\x1b[1;34;40m*\x1b[0m"
    raw = b"The dragonfly has just flown away.\r\n" + prompt + b"look,sql,fes,fex,fei\r\nDally Lane.\r\n"

    pre_echo, post_echo = split_on_echo(raw, "look,sql,fes,fex,fei")

    assert pre_echo == b"The dragonfly has just flown away.\r\n"
    assert post_echo == b"Dally Lane.\r\n"


def test_contains_echo_stays_linear_on_many_inline_prompt_markers():
    # Regression: a plain greedy ECHO_PREFIX backtracks exponentially on a run of inline
    # prompt markers that is not followed by the command echo (~12x per marker), hanging
    # bytes_to_observation on live payloads. The possessive quantifier keeps it linear.
    marker = b"\x1b[1;34;40m*\x1b[0m"
    raw = marker * 50 + b"unrelated game text\r\n"

    start = time.perf_counter()
    result = contains_echo(raw, "look,sql,fes,fex,fei")
    elapsed = time.perf_counter() - start

    assert result is False
    assert elapsed < 1.0, f"contains_echo took {elapsed:.2f}s -- regex is backtracking"


def test_split_response_chunks_berk(raw_bytes_berk):
    expected = [
        b"\x1b[1;37;40mmove south,sql,fes,fex,fei\r\n\x1b[32mDally Lane\x1b[37m.\r\n\x1b[0;32;40mIt is raining. \x1b[1;37;40m\x1b[36mA splendid necklace lies on the ground. \x1b[37m\r\n",
        b'The place known as "\x1b[1;32;40mDally Lane\x1b[0;37;40m" contains \x1b[1;36;40mthe necklace\x1b[0;37;40m, \x1b[32mrain\x1b[37m, \x1b[31mAlexander the protector\x1b[37m and \x1b[32mthe road\x1b[37m.\r\nYou are carrying the following:\r\n        nothing.\r\n',
        b"75\x1b[0;37;40m \x1b[1;32;40m75\x1b[0;37;40m 52 52 53 53 0 75 0200 N N N N 53 R\r\n",
        b"up out swampward southwest south west east north\r\n",
        b"necklace0\r\n========\r\n",
    ]
    assert [chunk for chunk in split_on_prompt(raw_bytes_berk) if chunk] == expected
