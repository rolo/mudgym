"""An action is one logical game input line.

A line break inside an action would smuggle an extra wire command past Session's line framing (the
observation-command line is sent separately), so the action charset excludes CR and LF while the
observation text charset keeps its line-break support.
"""

import pytest

from mudgym.envs.specs import ACTION_CHARSET, SINGLE_LINE_CHARSET, TEXT_CHARSET
from tests.scripted import scripted_response


@pytest.mark.parametrize("charset", [ACTION_CHARSET, SINGLE_LINE_CHARSET], ids=["action", "single-line"])
def test_single_line_charsets_exclude_line_breaks(charset):
    assert not {"\r", "\n"}.intersection(charset)


def test_text_charset_preserves_newlines():
    assert "\n" in TEXT_CHARSET


@pytest.mark.parametrize(
    "charset", [ACTION_CHARSET, TEXT_CHARSET, SINGLE_LINE_CHARSET], ids=["action", "text", "single-line"]
)
def test_charsets_are_seven_bit(charset):
    # High wire bytes are protocol codes, not text.
    assert all(ord(character) < 128 for character in charset)


@pytest.mark.parametrize("action", ["look\nsay hello", "look\rsay hello", "look\r\nsay hello"])
def test_actions_with_line_breaks_are_outside_the_action_space(scripted_env, action):
    assert scripted_env.action_space.contains(action) is False


@pytest.mark.parametrize("action", ["look", "get sword,say hi", "say hello", '"hello'])
def test_ordinary_and_speech_actions_remain_valid(scripted_env, action):
    assert scripted_env.action_space.contains(action) is True


@pytest.mark.parametrize(
    "original_bytes, replacement_bytes",
    [(b"dusty road", b"caf\xe9 road"), (b"necklace0", b"necklace\xe9")],
)
def test_a_high_wire_byte_in_game_text_fails_loudly(scripted_env_factory, original_bytes, replacement_bytes):
    # bytes above 0x7F are protocol codes, never text; silently decoding one would mask the leak
    env = scripted_env_factory(
        responses={"look": scripted_response(["look", "sql,fes,fex,fei"]).replace(original_bytes, replacement_bytes)}
    )
    env.reset()

    with pytest.raises(ValueError, match="Invalid text byte"):
        env.step("look")
