"""An action is one logical game input line.

A line break inside an action would smuggle an extra wire command past the batch assembly (the
env appends its own auto-command line), so the action charset excludes CR and LF while the
observation text charset keeps its line-break support.
"""

import pytest

from mudgym.envs.specs import ACTION_CHARSET, SINGLE_LINE_CHARSET, TEXT_CHARSET
from tests.scripted import scripted_response


def test_action_charset_excludes_line_breaks_text_charset_keeps_them():
    assert "\n" not in ACTION_CHARSET
    assert "\r" not in ACTION_CHARSET
    assert "\n" in TEXT_CHARSET


def test_charsets_are_seven_bit():
    # the game's entire text database is 7-bit; wire bytes above 0x7F are protocol codes, and
    # the game transliterates them in input (a typed e-acute echoes back as 'i')
    assert "\xe9" not in ACTION_CHARSET
    assert "\xe9" not in TEXT_CHARSET
    assert "\xe9" not in SINGLE_LINE_CHARSET
    assert "\n" not in SINGLE_LINE_CHARSET


@pytest.mark.parametrize("action", ["look\nsay hello", "look\rsay hello", "look\r\nsay hello"])
def test_actions_with_line_breaks_are_outside_the_action_space(scripted_env, action):
    assert scripted_env.action_space.contains(action) is False


@pytest.mark.parametrize("action", ["look", "get sword,say hi", "say hello", '"hello'])
def test_ordinary_and_speech_actions_remain_valid(scripted_env, action):
    assert scripted_env.action_space.contains(action) is True


@pytest.mark.parametrize("action", ["look\nnorth", "look\rnorth", "look\r\nnorth"])
def test_step_rejects_line_breaks_before_any_transport_send(scripted_env, action):
    scripted_env.reset()
    connection = scripted_env.unwrapped.session.connection
    commands_before = list(connection.commands)

    with pytest.raises(ValueError, match="single logical input line"):
        scripted_env.step(action)

    assert connection.commands == commands_before


@pytest.mark.parametrize("action", ["say hello\nCheerio!", "moan hello\nCheerio!"])
def test_line_break_cannot_manufacture_a_game_over_wire_line(scripted_env, action):
    scripted_env.reset()
    connection = scripted_env.unwrapped.session.connection
    commands_before = list(connection.commands)

    with pytest.raises(ValueError, match="single logical input line"):
        scripted_env.step(action)

    assert connection.commands == commands_before


def test_a_high_wire_byte_in_game_text_fails_loudly(scripted_env_factory):
    # bytes above 0x7F are protocol codes, never text; silently decoding one would mask the leak
    body_env = scripted_env_factory(
        responses={"look": scripted_response("look,sql,fes,fex,fei").replace(b"dusty road", b"caf\xe9 road")}
    )
    body_env.reset()

    with pytest.raises(ValueError, match="Invalid bytes"):
        body_env.step("look")


def test_an_ordinary_step_observation_fits_the_observation_space(scripted_env_factory):
    env = scripted_env_factory()
    env.reset()

    obs, reward, terminated, truncated, info = env.step("look")

    assert env.observation_space.contains(obs)
