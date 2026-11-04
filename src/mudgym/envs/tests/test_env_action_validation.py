"""An action is one logical game input line.

A line break inside an action would smuggle an extra wire command past Session's line framing (the observation-command
line is sent separately), so the action charset excludes CR and LF while the observation text charset keeps its
line-break support.
"""

import pytest

from tests.scripted import scripted_response


@pytest.mark.parametrize("action", ["look\nsay hello", "look\rsay hello"])
def test_actions_with_line_breaks_are_outside_the_action_space(scripted_env, action):
    assert scripted_env.action_space.contains(action) is False


@pytest.mark.parametrize(
    "original_bytes, replacement_bytes",
    [(b"dusty road", b"caf\xe9 road"), (b"necklace0", b"necklace\xe9")],
)
def test_a_high_wire_byte_in_game_text_fails_loudly(scripted_env_factory, original_bytes, replacement_bytes):
    # bytes above 0x7F are protocol codes, never text, so silently decoding one would mask the leak
    env = scripted_env_factory(
        responses={"look": scripted_response(["look", "sql,fes,fex,fei"]).replace(original_bytes, replacement_bytes)}
    )
    env.reset()

    with pytest.raises(ValueError, match="Invalid text byte"):
        env.step("look")
