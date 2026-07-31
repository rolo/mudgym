import re

import pytest

from mudgym.connections.prompts import Prompt, marker_up_to_next_prompt


def test_marker_up_to_next_prompt_preserves_the_marker_flags():
    """The wire form is rebuilt from the marker's pattern, so compile-time flags must carry over."""
    marker = re.compile(rb"a marker", re.IGNORECASE)

    wire_pattern = marker_up_to_next_prompt(marker)

    assert wire_pattern.flags & re.IGNORECASE


class TestGameOverPromptsRejectPlayerAuthoredText:
    """The game-over lines carry nothing but the message, so quoted speech and command echoes
    (which put the words mid-line) must not read as an episode end."""

    def test_the_real_quit_cheerio_line_matches(self):
        # captured from the live game: the buffer ends right after the word when pexpect matches
        assert Prompt.GAME_OVER_QUIT_CHEERIO.value.search(b"quit\r\nCheerio!")

    def test_the_real_episode_points_line_matches(self):
        # the seagull death capture ends in a bare \r with no trailing newline
        assert Prompt.GAME_OVER_EPISODE_POINTS.value.search(b"Overall, you scored 189 points this game.\r")
        assert Prompt.GAME_OVER_EPISODE_POINTS.value.search(b"Overall, you lost 12,800 points this game.\r\n")

    def test_the_real_swearing_kill_matches(self):
        raw_bytes = (
            b"fuck,sql,fes,fex,fei\r\nIn order to keep the game uncorrupted, you have been killed.\r\n"
            b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n"
        )
        assert Prompt.GAME_OVER_KILLED_FOR_SWEARING.value.search(raw_bytes)

    def test_spoken_cheerio_does_not_match(self):
        spoken = b'\x1b[0;33;40mBriana the protector says "\x1b[1;33;40mCheerio!\x1b[0;33;40m".\x1b[1;37;40m\r\n'
        assert Prompt.GAME_OVER_QUIT_CHEERIO.value.search(spoken) is None

    def test_echoed_cheerio_does_not_match(self):
        echo = b"say Cheerio!\r\n"
        assert Prompt.GAME_OVER_QUIT_CHEERIO.value.search(echo) is None

    def test_spoken_episode_points_does_not_match(self):
        spoken = b'Dumbo the novice says "\x1b[1;33;40mOverall, you scored 999 points this game.\x1b[0;33;40m".\r\n'
        assert Prompt.GAME_OVER_EPISODE_POINTS.value.search(spoken) is None

    def test_spoken_not_updating_persona_does_not_match(self):
        spoken = b'Dumbo the novice says "\x1b[1;33;40mNot updating persona.\x1b[0;33;40m".\r\n'
        assert Prompt.GAME_OVER_NOT_UPDATING_PERSONA.value.search(spoken) is None


@pytest.mark.parametrize(
    ("prompt", "genuine_wire", "player_echo"),
    [
        (Prompt.OPTION, b"\r\n\x1b[1;37;40mOption:", b"say Option:\r\n"),
        (
            Prompt.PERSONA_AVAILABLE,
            b"\r\n\x1b[1;37;40mBy what name shall I call you (Q to quit)?",
            b"say By what name shall I call you (Q to quit)?\r\n",
        ),
        (
            Prompt.PERSONA_NAME,
            b"\r\n\x1b[1;37;40mWhat shall I call you instead?",
            b"say What shall I call you instead?\r\n",
        ),
        (
            Prompt.PERSONA_SEX,
            b"\r\n\x1b[1;37;40mWhat sex do you wish to be?",
            b"say What sex do you wish to be?\r\n",
        ),
        (
            Prompt.SUPERSEDE,
            b"\r\n\x1b[1;37;40mDo you want to supersede this other session?",
            b"say Do you want to supersede this other session?\r\n",
        ),
        (Prompt.SESSION_DYING, b"\r\n\x1b[1;37;40mSession is dying", b"say Session is dying\r\n"),
        (Prompt.EXAMINE, b"\r\n\x1b[1;37;40mEXAMINE>", b"say EXAMINE>\r\n"),
        (Prompt.LIBRARY, b"\r\n\x1b[1;37;40mLIBRARY>", b"say LIBRARY>\r\n"),
        (
            Prompt.PAGER,
            b"\r\n\x1b[1;37;40m[Return to continue, S to stop]\x1b[0m",
            b"say [Return to continue, S to stop]\r\n",
        ),
    ],
)
def test_input_prompts_require_a_raw_wire_line_start(prompt, genuine_wire, player_echo):
    assert prompt.value.search(genuine_wire)
    assert prompt.value.search(player_echo) is None


def test_option_prompt_accepts_the_real_uncoloured_initial_menu_shape():
    assert Prompt.OPTION.value.search(b"[Mail file too busy to check]\r\nOption: ")


def test_option_prompt_accepts_screen_clear_and_cursor_controls():
    assert Prompt.OPTION.value.search(b"\r\n\x1b[2J\x1b[H\x1b[1;37;40mOption: ")


def test_colour_wrapped_spoken_option_is_not_a_trusted_input_prompt():
    spoken = b'\x1b[0;33;40mDumbo the novice says "\x1b[1;33;40mOption:\x1b[0;33;40m".\x1b[1;37;40m\r\n'

    assert Prompt.OPTION.value.search(spoken) is None
