import pytest

from mudgym.connections.prompts import COMMAND_REJECTION_LINES, Prompt


@pytest.mark.parametrize(
    "message",
    [
        b"I made sense of some of that:",
        b"I made no sense of that:",
        b'I don\'t know the word "frobnicate".',
        b"I don't know to what \"goat\" you're referring.",
        b"Your command is too long for me, sorry!",
    ],
)
def test_command_rejection_lines_match_system_lines_but_not_spoken_copies(message):
    genuine_response = b"\x1b[0;37;40m" + message + b"\r\n"
    spoken_copy = b'Raymond the protector says "' + message + b'".\r\n'

    assert any(pattern.search(genuine_response) for pattern in COMMAND_REJECTION_LINES)
    assert not any(pattern.search(spoken_copy) for pattern in COMMAND_REJECTION_LINES)


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

    def test_the_real_combat_death_matches(self):
        raw_bytes = (
            b"\x1b[0;30;41mYou feel your very existence severed from you...\r\n"
            b"You have been killed by the vampire.\x1b[1;37;40m\r\n"
            b"\x1b[0;31;40mNot updating persona.\x1b[1;37;40m\r"
        )
        assert Prompt.GAME_OVER_NOT_UPDATING_PERSONA.value.search(raw_bytes)

    def test_plain_not_updating_persona_does_not_match(self):
        assert Prompt.GAME_OVER_NOT_UPDATING_PERSONA.value.search(b"Not updating persona.\r\n") is None

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
