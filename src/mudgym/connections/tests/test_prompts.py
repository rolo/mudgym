import re

import pytest

from mudgym.connections.prompts import (
    INVALID_COMMAND_PROMPTS,
    Prompt,
    marker_up_to_next_prompt,
    regex_up_to_next_prompt,
)

# the game prompt as captured from the wire: blue star, then the colour the input echo will use
GAME_PROMPT = b"\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"


class TestGamePromptShapes:
    @pytest.mark.parametrize(
        "wire",
        [
            b"\r\n" + GAME_PROMPT,  # bold mortal star as captured, dangling
            b"\r\n" + GAME_PROMPT + b"fei\r\n",  # continuing with the echo of the next command
            GAME_PROMPT,  # at the very start of a read window
            b"\r\n(*)",  # invisible mortal
            b"\r\n\x1b[1;34;40m((*))\x1b[0;34;40m",  # colour-wrapped double invisibility
            b"\r\n(((*)))",
            b"\r\n----*",  # wiz
            b"\r\n(----*)",
            b"\r\n((----*))",
            b"\r\n(((----*)))",  # triple invisible wiz
        ],
    )
    def test_genuine_prompt_shapes_match(self, wire):
        assert Prompt.GAME.value.search(wire)

    @pytest.mark.parametrize(
        "wire",
        [
            b"\r\n\x1b[34m*\x1b[37mAlexis\r\n",  # login menu reprint: a plain star, not in game
            b"say ----*\r\n",  # command echo puts the shape mid-line
            b'Dumbo the novice says "\x1b[1;33;40m----*\x1b[0;33;40m".\r\n',  # spoken copy
            b"\r\n----------\r\n",  # a complete divider line
            b"\r\n---- Welcome to MUD ----\r\n",  # a banner rule
            b"\r\n((*))\r\n",  # prompt-shaped narrative: a complete line is never a prompt
        ],
    )
    def test_menu_reprints_echoes_and_narrative_do_not_match(self, wire):
        assert Prompt.GAME.value.search(wire) is None


class TestUpToNextPromptStopsAtAMidStreamPrompt:
    """A genuine prompt is never a complete line: it dangles awaiting input, or continues with the
    echo of whatever the player types next. When responses run ahead of the reader, the terminator
    must stop at the first such prompt rather than extending to the end of the received data and
    swallowing the following command's echo and marker."""

    # captured from a flaky docker_run read window: the rejection of the first line, its
    # prompt, and the whole fei exchange all arrived in one read
    RUN_AHEAD = (
        b'I made no sense of that: "not updating persona", but it\'s an excluding preposition.\r\n'
        + GAME_PROMPT
        + b"fei\r\n"
        + b"\x1b[0;37;40m========\r\n"
        + b"tea\r\n"
        + b"\x1b[1;37;40m"
        + GAME_PROMPT
    )

    def test_a_rejection_stops_at_its_own_prompt_not_at_the_end_of_data(self):
        pattern = regex_up_to_next_prompt(b"I made no sense of that:")

        match = pattern.search(self.RUN_AHEAD)

        assert match
        assert match.end() <= self.RUN_AHEAD.index(b"fei")

    def test_a_dangling_prompt_at_the_end_of_the_data_still_terminates(self):
        pattern = regex_up_to_next_prompt(rb"You watch the world go by\.")

        assert pattern.search(b"sip t\r\nYou watch the world go by.\r\n" + GAME_PROMPT)

    def test_a_response_followed_by_the_next_echo_still_terminates(self):
        pattern = regex_up_to_next_prompt(rb"You watch the world go by\.")
        wire = b"You watch the world go by.\r\n(Persona saved on +200 = 200).\r\n" + GAME_PROMPT + b"fscore\r\n"

        assert pattern.search(wire)

    def test_a_prompt_shaped_narrative_line_is_not_a_terminator(self):
        pattern = regex_up_to_next_prompt(rb"You watch the world go by\.")

        assert pattern.search(b"You watch the world go by.\r\n*\r\nmore narrative\r\n") is None

    def test_a_narrative_line_split_before_its_line_feed_is_not_a_terminator(self):
        pattern = regex_up_to_next_prompt(rb"You watch the world go by\.")

        assert pattern.search(b"You watch the world go by.\r\n*\r") is None


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
def test_invalid_command_prompts_match_system_lines_but_not_spoken_copies(message):
    genuine_response = b"\x1b[0;37;40m" + message + b"\r\n" + GAME_PROMPT
    spoken_copy = b'Raymond the protector says "' + message + b'".\r\n' + GAME_PROMPT

    assert any(pattern.search(genuine_response) for pattern in INVALID_COMMAND_PROMPTS)
    assert not any(pattern.search(spoken_copy) for pattern in INVALID_COMMAND_PROMPTS)


class TestTearoomScreenToleratesRunAheadOutput:
    SCREEN = (
        b"\x1b[32mElizabethan tearoom\x1b[37m.\r\n"
        b"\x1b[0;32;40mThis cosy, Tudor period room is where all MUD adventures start. \x1b[1;37;40m\r\n"
        b"Players:\r\n"
        b"\x1b[0;37;40m\x1b[31mAlexis\x1b[37m\r\n"
    )

    def test_the_screen_matches_with_its_prompt_at_the_end_of_the_data(self):
        assert Prompt.TEAROOM.value.search(self.SCREEN + GAME_PROMPT)

    def test_the_screen_matches_when_the_sip_echo_already_arrived(self):
        assert Prompt.TEAROOM.value.search(self.SCREEN + GAME_PROMPT + b"sip t\r\n")


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
