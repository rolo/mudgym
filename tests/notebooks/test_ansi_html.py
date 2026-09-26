from mudgym.featurizers.ansi import INPUT_STYLE
from mudgym.notebooks import show_ansi
from mudgym.notebooks.ansi import ANSI_BRIGHT, ANSI_NORMAL, TERMINAL_SURFACE, ansi_html

# The start of a move response, as the engine sends it after a prompt.
MOVE_RESPONSE = "\x1b[32mBeaten track\x1b[37m.\r\n\x1b[0;32;40mYou're on a rough east-west track."
INPUT = INPUT_STYLE.decode("latin-1")


def test_a_room_name_after_the_prompt_reads_bold_and_bright():
    rendered = ansi_html(INPUT + MOVE_RESPONSE)

    assert (
        f'<span style="color: {ANSI_BRIGHT[2]}; background-color: {TERMINAL_SURFACE}; font-weight: 700">Beaten track</span>'
        in rendered
    )
    assert f'<span style="color: {ANSI_NORMAL[2]}; background-color: {TERMINAL_SURFACE}">You\'re on' in rendered


def test_game_frames_paint_black_as_the_frame_surface():
    frame = show_ansi(INPUT + MOVE_RESPONSE).data

    assert f"background:{TERMINAL_SURFACE};" in frame
    assert f"background-color: {ANSI_NORMAL[0]}" not in frame


def test_text_is_escaped_and_other_control_sequences_are_dropped():
    assert ansi_html("<b>&</b>\x1b[2Jdone") == "&lt;b&gt;&amp;&lt;/b&gt;done"


def test_inverse_and_extended_colours():
    assert ansi_html("\x1b[7mx") == f'<span style="color: {TERMINAL_SURFACE}; background-color: #dde0ed">x</span>'
    assert ansi_html("\x1b[38;5;196mx") == '<span style="color: rgb(255, 0, 0)">x</span>'
    assert ansi_html("\x1b[48;2;1;2;3mx") == '<span style="background-color: rgb(1, 2, 3)">x</span>'
    assert ansi_html("\x1b[1;32mx\x1b[22my") == (
        f'<span style="color: {ANSI_BRIGHT[2]}; font-weight: 700">x</span><span style="color: {ANSI_NORMAL[2]}">y</span>'
    )
