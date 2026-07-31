import pytest

from mudgym.notebooks import (
    HTML,
    game_accordion,
    notebook_palette,
    show_ansi,
    show_game,
    show_game_tabs,
    show_text,
)


def test_notebook_palette_renders_named_colours_and_overrides():
    html = notebook_palette("mudgym", accent="#abcdef")

    assert 'data-notebook-palette="mudgym"' in html.data
    assert "--notebook-primary: #18352f;" in html.data
    assert "--notebook-accent: #abcdef;" in html.data


def test_notebook_palette_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown notebook palette"):
        notebook_palette("unknown")


def test_show_ansi_keeps_game_palette_blue():
    html = show_ansi(b"\x1b[0;34;40mBeaten track near cliff.\x1b[0m")

    assert "Beaten track near cliff." in html.data
    assert "color: #0000aa" in html.data
    assert "background-color: #000316" in html.data


def test_show_text_renders_plain_text_without_ansi_spans():
    html = show_text("<Beaten track near cliff.>")

    assert "&lt;Beaten track near cliff.&gt;" in html.data
    assert "<Beaten track near cliff.>" not in html.data
    assert "color: #0000aa" not in html.data


def test_show_game_renders_byte_chunks():
    html = show_game({"text_chunks": [b"\x1b[31mred\x1b[0m", b"plain"]}, key="text_chunks")

    assert "red" in html.data
    assert "plain" in html.data


def test_notebook_palette_defaults_to_mudgym():
    html = notebook_palette()

    assert 'data-notebook-palette="mudgym"' in html.data


def test_show_game_tabs_renders_both_streams_with_display_selected():
    html = show_game_tabs({"render_bytes": b"redrawn screen", "raw_bytes": b"step response"})

    assert "redrawn screen" in html.data
    assert "step response" in html.data
    assert ">Display</label>" in html.data
    assert ">Raw</label>" in html.data
    display_input, raw_input = html.data.split("<label")[0].split("<input")[1:]
    assert "checked" in display_input
    assert "checked" not in raw_input


def test_show_game_tabs_uses_a_fresh_radio_group_per_widget():
    info = {"render_bytes": b"screen", "raw_bytes": b"response"}
    first, second = show_game_tabs(info), show_game_tabs(info)

    first_group = first.data.split('name="')[1].split('"')[0]
    second_group = second.data.split('name="')[1].split('"')[0]
    assert first_group != second_group


def test_game_accordion_starts_expanded_and_can_start_collapsed():
    items = {"Episode 1": HTML("<b>frame</b>"), "Episode <2>": "<i>raw html</i>"}

    expanded = game_accordion(items)
    assert expanded.data.count("<details") == 2
    assert expanded.data.count('<details class="game-accordion" open') == 2
    assert "<b>frame</b>" in expanded.data
    assert "<i>raw html</i>" in expanded.data
    assert "Episode &lt;2&gt;" in expanded.data

    collapsed = game_accordion(items, expanded=False)
    assert " open" not in collapsed.data


def test_game_accordion_rejects_unrenderable_values():
    with pytest.raises(TypeError, match="game accordion"):
        game_accordion({"bad": object()})
