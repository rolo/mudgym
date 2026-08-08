import importlib

import pytest

import mudgym.notebooks
from mudgym.notebooks import (
    HTML,
    game_accordion,
    notebook_palette,
    show_ansi,
    show_game_tabs,
    show_table,
    show_text,
    show_turn_bars,
)


def test_every_exported_name_is_importable_from_the_package_root():
    """The browser copies of the examples import these by name from the wheel."""
    for name in mudgym.notebooks.__all__:
        assert getattr(mudgym.notebooks, name, None) is not None, name


def test_submodules_are_importable_on_their_own():
    for module in ("frames", "panels", "room_map", "style", "tables"):
        importlib.import_module(f"mudgym.notebooks.{module}")


def test_show_turn_bars_renders_a_hover_target_per_turn_and_labels_the_peak():
    html = show_turn_bars(
        [{"Attacking": 0}, {"Attacking": 3}, {"Attacking": 1}],
        value_key="Attacking",
        label="Attacks per turn",
        max_value=4,
    )

    assert html.data.count("<title>") == 3
    assert html.data.count('rx="1.5"') == 2  # a zero turn keeps its hover target but draws no bar
    assert "Attacks per turn" in html.data
    assert "peak 3 on turn 2" in html.data


def test_show_turn_bars_without_rows_reads_no_turns():
    html = show_turn_bars([], value_key="Attacking", label="Attacks per turn")

    assert "No turns." in html.data


def test_notebook_palette_renders_named_colours_and_overrides():
    html = notebook_palette("mudgym", accent="#abcdef")

    assert 'data-notebook-palette="mudgym"' in html.data
    assert "--notebook-primary: #18352f;" in html.data
    assert "--notebook-accent: #abcdef;" in html.data


def test_notebook_palette_defaults_to_mudgym():
    html = notebook_palette()

    assert 'data-notebook-palette="mudgym"' in html.data


def test_notebook_palette_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown notebook palette"):
        notebook_palette("unknown")


def test_show_ansi_keeps_game_palette_blue():
    html = show_ansi(b"\x1b[0;34;40mBeaten track near cliff.\x1b[0m")

    assert "Beaten track near cliff." in html.data
    assert "color: #0000aa" in html.data
    assert "background-color: #000316" in html.data


def test_show_ansi_joins_byte_chunks():
    html = show_ansi([b"\x1b[31mred\x1b[0m", b"plain"])

    assert "red" in html.data
    assert "plain" in html.data


def test_show_text_renders_plain_text_without_ansi_spans():
    html = show_text("<Beaten track near cliff.>")

    assert "&lt;Beaten track near cliff.&gt;" in html.data
    assert "<Beaten track near cliff.>" not in html.data
    assert "color: #0000aa" not in html.data


def test_show_game_tabs_renders_both_streams_with_display_selected():
    html = show_game_tabs({"render_bytes": b"redrawn screen", "raw_bytes": b"step response"})

    assert "redrawn screen" in html.data
    assert "step response" in html.data
    assert ">display</label>" in html.data
    assert ">raw</label>" in html.data
    display_input, raw_input = html.data.split("<label")[0].split("<input")[1:]
    assert "checked" in display_input
    assert "checked" not in raw_input


def test_show_game_tabs_uses_a_fresh_radio_group_per_widget():
    info = {"render_bytes": b"screen", "raw_bytes": b"response"}
    first, second = show_game_tabs(info), show_game_tabs(info)

    first_group = first.data.split('name="')[1].split('"')[0]
    second_group = second.data.split('name="')[1].split('"')[0]
    assert first_group != second_group


def test_show_game_tabs_adds_an_observation_tab_when_given_one():
    html = show_game_tabs(
        {"render_bytes": b"screen", "raw_bytes": b"response"},
        observation={"room_name": b"Tearoom", "available_exits": (1, 0, 1)},
    )

    assert ">observation</label>" in html.data
    assert "Tearoom" in html.data
    assert "1, 0, 1" in html.data


def test_show_table_right_aligns_numbers_and_reads_booleans_as_words():
    html = show_table([{"Agent": "player_0", "Score": 12, "Alive": True}], mono_columns=("Agent",))

    assert "player_0" in html.data
    assert "yes" in html.data
    assert "text-align:right" in html.data


def test_show_table_without_rows_reads_no_rows():
    assert "No rows." in show_table([]).data


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
