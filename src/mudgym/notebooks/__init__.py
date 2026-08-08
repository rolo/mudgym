"""Helpers for rendering MUD output nicely inside notebooks.

Everything the notebooks and docs are meant to use is re-exported here. The
browser copies of the examples import these names from a wheel built elsewhere,
so a rename that forgets one fails in a notebook cell with nowhere to show it.
"""

from mudgym.notebooks.frames import (
    game_accordion,
    show_ansi,
    show_episode,
    show_frames,
    show_game_tabs,
    show_text,
    transition_caption,
)
from mudgym.notebooks.panels import docker_is_running, env_setup_panel, notebook_palette
from mudgym.notebooks.room_map import show_room_map
from mudgym.notebooks.style import HTML
from mudgym.notebooks.tables import show_cards, show_stats, show_table, show_turn_bars

__all__ = [
    "HTML",
    "docker_is_running",
    "env_setup_panel",
    "game_accordion",
    "notebook_palette",
    "show_ansi",
    "show_cards",
    "show_episode",
    "show_frames",
    "show_game_tabs",
    "show_room_map",
    "show_stats",
    "show_table",
    "show_text",
    "show_turn_bars",
    "transition_caption",
]
