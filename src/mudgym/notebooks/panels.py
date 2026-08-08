"""Palette and control-panel widgets for the example notebooks."""

import html as html_lib
import subprocess
from collections.abc import Sequence
from typing import Any

from mudgym.notebooks.style import HTML

NOTEBOOK_PALETTES = {
    "mudgym": {
        "primary": "#18352f",
        "secondary": "#244c44",
        "accent": "#b58a2a",
        "highlight": "#f6ff31",
        "background": "#ffffff",
        "ink": "#1b2825",
        "code_background": "#0b0d0c",
    },
}

CSS_VARIABLE_BY_COLOUR = {
    "primary": "--notebook-primary",
    "secondary": "--notebook-secondary",
    "accent": "--notebook-accent",
    "highlight": "--notebook-highlight",
    "background": "--notebook-background",
    "ink": "--notebook-ink",
    "code_background": "--notebook-code-background",
}


def notebook_palette(
    palette: str = "mudgym",
    *,
    primary: str | None = None,
    secondary: str | None = None,
    accent: str | None = None,
    highlight: str | None = None,
    background: str | None = None,
    ink: str | None = None,
    code_background: str | None = None,
) -> HTML:
    """Select a named palette and optionally replace any of its main colours."""
    if palette not in NOTEBOOK_PALETTES:
        available_palettes = ", ".join(sorted(NOTEBOOK_PALETTES))
        raise ValueError(f"Unknown notebook palette {palette!r}. Choose from: {available_palettes}")

    colour_overrides = {
        "primary": primary,
        "secondary": secondary,
        "accent": accent,
        "highlight": highlight,
        "background": background,
        "ink": ink,
        "code_background": code_background,
    }
    selected_colours = {
        colour_name: default_colour if colour_overrides[colour_name] is None else colour_overrides[colour_name]
        for colour_name, default_colour in NOTEBOOK_PALETTES[palette].items()
    }
    css_declarations = "\n".join(
        f"        {CSS_VARIABLE_BY_COLOUR[colour_name]}: {colour};" for colour_name, colour in selected_colours.items()
    )
    escaped_palette = html_lib.escape(palette, quote=True)
    return HTML(
        f"""
<style data-notebook-palette="{escaped_palette}">
  #App {{
{css_declarations}
  }}
</style>
"""
    )


def docker_is_running() -> bool:
    """True if the Docker daemon is reachable."""
    try:
        completed = subprocess.run(["docker", "info"], capture_output=True)
    except FileNotFoundError:
        return False
    return completed.returncode == 0


def env_setup_panel(
    *,
    players: int = 4,
    players_range: tuple[int, int] = (2, 8),
    max_steps: int = 200,
    max_steps_range: tuple[int, int] = (5, 1000),
    tearoom_commands: str = "",
    gather_on: str | None = None,
    mobiles: Sequence[str] = ("banshee", "wraith"),
    submit_label: str | None = "Go",
) -> Any:
    """One shared control panel for the multi-agent notebooks: party size, step budget, and setup.

    Returns a marimo element whose ``.value`` holds ``players``, ``max_steps``,
    ``tearoom_commands`` (the comma-chained command line issued at episode
    start, ready to pass to ``make_env``), and a ``gather_on`` dropdown when
    that names a default mobile. With a ``submit_label`` the panel is a form, so
    ``.value`` stays ``None`` until the button is pressed and expensive runs
    only start deliberately; ``None`` gives a live panel instead.
    """
    # marimo ships with the examples, not the library, so this import stays local.
    import marimo as mo  # noqa: PLC0415

    elements = {
        "players": mo.ui.slider(
            start=players_range[0],
            stop=players_range[1],
            step=1,
            value=players,
            debounce=True,
            show_value=True,
            label="Players",
        ),
        "max_steps": mo.ui.slider(
            start=max_steps_range[0],
            stop=max_steps_range[1],
            step=5,
            value=max_steps,
            debounce=True,
            show_value=True,
            label="Max steps",
        ),
        "tearoom_commands": mo.ui.text(
            value=tearoom_commands,
            label="Setup commands",
        ),
    }
    setup_row = "{gather_on} &nbsp; {tearoom_commands}" if gather_on is not None else "{tearoom_commands}"
    if gather_on is not None:
        elements["gather_on"] = mo.ui.dropdown(
            options=list(mobiles),
            value=gather_on,
            label="Gather on",
        )

    panel = mo.md("{players} &nbsp; {max_steps} &nbsp; " + setup_row).batch(**elements)
    if submit_label is None:
        return panel
    return panel.form(submit_button_label=submit_label, bordered=False)
