"""Helpers for rendering MUD output nicely inside notebooks."""

import html as html_lib
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import ansi2html

from mudgym.featurizers.strings import decode_text_bytes


class HTML:
    """Minimal stand-in for ``IPython.display.HTML``.

    Notebook frontends (Jupyter, marimo) render any object exposing
    ``_repr_html_``, so returning one of these displays styled game output
    without a hard dependency on IPython and its interactive-shell stack.
    """

    def __init__(self, data: str) -> None:
        self.data = data

    def _repr_html_(self) -> str:
        return self.data


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

_CSS_VARIABLE_BY_COLOUR = {
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
        f"        {_CSS_VARIABLE_BY_COLOUR[colour_name]}: {colour};" for colour_name, colour in selected_colours.items()
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


# Game-output frame styling, applied inline on each frame rather than through a
# stylesheet. The inline copy is what makes show_game render correctly inside
# marimo components such as mo.accordion/mo.ui.tabs, whose content sits behind a
# boundary a global `.game-frame` class does not reach. The var() references
# resolve against the palette on #App and fall back to the mudgym defaults.
_FRAME_STYLE = (
    "background:var(--notebook-code-background,#0b0d0c);"
    "color:#eee;"
    "font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;"
    "font-size:14px;"
    "line-height:1.5;"
    "padding:1.1em 1.2em;"
    "margin:0;"
    "white-space:pre-wrap;"
    "border:1px solid var(--notebook-primary,#18352f);"
    "border-left:4px solid var(--notebook-accent,#b58a2a);"
    "border-radius:12px;"
    "box-shadow:0 12px 28px rgba(24,53,47,0.12);"
    "max-height:60vh;"
    "overflow:auto;"
)


def display_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return b"\n".join(bytes(chunk) for chunk in value)
    return bytes(value)


def display_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return decode_text_bytes(display_bytes(value))


def show_ansi(value: Any) -> HTML:
    """Render ANSI text or bytes as styled HTML suitable for marimo notebooks."""

    text = display_text(value)
    converter = ansi2html.Ansi2HTMLConverter(inline=True)
    html = converter.convert(text, full=False)
    # Style inline (not only via the .game-frame class) so the frame survives
    # inside marimo components like mo.accordion. See _FRAME_STYLE above.
    return HTML(f'<div class="game-frame" style="{_FRAME_STYLE}">{html}</div>')


def show_text(value: Any) -> HTML:
    """Render plain text in the game-output frame without interpreting ANSI state."""

    text = html_lib.escape(display_text(value))
    return HTML(f'<div class="game-frame" style="{_FRAME_STYLE}">{text}</div>')


def show_game(input_dict: Any, key: str = "raw_bytes") -> HTML:
    """Render MUD response bytes as styled HTML suitable for notebooks."""

    return show_ansi(input_dict[key])


# Tab widgets need unique radio-group names so browsers keep each widget's
# selection independent. A process-wide counter is enough: every render within
# one kernel (or one export) gets a fresh group.
_TAB_GROUP_IDS = itertools.count(1)

# Structural CSS for the tab widget, emitted inside every widget for the same
# reason _FRAME_STYLE is inlined: content placed inside marimo components sits
# behind a boundary a global stylesheet does not reach. Checked-state selectors
# cannot be inline attributes, so the widget carries its own <style> block;
# duplicates are harmless.
_TAB_STYLE = """<style>
.game-tabs > input { position: absolute; opacity: 0; pointer-events: none; }
.game-tabs .game-tab-panel { display: none; }
.game-tabs > input:nth-of-type(1):checked ~ .game-tab-panel-display { display: block; }
.game-tabs > input:nth-of-type(2):checked ~ .game-tab-panel-raw { display: block; }
.game-tabs .game-tab-labels { display: flex; gap: 0.4rem; margin: 0 0 0.5rem; }
.game-tabs .game-tab-labels label {
    cursor: pointer;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--notebook-primary, #18352f);
    color: var(--notebook-primary, #18352f);
    opacity: 0.65;
}
.game-tabs > input:nth-of-type(1):checked ~ .game-tab-labels label:nth-of-type(1),
.game-tabs > input:nth-of-type(2):checked ~ .game-tab-labels label:nth-of-type(2) {
    background: var(--notebook-primary, #18352f);
    color: #ffffff;
    opacity: 1;
}
</style>"""


def show_game_tabs(input_dict: Any) -> HTML:
    """Render one game response as tabs for its display and raw protocol output.

    ``Display`` shows ``render_bytes`` (the game display after the action), while
    ``Raw`` shows ``raw_bytes`` (the protocol output produced by the action).
    The tabs are pure HTML/CSS radio inputs, so switching between them works in
    static HTML exports with no kernel attached, and inside accordions.
    """

    group = f"game-tabs-{next(_TAB_GROUP_IDS)}"
    display = show_ansi(input_dict["render_bytes"]).data
    raw = show_ansi(input_dict["raw_bytes"]).data
    return HTML(
        f'<div class="game-tabs">'
        f"{_TAB_STYLE}"
        f'<input type="radio" name="{group}" id="{group}-display" checked>'
        f'<input type="radio" name="{group}" id="{group}-raw">'
        f'<div class="game-tab-labels">'
        f'<label for="{group}-display">Display</label>'
        f'<label for="{group}-raw">Raw</label>'
        f"</div>"
        f'<div class="game-tab-panel game-tab-panel-display">{display}</div>'
        f'<div class="game-tab-panel game-tab-panel-raw">{raw}</div>'
        f"</div>"
    )


_ACCORDION_SUMMARY_STYLE = (
    "cursor:pointer;"
    "font-size:0.85rem;"
    "font-weight:600;"
    "padding:0.45rem 0.8rem;"
    "border-radius:8px;"
    "background:var(--notebook-primary,#18352f);"
    "color:#ffffff;"
)


def _accordion_item_html(value: Any) -> str:
    if hasattr(value, "_repr_html_"):
        return value._repr_html_()
    if isinstance(value, str):
        return value
    # marimo's Html and UI elements stringify to markup marimo can hydrate.
    module_root = type(value).__module__.split(".")[0]
    if module_root == "marimo":
        return str(value)
    raise TypeError(f"Cannot render {type(value).__name__} in a game accordion; pass HTML or a marimo object.")


def game_accordion(items: Mapping[str, Any], *, expanded: bool = True) -> HTML:
    """Stack titled, collapsible sections that are expanded by default.

    A plain-HTML replacement for ``mo.accordion`` built from ``<details>``
    blocks, so sections can start open and still collapse - including in static
    HTML exports where ``mo.accordion`` always starts closed. Values may be
    anything with ``_repr_html_`` (``show_game`` frames, ``show_game_tabs``),
    marimo objects, or raw HTML strings.
    """

    open_attribute = " open" if expanded else ""
    sections = "".join(
        f'<details class="game-accordion"{open_attribute} style="margin:0 0 0.6rem;">'
        f'<summary style="{_ACCORDION_SUMMARY_STYLE}">{html_lib.escape(str(title))}</summary>'
        f'<div style="padding:0.6rem 0 0;">{_accordion_item_html(value)}</div>'
        f"</details>"
        for title, value in items.items()
    )
    return HTML(sections)
