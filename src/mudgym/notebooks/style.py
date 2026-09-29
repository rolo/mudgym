"""Visual tokens and value formatters shared by the notebook renderers."""

import html as html_lib
import re
from importlib import resources
from typing import Any


class HTML:
    """Minimal stand-in for ``IPython.display.HTML``.

    Jupyter and marimo render anything exposing ``_repr_html_``, so notebook
    output needs no dependency on IPython's interactive-shell stack.
    """

    def __init__(self, data: str) -> None:
        self.data = data

    def _repr_html_(self) -> str:
        return self.data


# Every value lives in theme.css, which the docs load directly. Notebooks never
# load it, so each token carries its value from the file as the var() fallback.
THEME_CSS = resources.files("mudgym.notebooks").joinpath("theme.css").read_text(encoding="utf-8")
THEME = dict(re.findall(r"--notebook-([a-z-]+):\s*([^;]+);", THEME_CSS))

PALETTE_COLOURS = ("primary", "secondary", "accent", "highlight", "background", "ink", "code_background")
MUDGYM_PALETTE = {colour_name: THEME[colour_name.replace("_", "-")] for colour_name in PALETTE_COLOURS}


def css_variable(colour_name: str) -> str:
    return "--notebook-" + colour_name.replace("_", "-")


def token(name: str) -> str:
    """Reference a theme token, falling back to its theme.css value with any nested references resolved too."""
    fallback = re.sub(r"var\(--notebook-([a-z-]+)\)", lambda match: token(match.group(1)), THEME[name])
    return f"var(--notebook-{name},{fallback})"


SANS = token("sans")
MONO = token("mono")
RULE = token("rule")
TINT = token("tint")
MUTED = token("muted")
INK = token("ink")
PRIMARY = token("primary")
SECONDARY = token("secondary")
ACCENT = token("accent")
PAGE = token("background")
RADIUS = token("radius")
SHADOW = token("shadow")
# SVG drawings take the radius as a bare number.
CORNER_RADIUS = float(THEME["radius"].removesuffix("px"))

PANEL_STYLE = f"border:1px solid {RULE};border-radius:{RADIUS};background:{PAGE};"

# The vertical rhythm between stacked widgets.
BLOCK_MARGIN = "margin:0.3rem 0 0.5rem;"

CODE_SPAN_STYLE = (
    f"font-family:{MONO};background:{TINT};color:{SECONDARY};"
    f"border-radius:{RADIUS};padding:0.12em 0.38em;font-size:0.92em;font-weight:560;"
)

LABEL_STYLE = (
    f"font-family:{MONO};font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};"
)


def empty_html(message: str) -> HTML:
    """Render the muted line a widget shows when it has nothing to draw."""
    return HTML(f'<p style="font-family:{SANS};color:{MUTED};margin:0.4rem 0;">{html_lib.escape(message)}</p>')


def caption_html(text: str) -> str:
    """Escape a caption, rendering `backticked` spans as inline code."""
    escaped = html_lib.escape(text)
    parts = escaped.split("`")
    if len(parts) % 2 == 0:
        # An unpaired backtick: leave the text exactly as written.
        return escaped
    return "".join(
        f'<code style="{CODE_SPAN_STYLE}">{part}</code>' if index % 2 else part for index, part in enumerate(parts)
    )


def cell_text(value: Any) -> str:
    """Format one table or card value, rendering a missing one as a blank cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
