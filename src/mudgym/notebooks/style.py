"""Visual tokens and value formatters shared by the notebook renderers."""

import html as html_lib
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


MUDGYM_PALETTE = {
    "primary": "#18352f",
    "secondary": "#244c44",
    "accent": "#b58a2a",
    "highlight": "#f6ff31",
    "background": "#ffffff",
    "ink": "#1b2825",
    "code_background": "#0b0d0c",
}


def css_variable(colour_name: str) -> str:
    return "--notebook-" + colour_name.replace("_", "-")


def palette_colour(colour_name: str) -> str:
    return f"var({css_variable(colour_name)},{MUDGYM_PALETTE[colour_name]})"


# Literal fallbacks so a notebook still renders without MudGym's stylesheet.
SANS = "var(--notebook-sans,'Notebook Inter',Inter,'Segoe UI',sans-serif)"
MONO = "var(--notebook-mono,'Notebook JetBrains Mono','JetBrains Mono','Fira Mono',monospace)"
RULE = "var(--notebook-rule,#e3e7e5)"
TINT = "var(--notebook-tint,#eef2f0)"
MUTED = "var(--notebook-muted,#6c7a75)"
INK = palette_colour("ink")
PRIMARY = palette_colour("primary")
SECONDARY = palette_colour("secondary")
ACCENT = palette_colour("accent")
PAGE = palette_colour("background")

PANEL_STYLE = f"border:1px solid {RULE};border-radius:8px;background:{PAGE};"

# The vertical rhythm between stacked widgets.
BLOCK_MARGIN = "margin:0.3rem 0 0.5rem;"

CODE_SPAN_STYLE = (
    f"font-family:{MONO};background:{TINT};color:{SECONDARY};"
    "border-radius:0.3rem;padding:0.12em 0.38em;font-size:0.92em;font-weight:560;"
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
