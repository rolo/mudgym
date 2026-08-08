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


# Literal fallbacks so a notebook still renders without MudGym's stylesheet.
SANS = "var(--notebook-sans,'Notebook Inter',Inter,'Segoe UI',sans-serif)"
MONO = "var(--notebook-mono,'Notebook JetBrains Mono','JetBrains Mono','Fira Mono',monospace)"
RULE = "var(--notebook-rule,#e3e7e5)"
TINT = "var(--notebook-tint,#eef2f0)"
MUTED = "var(--notebook-muted,#6c7a75)"
INK = "var(--notebook-ink,#1b2825)"
PRIMARY = "var(--notebook-primary,#18352f)"
SECONDARY = "var(--notebook-secondary,#244c44)"
ACCENT = "var(--notebook-accent,#b58a2a)"
PAGE = "var(--notebook-background,#ffffff)"

PANEL_STYLE = f"border:1px solid {RULE};border-radius:12px;background:{PAGE};"

CODE_SPAN_STYLE = (
    f"font-family:{MONO};background:{TINT};color:{SECONDARY};"
    "border-radius:0.45rem;padding:0.12em 0.38em;font-size:0.92em;font-weight:560;"
)

LABEL_STYLE = (
    f"font-family:{MONO};font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};"
)


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
