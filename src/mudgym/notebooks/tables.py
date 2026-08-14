"""Tables, cards, stat tiles, and sparklines for notebook output."""

import html as html_lib
from collections.abc import Mapping, Sequence
from typing import Any

from mudgym.notebooks.style import (
    HTML,
    INK,
    LABEL_STYLE,
    MONO,
    MUTED,
    PAGE,
    PANEL_STYLE,
    RULE,
    SANS,
    SECONDARY,
    TINT,
    cell_text,
)


def show_table(rows: Sequence[Mapping[str, Any]], *, mono_columns: Sequence[str] = ()) -> HTML:
    """Render rows of data as a quiet, readable table.

    Columns come from the keys of the first row. Numbers are right-aligned in
    tabular figures and booleans read as yes/no. Columns named in
    ``mono_columns`` are set in the code face, which suits agent names,
    commands, and room ids.
    """
    rows = list(rows)
    if not rows:
        return HTML(f'<p style="font-family:{SANS};color:{MUTED};margin:0.4rem 0;">No rows.</p>')

    columns = list(rows[0])
    numeric_columns = {
        column
        for column in columns
        if all(isinstance(row.get(column), (int, float)) and not isinstance(row.get(column), bool) for row in rows)
    }
    mono_columns = set(mono_columns)

    header_cells = "".join(
        f'<th style="{LABEL_STYLE}padding:0.5rem 0.8rem;border-bottom:1px solid {RULE};'
        f'text-align:{"right" if column in numeric_columns else "left"};white-space:nowrap;">'
        f"{html_lib.escape(str(column))}</th>"
        for column in columns
    )

    body_rows = []
    for index, row in enumerate(rows):
        stripe = f"background:{TINT};" if index % 2 else ""
        cells = []
        for position, column in enumerate(columns):
            fixed = column in numeric_columns or column in mono_columns
            alignment = "right" if column in numeric_columns else "left"
            face = f"font-family:{MONO};font-size:0.82rem;" if fixed else ""
            # Names, commands, and numbers keep one line. Prose columns wrap.
            wrap = "white-space:nowrap;" if fixed else "overflow-wrap:anywhere;"
            weight = "font-weight:600;" if position == 0 else ""
            cells.append(
                f'<td style="padding:0.42rem 0.8rem;text-align:{alignment};{face}{weight}{wrap}'
                f'font-variant-numeric:tabular-nums;">'
                f"{html_lib.escape(cell_text(row.get(column)))}</td>"
            )
        body_rows.append(f'<tr style="{stripe}">{"".join(cells)}</tr>')

    return HTML(
        f'<div class="notebook-table" style="{PANEL_STYLE}overflow-x:auto;margin:0.3rem 0 0.5rem;">'
        f'<table style="border-collapse:collapse;width:100%;font-family:{SANS};font-size:0.92rem;color:{INK};">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table></div>"
    )


def show_turn_bars(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    label: str,
    max_value: int | None = None,
) -> HTML:
    """Render one integer series as a thin bar sparkline over turns.

    Every turn is a full-height hover target with a native tooltip, and the
    caption direct-labels the peak. ``max_value`` pins the scale to a known
    ceiling, such as the party size, so half-full reads as half-full.
    """
    values = [int(row[value_key]) for row in rows]
    if not values:
        return HTML(f'<p style="font-family:{SANS};color:{MUTED};margin:0.4rem 0;">No turns.</p>')

    top = max([*values, max_value or 1, 1])
    peak = max(values)
    peak_turn = values.index(peak) + 1
    bar_fill = "var(--notebook-secondary,#244c44)"

    bar_width = min(40, max(2, 640 // len(values) - 2))
    stride = bar_width + 2
    width = stride * len(values)
    plot_height = 56

    bars = []
    for index, value in enumerate(values):
        x = index * stride
        bar_height = round(plot_height * value / top)
        visible_bar = (
            f'<rect x="{x}" y="{plot_height - bar_height}" width="{bar_width}" height="{bar_height}" rx="1.5" fill="'
            f'{bar_fill}"/>'
            if bar_height
            else ""
        )
        bars.append(
            f'<g><title>Turn {index + 1} - {value}</title><rect x="'
            f'{x}" y="0" width="{stride}" height="{plot_height}" fill="transparent"/>'
            f"{visible_bar}</g>"
        )

    caption = f"peak {peak} on turn {peak_turn} · {len(values)} turn{'s' if len(values) != 1 else ''}"
    return HTML(
        f'<div style="font-family:{SANS};margin:0.3rem 0 0.5rem;">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;">'
        f'<span style="{LABEL_STYLE}">{html_lib.escape(label)}</span>'
        f'<span style="font-family:{MONO};font-size:0.78rem;color:{MUTED};">{caption}</span>'
        '</div><div style="overflow-x:auto;">'
        f'<svg width="{width}" height="{plot_height + 2}" viewBox="0 0 {width} {plot_height + 2}" role="img" aria-label="'
        f'{html_lib.escape(label)}">'
        f"{''.join(bars)}"
        f'<rect x="0" y="{plot_height}" width="{width}" height="1" fill="{RULE}"/></svg></div></div>'
    )


def show_cards(cards: Mapping[str, Mapping[str, Any]], *, minimum_width: str = "17rem") -> HTML:
    """Render titled label/value panels in a responsive grid.

    Reads better than a table for a handful of records with long values, such
    as one observation per agent.
    """
    panels = []
    for title, fields in cards.items():
        field_rows = "".join(
            f'<dt style="{LABEL_STYLE}margin-bottom:0.1rem;">{html_lib.escape(str(label))}</dt>'
            f'<dd style="margin:0 0 0.55rem;overflow-wrap:anywhere;">{html_lib.escape(cell_text(value))}</dd>'
            for label, value in fields.items()
        )
        panels.append(
            f'<section style="{PANEL_STYLE}padding:0.8rem 0.95rem 0.4rem;">'
            f'<h4 style="margin:0 0 0.6rem;padding-bottom:0.45rem;border-bottom:1px solid {RULE};'
            f'font-family:{MONO};font-size:0.8rem;font-weight:700;letter-spacing:0.02em;color:{SECONDARY};">'
            f"{html_lib.escape(str(title))}</h4>"
            f'<dl style="margin:0;font-family:{SANS};font-size:0.88rem;line-height:1.45;color:{INK};">'
            f"{field_rows}</dl></section>"
        )

    # auto-fit, not auto-fill, so a single card fills the width instead of
    # sitting in a narrow column beside empty ones.
    return HTML(
        f'<div class="notebook-cards" style="display:grid;gap:0.6rem;margin:0.3rem 0 0.5rem;'
        f'grid-template-columns:repeat(auto-fit,minmax({minimum_width},1fr));">{"".join(panels)}</div>'
    )


def show_stats(stats: Mapping[str, Any]) -> HTML:
    """Render a row of headline numbers, one tile per entry."""
    tiles = "".join(
        f'<div style="border:1px solid {RULE};background:{PAGE};padding:0.6rem 0.85rem;">'
        f'<div style="{LABEL_STYLE}">{html_lib.escape(str(label))}</div>'
        f'<div style="font-family:{SANS};font-size:1.5rem;font-weight:740;letter-spacing:-0.02em;'
        f'color:{INK};font-variant-numeric:tabular-nums;">{html_lib.escape(cell_text(value))}</div>'
        f"</div>"
        for label, value in stats.items()
    )
    return HTML(
        f'<div class="notebook-stats" style="display:grid;gap:0.6rem;margin:0.3rem 0 0.5rem;'
        f'grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));">{tiles}</div>'
    )
