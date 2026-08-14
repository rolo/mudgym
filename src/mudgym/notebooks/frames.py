"""Game-output frames, tabs, accordions, and episode replays."""

import html as html_lib
import itertools
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ansi2html

from mudgym.db.directions import DIRECTIONS
from mudgym.featurizers.strings import decode_text_bytes
from mudgym.notebooks.style import (
    HTML,
    LABEL_STYLE,
    MONO,
    MUTED,
    PAGE,
    PRIMARY,
    RULE,
    SANS,
    SECONDARY,
    TINT,
    caption_html,
    cell_text,
)
from mudgym.notebooks.tables import show_table

# Every widget here carries its own inline styles and <style> block. That is
# what makes it render inside marimo components such as mo.accordion, whose
# content sits behind a boundary a page stylesheet does not reach, and in static
# HTML exports with no kernel attached. Duplicated blocks are harmless.
FRAME_STYLE = (
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
)

# Long output collapses rather than scrolling: a frame with its own scrollbar
# inside the notebook's is awkward to drive and hides how much output there is.
COLLAPSED_LINE_COUNT = 24

# Each widget needs its own id or two of them on a page drive each other.
FRAME_IDS = itertools.count(1)
TAB_GROUP_IDS = itertools.count(1)
FRAME_SET_IDS = itertools.count(1)

# The collapsed height lives here, not inline, so the `:checked` rule can win.
COLLAPSE_STYLE = f"""<style>
.game-frame-collapsible {{ position: relative; }}
.game-frame-collapsible > input {{ position: absolute; opacity: 0; pointer-events: none; }}
.game-frame-collapsible > .game-frame {{
    max-height: {COLLAPSED_LINE_COUNT * 1.5:g}em;
    overflow: hidden;
}}
.game-frame-collapsible > input:checked ~ .game-frame {{ max-height: none; }}
.game-frame-collapsible > .game-frame-more {{
    position: relative;
    display: block;
    margin: 0;
    padding: 0.45rem 0 0;
    cursor: pointer;
    text-align: center;
    font-family: {SANS};
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: {PRIMARY};
}}
.game-frame-collapsible > .game-frame-more::before {{
    content: "";
    position: absolute;
    bottom: 100%;
    left: 0;
    right: 0;
    height: 5em;
    border-radius: 0 0 12px 12px;
    background: linear-gradient(to bottom, transparent, var(--notebook-code-background,#0b0d0c));
    pointer-events: none;
}}
.game-frame-collapsible > .game-frame-more::after {{ content: "show more"; }}
.game-frame-collapsible > input:checked ~ .game-frame-more::before {{ content: none; }}
.game-frame-collapsible > input:checked ~ .game-frame-more::after {{ content: "show less"; }}
</style>"""

TAB_STYLE = """<style>
.game-tabs > input { position: absolute; opacity: 0; pointer-events: none; }
.game-tabs .game-tab-panel { display: none; }
.game-tabs > input:nth-of-type(1):checked ~ .game-tab-panel-display { display: block; }
.game-tabs > input:nth-of-type(2):checked ~ .game-tab-panel-raw { display: block; }
.game-tabs > input:nth-of-type(3):checked ~ .game-tab-panel-observation { display: block; }
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
.game-tabs > input:nth-of-type(2):checked ~ .game-tab-labels label:nth-of-type(2),
.game-tabs > input:nth-of-type(3):checked ~ .game-tab-labels label:nth-of-type(3) {
    background: var(--notebook-primary, #18352f);
    color: #ffffff;
    opacity: 1;
}
</style>"""

ACCORDION_SUMMARY_STYLE = (
    "cursor:pointer;"
    "font-size:0.85rem;"
    "font-weight:600;"
    "padding:0.45rem 0.8rem;"
    "border-radius:8px;"
    "background:var(--notebook-primary,#18352f);"
    "color:#ffffff;"
)

# Episode captions read as the h3 headings the notebooks write by hand, so a
# rendered episode sits in the same register as the surrounding markdown.
CAPTION_STYLE = f"font-family:{SANS};font-size:1.18rem;font-weight:720;color:{SECONDARY};margin:0 0 0.45rem;"


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


def is_long_output(text: str) -> bool:
    """Report whether rendered output is long enough to be worth collapsing."""
    return text.count("\n") + 1 > COLLAPSED_LINE_COUNT


def frame_html(inner_html: str, *, collapse: bool) -> HTML:
    """Wrap rendered output in the game frame, collapsing it behind a toggle when long."""
    frame = f'<div class="game-frame" style="{FRAME_STYLE}">{inner_html}</div>'
    if not collapse:
        return HTML(frame)

    toggle = f"game-frame-{next(FRAME_IDS)}"
    return HTML(
        f'<div class="game-frame-collapsible">'
        f"{COLLAPSE_STYLE}"
        f'<input type="checkbox" id="{toggle}">'
        f"{frame}"
        f'<label class="game-frame-more" for="{toggle}"></label>'
        f"</div>"
    )


def show_ansi(value: Any, *, scroll: bool = True) -> HTML:
    """Render ANSI text or bytes in the game-output frame.

    Long output collapses to about a screen behind a ``show more`` toggle;
    ``scroll=False`` renders the frame at its full height instead.
    """
    text = display_text(value)
    converter = ansi2html.Ansi2HTMLConverter(inline=True)
    html = converter.convert(text, full=False)
    return frame_html(html, collapse=scroll and is_long_output(text))


def show_text(value: Any, *, scroll: bool = True) -> HTML:
    """Render plain text in the game-output frame without interpreting ANSI state."""
    text = display_text(value)
    return frame_html(html_lib.escape(text), collapse=scroll and is_long_output(text))


def observation_text(value: Any) -> str:
    """Format one observation field for the observation tab.

    Arrays are unwrapped through ``tolist`` rather than an isinstance check, so
    this module renders Gymnasium observations without importing numpy.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return display_text(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    return cell_text(value)


def show_game_tabs(
    info: Mapping[str, Any],
    *,
    render_bytes: bytes,
    observation: Mapping[str, Any] | None = None,
    scroll: bool = True,
) -> HTML:
    """Render one game response as tabs for its display and raw protocol output.

    ``display`` shows the explicitly supplied rendered game display, while ``raw`` shows
    ``info["raw_bytes"]`` (the protocol output the action produced).
    Passing ``observation`` adds a third tab listing the fields the environment
    produced for the same step.
    """
    group = f"game-tabs-{next(TAB_GROUP_IDS)}"
    display = show_ansi(render_bytes, scroll=scroll).data
    raw = show_ansi(info["raw_bytes"], scroll=scroll).data

    inputs = [
        f'<input type="radio" name="{group}" id="{group}-display" checked>',
        f'<input type="radio" name="{group}" id="{group}-raw">',
    ]
    labels = [
        f'<label for="{group}-display">display</label>',
        f'<label for="{group}-raw">raw</label>',
    ]
    panels = [
        f'<div class="game-tab-panel game-tab-panel-display">{display}</div>',
        f'<div class="game-tab-panel game-tab-panel-raw">{raw}</div>',
    ]

    if observation is not None:
        rows = [{"Field": field, "Value": observation_text(value)} for field, value in observation.items()]
        fields = show_table(rows, mono_columns=("Field",)).data
        inputs.append(f'<input type="radio" name="{group}" id="{group}-observation">')
        labels.append(f'<label for="{group}-observation">observation</label>')
        panels.append(f'<div class="game-tab-panel game-tab-panel-observation">{fields}</div>')

    return HTML(
        f'<div class="game-tabs">'
        f"{TAB_STYLE}"
        f"{''.join(inputs)}"
        f'<div class="game-tab-labels">{"".join(labels)}</div>'
        f"{''.join(panels)}"
        f"</div>"
    )


def accordion_item_html(value: Any) -> str:
    if hasattr(value, "_repr_html_"):
        return value._repr_html_()
    if isinstance(value, str):
        return value
    # marimo's Html and UI elements stringify to markup marimo can hydrate.
    module_root = type(value).__module__.split(".")[0]
    if module_root == "marimo":
        return str(value)
    raise TypeError(f"Cannot render {type(value).__name__} in a game accordion; pass HTML or a marimo object.")


def show_frames(frames: Mapping[str, Any], *, label: str = "", selected: int = -1) -> HTML:
    """Show one of a sequence of rendered frames at a time, picked from a row of chips.

    Values may be anything with ``_repr_html_``, a marimo object, or raw HTML.
    """
    frames = dict(frames)
    if not frames:
        return HTML(f'<p style="font-family:{SANS};color:{MUTED};margin:0.4rem 0;">No frames.</p>')

    group = f"frames-{next(FRAME_SET_IDS)}"
    count = len(frames)
    chosen = range(count)[selected] + 1

    rules = "\n".join(
        f".{group} > input:nth-of-type({position}):checked ~ .frame-panels > *:nth-child({position}) "
        "{ display: block; }\n"
        f".{group} > input:nth-of-type({position}):checked ~ .frame-chips > label:nth-of-type({position}) "
        f"{{ background: {PRIMARY}; color: {PAGE}; opacity: 1; }}"
        for position in range(1, count + 1)
    )
    style = (
        f"<style>.{group} > input {{ position: absolute; opacity: 0; pointer-events: none; }}\n"
        f".{group} .frame-panels > * {{ display: none; }}\n"
        f".{group} .frame-chips {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem; "
        "margin: 0 0 0.6rem; }\n"
        f".{group} .frame-chips label {{ cursor: pointer; border: 1px solid {RULE}; border-radius: 999px; "
        f"padding: 0.16rem 0.6rem; font-family: {MONO}; font-size: 0.76rem; font-weight: 600; "
        f"color: {SECONDARY}; }}\n"
        f".{group} .frame-chips label:hover {{ background: {TINT}; }}\n{rules}</style>"
    )

    inputs = "".join(
        f'<input type="radio" name="{group}" id="{group}-{position}"{" checked" if position == chosen else ""}>'
        for position in range(1, count + 1)
    )
    chips = "".join(
        f'<label for="{group}-{position}">{html_lib.escape(str(title))}</label>'
        for position, title in enumerate(frames, start=1)
    )
    caption = f'<span style="{LABEL_STYLE}margin-right:0.2rem;">{html_lib.escape(label)}</span>' if label else ""
    panels = "".join(f"<div>{accordion_item_html(value)}</div>" for value in frames.values())

    return HTML(
        f'<div class="{group}">{style}{inputs}'
        f'<div class="frame-chips">{caption}{chips}</div>'
        f'<div class="frame-panels">{panels}</div></div>'
    )


def game_accordion(items: Mapping[str, Any], *, expanded: bool = True) -> HTML:
    """Stack titled, collapsible sections that are expanded by default.

    A ``<details>``-based replacement for ``mo.accordion``, which always starts
    closed in a static HTML export.
    """
    open_attribute = " open" if expanded else ""
    sections = "".join(
        f'<details class="game-accordion"{open_attribute} style="margin:0 0 0.6rem;">'
        f'<summary style="{ACCORDION_SUMMARY_STYLE}">{html_lib.escape(str(title))}</summary>'
        f'<div style="padding:0.6rem 0 0;">{accordion_item_html(value)}</div>'
        f"</details>"
        for title, value in items.items()
    )
    return HTML(sections)


def transition_caption(transition: Mapping[str, Any], position: int) -> str:
    """Describe one transition by its step, the action taken, and the reward.

    Every part is optional, so a bare ``{"info": ...}`` still captions cleanly.
    """
    parts = [f"Step `{transition.get('step', position)}`"]

    action = transition.get("action")
    if action is not None:
        # Gymnasium hands back numpy integers, so index by protocol rather than
        # isinstance to name those as directions too.
        indexable = hasattr(action, "__index__") and not isinstance(action, bool)
        action_index = operator.index(action) if indexable else None
        if action_index is not None and 0 <= action_index < len(DIRECTIONS):
            parts.append(f"action `{action_index}`, `move {DIRECTIONS[action_index]}`")
        else:
            parts.append(f"action `{action}`")

    reward = transition.get("reward")
    if reward is not None:
        parts.append(f"reward `{cell_text(reward)}`")

    return " · ".join(parts)


def show_episode(
    transitions: Sequence[Mapping[str, Any]],
    *,
    caption: Callable[[Mapping[str, Any], int], str] = transition_caption,
    scroll: bool = True,
) -> HTML:
    """Render one episode as a captioned game frame per transition.

    Each transition holds at least ``info``, the Gymnasium step info carrying
    the rendered bytes. The frame shows the result of the action, so
    ``next_observation`` fills the observation tab, falling back to
    ``observation``. ``caption`` receives the transition and its 1-based
    position and returns the line above each frame.
    """
    transitions = list(transitions)
    if not transitions:
        return HTML(f'<p style="font-family:{SANS};color:{MUTED};margin:0.4rem 0;">No transitions.</p>')

    blocks = []
    for position, transition in enumerate(transitions, start=1):
        observation = transition.get("next_observation", transition.get("observation"))
        frame = show_game_tabs(transition["info"], observation=observation, scroll=scroll).data
        blocks.append(
            f'<div style="margin:0 0 1.1rem;">'
            f'<div style="{CAPTION_STYLE}">{caption_html(caption(transition, position))}</div>'
            f"{frame}</div>"
        )
    return HTML(f'<div class="episode-replay">{"".join(blocks)}</div>')
