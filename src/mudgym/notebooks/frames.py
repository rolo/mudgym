"""Game-output frames, tabs, accordions, and episode replays."""

import html as html_lib
import itertools
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mudgym.db.directions import DIRECTIONS
from mudgym.featurizers.ansi import INPUT_STYLE
from mudgym.featurizers.strings import decode_text_bytes
from mudgym.notebooks.ansi import TERMINAL_FONT, TERMINAL_INK, TERMINAL_MUTED, TERMINAL_SURFACE, ansi_html
from mudgym.notebooks.style import (
    HTML,
    LABEL_STYLE,
    MONO,
    PAGE,
    PRIMARY,
    RADIUS,
    RULE,
    SANS,
    SECONDARY,
    THEME,
    TINT,
    caption_html,
    cell_text,
    empty_html,
)
from mudgym.notebooks.tables import show_table

# Every widget here carries its own inline styles and <style> block. That is what makes it render inside marimo
# components such as mo.accordion, whose content sits behind a boundary a page stylesheet does not reach, and in static
# HTML exports with no kernel attached. Duplicated blocks are harmless.
FRAME_STYLE = (
    f"background:{TERMINAL_SURFACE};"
    f"color:{TERMINAL_INK};"
    f"font-family:{TERMINAL_FONT};"
    "font-size:14px;"
    "line-height:1.5;"
    "padding:1.1em 1.2em;"
    "margin:0;"
    "white-space:pre-wrap;"
    "overflow-wrap:anywhere;"
    f"border:1px solid {THEME['primary']};"
)

# Long output collapses rather than scrolling: a frame with its own scrollbar inside the notebook's is awkward to drive
# and hides how much output there is.
COLLAPSED_LINE_COUNT = 24

# A frame fixed at a height scrolls as a terminal does, opening on its newest output with the earlier output above it.
# Reversing the column starts the scroll at the bottom, and the inner auto margin keeps output that fits at the top.
FIXED_FRAME_STYLE = (
    f"overflow-y:auto;display:flex;flex-direction:column-reverse;scrollbar-color:{TERMINAL_MUTED} {TERMINAL_SURFACE};"
)

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
    background: linear-gradient(to bottom, transparent, {TERMINAL_SURFACE});
    pointer-events: none;
}}
.game-frame-collapsible > .game-frame-more::after {{ content: "show more"; }}
.game-frame-collapsible > input:checked ~ .game-frame-more::before {{ content: none; }}
.game-frame-collapsible > input:checked ~ .game-frame-more::after {{ content: "show less"; }}
</style>"""

TAB_STYLE = f"""<style>
.game-tabs > input {{ position: absolute; opacity: 0; pointer-events: none; }}
.game-tabs .game-tab-panel {{ display: none; }}
.game-tabs > input:nth-of-type(1):checked ~ .game-tab-panel-display {{ display: block; }}
.game-tabs > input:nth-of-type(2):checked ~ .game-tab-panel-raw {{ display: block; }}
.game-tabs > input:nth-of-type(3):checked ~ .game-tab-panel-observation {{ display: block; }}
.game-tabs .game-tab-labels {{ display: flex; gap: 0.4rem; margin: 0 0 0.5rem; }}
.game-tabs .game-tab-labels label {{
    cursor: pointer;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.25rem 0.75rem;
    border-radius: {RADIUS};
    border: 1px solid {PRIMARY};
    color: {PRIMARY};
    opacity: 0.65;
}}
.game-tabs > input:nth-of-type(1):checked ~ .game-tab-labels label:nth-of-type(1),
.game-tabs > input:nth-of-type(2):checked ~ .game-tab-labels label:nth-of-type(2),
.game-tabs > input:nth-of-type(3):checked ~ .game-tab-labels label:nth-of-type(3) {{
    background: {PRIMARY};
    color: {PAGE};
    opacity: 1;
}}
</style>"""

ACCORDION_SUMMARY_STYLE = (
    "cursor:pointer;"
    "font-size:0.85rem;"
    "font-weight:600;"
    "padding:0.45rem 0.8rem;"
    f"border-radius:{RADIUS};"
    f"background:{PRIMARY};"
    f"color:{PAGE};"
)

# Episode captions read as the h3 headings the notebooks write by hand, so a rendered episode sits in the same register
# as the surrounding markdown.
CAPTION_STYLE = f"font-family:{SANS};font-size:1.18rem;font-weight:720;color:{SECONDARY};margin:0 0 0.45rem;"

# MUD2's prompt is a bright blue asterisk, and the player's command follows it in the input style.
PROMPT = "\x1b[1;34;40m*" + INPUT_STYLE.decode("latin-1")

# A transcript step's note sits at the end of its command's line, in a dimmer terminal ink.
NOTE_STYLE = f"float:right;margin-left:1rem;font-size:0.78rem;color:{TERMINAL_MUTED};"


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


def frame_html(inner_html: str, *, collapse: bool, lines: int | None = None) -> HTML:
    """Wrap rendered output in the game frame, collapsing it behind a toggle when long, or fixing it ``lines`` tall."""
    if lines is None:
        frame = f'<div class="game-frame" style="{FRAME_STYLE}">{inner_html}</div>'
    else:
        frame_id = f"game-frame-fixed-{next(FRAME_IDS)}"
        frame = (
            f'<div class="game-frame" id="{frame_id}" style="{FRAME_STYLE}height:{lines * 1.5:g}em;{FIXED_FRAME_STYLE}">'
            f'<div style="margin-bottom:auto;">{inner_html}</div></div>'
            # Where scripts run, also scroll to the newest output, for browsers that open a reversed scroll at its top.
            f"<script>(function () {{ var frame = document.getElementById({frame_id!r}); "
            f"if (frame) frame.scrollTop = frame.scrollHeight; }})();</script>"
        )
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

    Long output collapses to about a screen behind a ``show more`` toggle; ``scroll=False`` renders the frame at its
    full height instead.
    """
    text = display_text(value)
    return frame_html(ansi_html(text), collapse=scroll and is_long_output(text))


def show_text(value: Any, *, scroll: bool = True) -> HTML:
    """Render plain text in the game-output frame without interpreting ANSI state."""
    text = display_text(value)
    return frame_html(html_lib.escape(text), collapse=scroll and is_long_output(text))


def observation_text(value: Any) -> str:
    """Format one observation field for the observation tab.

    Arrays are unwrapped through ``tolist`` rather than an isinstance check, so this module renders Gymnasium
    observations without importing numpy.
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
    render_bytes: bytes | None = None,
    observation: Mapping[str, Any] | None = None,
    scroll: bool = True,
) -> HTML:
    """Render one game response as tabs for its display and raw protocol output.

    ``display`` shows ``info["render_bytes"]``, unless an explicit override is supplied, while ``raw`` shows
    ``info["raw_bytes"]`` (the protocol output the action produced). Passing ``observation`` adds a third tab listing
    the fields the environment produced for the same step.
    """
    group = f"game-tabs-{next(TAB_GROUP_IDS)}"
    display_bytes = info["render_bytes"] if render_bytes is None else render_bytes
    display = show_ansi(display_bytes, scroll=scroll).data
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
        return empty_html("No frames.")

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
        f".{group} .frame-chips label {{ cursor: pointer; border: 1px solid {RULE}; border-radius: {RADIUS}; "
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

    A ``<details>``-based replacement for ``mo.accordion``, which always starts closed in a static HTML export.
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


def action_command(action: Any) -> str:
    """The command an action sent: a direction index as its move, and anything else, such as text, as it is."""
    # Gymnasium hands back numpy integers, so index by protocol rather than isinstance to name those as directions too.
    if hasattr(action, "__index__") and not isinstance(action, bool) and 0 <= operator.index(action) < len(DIRECTIONS):
        return f"move {DIRECTIONS[operator.index(action)]}"
    return str(action)


def transition_caption(transition: Mapping[str, Any], position: int) -> str:
    """Describe one transition by its step, the action taken, and the reward.

    Every part is optional, so a bare ``{"info": ...}`` still captions cleanly.
    """
    parts = [f"Step `{transition.get('step', position)}`"]

    action = transition.get("action")
    if action is not None:
        command = action_command(action)
        parts.append(f"action `{action}`" if command == str(action) else f"action `{action}`, `{command}`")

    reward = transition.get("reward")
    if reward is not None:
        parts.append(f"reward `{cell_text(reward)}`")

    return " · ".join(parts)


def episode_transitions(episode: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """An episode's transitions, given either the transitions themselves or a mapping holding them as ``transitions``."""
    return list(episode["transitions"] if isinstance(episode, Mapping) else episode)


def show_episode(
    episode: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    caption: Callable[[Mapping[str, Any], int], str] = transition_caption,
    scroll: bool = True,
) -> HTML:
    """Render one episode as a captioned game frame per transition.

    ``episode`` is its transitions, or a mapping holding them as ``transitions``. Each transition holds at least
    ``info``, the Gymnasium step info carrying the rendered bytes. The frame shows the result of the action, so
    ``next_observation`` fills the observation tab, falling back to ``observation``. ``caption`` receives the transition
    and its 1-based position and returns the line above each frame.
    """
    transitions = episode_transitions(episode)
    if not transitions:
        return empty_html("No transitions.")

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


def transition_note(transition: Mapping[str, Any], position: int) -> str:
    """A transition's step and reward, as a transcript shows them at the end of its command's line."""
    note = f"step {transition.get('step', position)}"
    reward = transition.get("reward")
    return note if reward is None else f"{note} · reward {reward:+.2f}"


def show_episode_transcript(
    episode: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    scroll: bool = True,
    lines: int | None = None,
) -> HTML:
    """Render an episode in one game frame, as the terminal shows it.

    Takes the same episode as ``show_episode``. Each command follows MUD2's prompt with its step and reward at the end
    of the line, and the game's reply from ``info`` comes under it. ``title`` heads the frame as ``show_episode``
    captions its steps. ``lines`` fixes the frame at that many lines tall, scrolling and opening on the latest output, so
    a transcript redrawn while an episode plays holds still.
    """
    transitions = episode_transitions(episode)
    if not transitions:
        return empty_html("No transitions.")

    texts = [
        f"{PROMPT}{action_command(transition['action'])}\n{display_text(transition['info']['render_bytes'])}"
        for transition in transitions
    ]
    rendered = "".join(
        f'<div style="margin-top:{"0.6rem" if position > 1 else "0"};">'
        f'<span style="{NOTE_STYLE}">{html_lib.escape(transition_note(transition, position))}</span>'
        f"{ansi_html(text)}</div>"
        for position, (transition, text) in enumerate(zip(transitions, texts, strict=True), start=1)
    )
    collapse = scroll and lines is None and is_long_output("\n".join(texts))
    frame = frame_html(rendered, collapse=collapse, lines=lines)
    if title is None:
        return frame
    return HTML(f'<div style="{CAPTION_STYLE}">{caption_html(title)}</div>{frame.data}')
