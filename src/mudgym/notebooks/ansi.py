"""Render MUD2's ANSI output as static HTML in the game frame's terminal palette."""

import html
import re
from dataclasses import dataclass

from mudgym.notebooks.style import palette_colour

ANSI_NORMAL = ("#0f1f26", "#e05a48", "#3cb884", "#f2b43c", "#315fba", "#b478d0", "#3fb9c4", "#dde0ed")
ANSI_BRIGHT = ("#5f7885", "#ff7a5c", "#43d9a3", "#ffcc63", "#315fba", "#d09ae8", "#5ee0e8", "#dde0ed")
# The game frame's surface and ink.
TERMINAL_SURFACE = palette_colour("code_background")
TERMINAL_INK = "var(--notebook-terminal-ink,#dde0ed)"

CONTROL_SEQUENCE = re.compile(r"\x1b\[([0-9;]*)([ -/]*[@-~])")


@dataclass
class Colour:
    index: int | None = None
    bright: bool = False
    direct: str | None = None


@dataclass
class Rendition:
    foreground: Colour | None = None
    background: Colour | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    inverse: bool = False


def palette(colour: Colour | None, bold_foreground: bool) -> str | None:
    if colour is None:
        return None
    if colour.direct is not None:
        return colour.direct
    return (ANSI_BRIGHT if colour.bright or bold_foreground else ANSI_NORMAL)[colour.index]


def xterm256(code: int) -> str | None:
    if not 0 <= code <= 255:
        return None
    if code < 16:
        return (ANSI_NORMAL if code < 8 else ANSI_BRIGHT)[code % 8]
    if code >= 232:
        level = 8 + (code - 232) * 10
        return f"rgb({level}, {level}, {level})"
    value = code - 16
    red, green, blue = value // 36, (value % 36) // 6, value % 6
    component = [0 if level == 0 else 55 + level * 40 for level in (red, green, blue)]
    return f"rgb({component[0]}, {component[1]}, {component[2]})"


def apply_parameters(rendition: Rendition, raw: str) -> Rendition:
    """Return the rendition after one SGR sequence's parameters."""
    parameters = [int(value) if value else 0 for value in raw.split(";")] if raw else [0]
    index = 0
    while index < len(parameters):
        code = parameters[index]
        if code == 0:
            rendition = Rendition()
        elif code == 1:
            rendition.bold = True
        elif code == 2:
            rendition.dim = True
        elif code == 3:
            rendition.italic = True
        elif code == 4:
            rendition.underline = True
        elif code == 7:
            rendition.inverse = True
        elif code == 22:
            rendition.bold = rendition.dim = False
        elif code == 23:
            rendition.italic = False
        elif code == 24:
            rendition.underline = False
        elif code == 27:
            rendition.inverse = False
        elif 30 <= code <= 37:
            rendition.foreground = Colour(index=code - 30)
        elif code == 39:
            rendition.foreground = None
        elif 40 <= code <= 47:
            rendition.background = Colour(index=code - 40)
        elif code == 49:
            rendition.background = None
        elif 90 <= code <= 97:
            rendition.foreground = Colour(index=code - 90, bright=True)
        elif 100 <= code <= 107:
            rendition.background = Colour(index=code - 100, bright=True)
        elif code in (38, 48) and parameters[index + 1 : index + 2] == [5]:
            colour = xterm256(parameters[index + 2]) if index + 2 < len(parameters) else None
            if colour is not None:
                setattr(rendition, "foreground" if code == 38 else "background", Colour(direct=colour))
            index += 2
        elif code in (38, 48) and parameters[index + 1 : index + 2] == [2]:
            channels = parameters[index + 2 : index + 5]
            if len(channels) == 3 and all(0 <= channel <= 255 for channel in channels):
                colour = Colour(direct=f"rgb({channels[0]}, {channels[1]}, {channels[2]})")
                setattr(rendition, "foreground" if code == 38 else "background", colour)
            index += 4
        index += 1
    return rendition


def css(rendition: Rendition) -> str:
    foreground = palette(rendition.foreground, rendition.bold)
    background = palette(rendition.background, False)
    # Palette black as a background is the terminal surface, not a lighter rectangle behind each run.
    if background == ANSI_NORMAL[0]:
        background = TERMINAL_SURFACE
    if rendition.inverse:
        foreground, background = background or TERMINAL_SURFACE, foreground or TERMINAL_INK
    declarations = []
    if foreground:
        declarations.append(f"color: {foreground}")
    if background:
        declarations.append(f"background-color: {background}")
    if rendition.bold:
        declarations.append("font-weight: 700")
    if rendition.dim:
        declarations.append("opacity: 0.72")
    if rendition.italic:
        declarations.append("font-style: italic")
    if rendition.underline:
        declarations.append("text-decoration: underline")
    return "; ".join(declarations)


def ansi_html(text: str) -> str:
    """Render ANSI text as escaped spans. Control sequences other than SGR are dropped."""
    rendition = Rendition()
    parts = []

    def append(run: str) -> None:
        if not run:
            return
        style = css(rendition)
        escaped = html.escape(run, quote=False)
        parts.append(f'<span style="{style}">{escaped}</span>' if style else escaped)

    offset = 0
    for match in CONTROL_SEQUENCE.finditer(text):
        append(text[offset : match.start()])
        if match.group(2).endswith("m"):
            rendition = apply_parameters(rendition, match.group(1))
        offset = match.end()
    append(text[offset:])
    return "".join(parts)
