"""Room-map rendering for notebook output."""

import html as html_lib
import itertools
from collections.abc import Iterable, Mapping, Sequence

from mudgym.db.directions import DIRECTION_INDEX_BY_NAME
from mudgym.notebooks.style import (
    ACCENT,
    HTML,
    INK,
    MONO,
    MUTED,
    PAGE,
    PANEL_STYLE,
    PRIMARY,
    SANS,
    SECONDARY,
)

# Every drawn map needs its own arrowhead marker id.
MAP_IDS = itertools.count(1)

# Where each direction pushes the next room on the drawing grid. The vertical
# moves and the two named MUD2 exits borrow the nearest compass square.
MAP_OFFSET_BY_DIRECTION = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
    "northeast": (1, -1),
    "southeast": (1, 1),
    "southwest": (-1, 1),
    "northwest": (-1, -1),
    "up": (0, -1),
    "down": (0, 1),
    "in": (-1, 0),
    "out": (1, 0),
    "over": (1, -1),
    "swampward": (-1, 1),
}
MAP_FALLBACK_OFFSET = (1, 0)

MAP_LIVE_FILL = f"color-mix(in srgb, {ACCENT} 40%, {PAGE})"
MAP_ROOM_FILL = f"color-mix(in srgb, {SECONDARY} 8%, {PAGE})"


def wrap_room_name(room: str, line_length: int) -> list[str]:
    """Split a room name over at most two lines of roughly ``line_length``."""
    lines: list[str] = []
    current = ""
    for word in room.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > line_length:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], f"{lines[1]}…"]
    return lines or ["unknown"]


def free_grid_cell(anchor, offset, occupied, search_limit):
    """Return the cell one step along ``offset``, or the closest free one to it."""
    preferred = (anchor[0] + offset[0], anchor[1] + offset[1])
    if preferred not in occupied:
        return preferred
    for radius in range(1, search_limit + 2):
        ring = [
            (delta_x, delta_y)
            for delta_x in range(-radius, radius + 1)
            for delta_y in range(-radius, radius + 1)
            if max(abs(delta_x), abs(delta_y)) == radius
        ]
        # Prefer cells that keep going the way the exit pointed, and that stay
        # close to its line, so a corridor of rooms still reads as a corridor.
        ring.sort(
            key=lambda cell: (
                (cell[0] * offset[0] + cell[1] * offset[1]) < 0,
                abs(cell[0] * offset[1] - cell[1] * offset[0]),
                -(cell[0] * offset[0] + cell[1] * offset[1]),
                abs(cell[0]) + abs(cell[1]),
                cell[1],
                cell[0],
            )
        )
        for delta_x, delta_y in ring:
            candidate = (preferred[0] + delta_x, preferred[1] + delta_y)
            if candidate not in occupied:
                return candidate
    raise RuntimeError("Could not place a room on the map grid")


def place_rooms(ordered_rooms: Sequence[str], edges: Sequence[tuple[str, str, str]]) -> dict[str, tuple[int, int]]:
    """Lay the rooms out on a compass grid, following the moves that linked them."""
    rank = {room: index for index, room in enumerate(ordered_rooms)}
    placement_order = sorted(
        edges,
        key=lambda edge: (
            rank[edge[0]],
            DIRECTION_INDEX_BY_NAME.get(edge[1], len(DIRECTION_INDEX_BY_NAME)),
            rank[edge[2]],
        ),
    )

    positions = {ordered_rooms[0]: (0, 0)}
    occupied = {(0, 0)}
    unplaced = set(ordered_rooms[1:])
    while unplaced:
        progressed = False
        for source, direction, destination in placement_order:
            offset = MAP_OFFSET_BY_DIRECTION.get(direction, MAP_FALLBACK_OFFSET)
            if source in positions and destination in unplaced:
                anchor, step = positions[source], offset
            elif destination in positions and source in unplaced:
                anchor, step = positions[destination], (-offset[0], -offset[1])
            else:
                continue
            room = destination if destination in unplaced else source
            positions[room] = free_grid_cell(anchor, step, occupied, len(ordered_rooms))
            occupied.add(positions[room])
            unplaced.remove(room)
            progressed = True
        if not progressed:
            # Rooms nothing links to yet: park them around the anchor.
            room = min(unplaced, key=rank.get)
            offsets = tuple(MAP_OFFSET_BY_DIRECTION.values())
            positions[room] = free_grid_cell((0, 0), offsets[rank[room] % len(offsets)], occupied, len(ordered_rooms))
            occupied.add(positions[room])
            unplaced.remove(room)

    return positions


def show_room_map(
    edges: Iterable[Sequence[str]],
    *,
    occupants: Mapping[str, Sequence[str]] | None = None,
    occupants_label: str = "Agents here",
    new_rooms: Iterable[str] = (),
    new_rooms_label: str = "New this turn",
    edge_labels: Mapping[tuple[str, str, str], str] | None = None,
    caption: str | None = None,
) -> HTML:
    """Draw discovered rooms, and the moves that linked them, as an SVG map.

    ``edges`` are ``(source, direction, destination)`` triples from moves that
    actually changed room. ``occupants`` marks rooms currently holding agents,
    named in the legend by ``occupants_label``; ``new_rooms`` outlines a set
    worth picking out, named by ``new_rooms_label``. An arrow is captioned with
    its direction unless ``edge_labels`` gives that exact triple different text,
    which is how a map annotates a move with what it paid or what it is worth.

    Parsed observations expose room names rather than ids, so rooms sharing a
    displayed name are one node here.
    """
    occupants = {room: list(names) for room, names in (occupants or {}).items() if room}
    edge_labels = dict(edge_labels or {})
    new_rooms = {room for room in new_rooms if room}
    clean_edges = [
        (str(source), str(direction), str(destination))
        for source, direction, destination in edges
        if source and destination and source != destination
    ]

    ordered_rooms = list(
        dict.fromkeys(
            [room for edge in clean_edges for room in (edge[0], edge[2])] + list(occupants) + sorted(new_rooms)
        )
    )
    if not ordered_rooms:
        return HTML(f'<p style="font-family:{SANS};color:{MUTED};margin:0.4rem 0;">No rooms discovered yet.</p>')

    # Everything shrinks as the map fills up, so a long mission still fits.
    density = min(1.0, max(0.0, (len(ordered_rooms) - 4) / 28))
    room_width = round(186 - 74 * density)
    room_height = round(78 - 20 * density)
    room_font = 13 - 3 * density
    small_font = 10.5 - 1.5 * density
    corner = round(12 - 4 * density)
    column_pitch = room_width + round(50 - 14 * density)
    row_pitch = room_height + round(54 - 14 * density)
    side_padding = 40
    top_padding = 78

    grid = place_rooms(ordered_rooms, clean_edges)
    minimum_column = min(cell[0] for cell in grid.values())
    minimum_row = min(cell[1] for cell in grid.values())
    content_width = room_width + (max(cell[0] for cell in grid.values()) - minimum_column) * column_pitch
    content_height = room_height + (max(cell[1] for cell in grid.values()) - minimum_row) * row_pitch
    svg_width = max(560, 2 * side_padding + content_width)
    svg_height = max(240, top_padding + content_height + 66)
    left_origin = (svg_width - content_width) / 2

    positions = {
        room: (
            left_origin + (column - minimum_column) * column_pitch,
            top_padding + (row - minimum_row) * row_pitch,
        )
        for room, (column, row) in grid.items()
    }

    def edge_endpoint(start, end):
        """Meet the room rectangle's edge instead of stopping at its centre."""
        delta_x, delta_y = end[0] - start[0], end[1] - start[1]
        horizontal = (room_width / 2) / abs(delta_x) if delta_x else float("inf")
        vertical = (room_height / 2) / abs(delta_y) if delta_y else float("inf")
        scale = min(horizontal, vertical)
        return start[0] + delta_x * scale, start[1] + delta_y * scale

    # One arrow per room pair, captioned with every direction that linked them,
    # so a pair walked both ways is drawn once rather than twice on itself.
    labels_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for edge in clean_edges:
        source, direction, destination = edge
        labels_by_pair.setdefault((source, destination), {})[direction] = str(edge_labels.get(edge, direction))

    map_id = f"room-map-{next(MAP_IDS)}"
    line_style = f"stroke:{SECONDARY};stroke-opacity:0.5;stroke-width:2;"
    label_style = (
        f"font-family:{MONO};fill:{SECONDARY};paint-order:stroke;stroke:{PAGE};stroke-width:4px;stroke-linejoin:round;"
    )
    edge_elements = []
    edge_captions = []
    for (source, destination), pair_labels in sorted(labels_by_pair.items()):
        source_x, source_y = positions[source]
        destination_x, destination_y = positions[destination]
        start = edge_endpoint(
            (source_x + room_width / 2, source_y + room_height / 2),
            (destination_x + room_width / 2, destination_y + room_height / 2),
        )
        end = edge_endpoint(
            (destination_x + room_width / 2, destination_y + room_height / 2),
            (source_x + room_width / 2, source_y + room_height / 2),
        )
        label = ", ".join(
            pair_labels[direction]
            for direction in sorted(pair_labels, key=lambda name: DIRECTION_INDEX_BY_NAME.get(name, 99))
        )
        # Nudge the two labels of a mapped pair apart so they never collide.
        label_offset = -8 if ordered_rooms.index(source) < ordered_rooms.index(destination) else 14
        edge_elements.append(
            f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            f'style="{line_style}" marker-end="url(#{map_id}-arrow)" />'
        )
        # Captions are drawn after the rooms: a label longer than the gap
        # between two rooms would otherwise be painted over by the room it
        # reaches into and read as a truncated word.
        edge_captions.append(
            f'<text x="{(start[0] + end[0]) / 2:.1f}" y="{(start[1] + end[1]) / 2 + label_offset:.1f}" '
            f'text-anchor="middle" font-size="{small_font:.1f}" style="{label_style}">'
            f"{html_lib.escape(label)}</text>"
        )

    room_elements = []
    for room in ordered_rooms:
        x, y = positions[room]
        here = sorted(occupants.get(room, ()))
        fill = MAP_LIVE_FILL if here else MAP_ROOM_FILL
        stroke = ACCENT if room in new_rooms else PRIMARY
        stroke_width = 3 if room in new_rooms else 1.5
        lines = wrap_room_name(room, max(12, round(room_width / 7.2)))
        line_height = room_font + 2
        centre_y = y + room_height * (0.38 if here else 0.5)
        first_line_y = centre_y - (len(lines) - 1) * line_height / 2 + room_font / 3
        texts = [
            f'<text x="{x + room_width / 2:.1f}" y="{first_line_y + index * line_height:.1f}" text-anchor="middle" '
            f'font-size="{room_font:.1f}" style="font-family:{SANS};font-weight:640;fill:{INK};">'
            f"{html_lib.escape(line)}</text>"
            for index, line in enumerate(lines)
        ]
        if here:
            # Only as many names as the room box can hold, then a count.
            named = 2 if room_width > 150 else 1
            shown = ", ".join(here[:named]) + (f" +{len(here) - named}" if len(here) > named else "")
            texts.append(
                f'<text x="{x + room_width / 2:.1f}" y="{y + room_height - 12:.1f}" text-anchor="middle" '
                f'font-size="{small_font:.1f}" style="font-family:{MONO};fill:{PRIMARY};">'
                f"{html_lib.escape(shown)}</text>"
            )
        room_elements.append(
            f"<g><title>{html_lib.escape(room)}</title>"
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{room_width}" height="{room_height}" rx="{corner}" '
            f'style="fill:{fill};stroke:{stroke};stroke-width:{stroke_width};" />'
            f"{''.join(texts)}</g>"
        )

    compass_x, compass_y = svg_width - 52, 40
    compass = (
        f'<g font-size="11" text-anchor="middle" style="font-family:{MONO};font-weight:700;fill:{MUTED};">'
        f'<line x1="{compass_x}" y1="{compass_y - 12}" x2="{compass_x}" y2="{compass_y + 12}" style="{line_style}" />'
        f'<line x1="{compass_x - 12}" y1="{compass_y}" x2="{compass_x + 12}" y2="{compass_y}" style="{line_style}" />'
        f'<text x="{compass_x}" y="{compass_y - 17}">N</text>'
        f'<text x="{compass_x + 20}" y="{compass_y + 4}">E</text>'
        f'<text x="{compass_x}" y="{compass_y + 26}">S</text>'
        f'<text x="{compass_x - 20}" y="{compass_y + 4}">W</text></g>'
    )
    legend_y = svg_height - 24
    legend_text_style = f"font-family:{SANS};fill:{MUTED};"
    # Start the second entry past the first one's text, so renaming either keeps
    # a gap instead of printing one legend on top of the other.
    second_entry_x = side_padding + 24 + max(70, round(6.4 * len(occupants_label))) + 24
    legend = (
        f'<rect x="{side_padding}" y="{legend_y - 13}" width="16" height="16" rx="4" '
        f'style="fill:{MAP_LIVE_FILL};stroke:{PRIMARY};stroke-width:1.5;" />'
        f'<text x="{side_padding + 24}" y="{legend_y}" font-size="11.5" style="{legend_text_style}">{html_lib.escape(occupants_label)}</text>'
        f'<rect x="{second_entry_x}" y="{legend_y - 13}" width="16" height="16" rx="4" '
        f'style="fill:{MAP_ROOM_FILL};stroke:{ACCENT};stroke-width:3;" />'
        f'<text x="{second_entry_x + 24}" y="{legend_y}" font-size="11.5" style="{legend_text_style}">{html_lib.escape(new_rooms_label)}</text>'
    )

    caption_markup = (
        f'<p style="font-family:{SANS};font-size:0.85rem;color:{MUTED};margin:0.55rem 0.2rem 0.1rem;">'
        f"{html_lib.escape(caption)}</p>"
        if caption
        else ""
    )
    # A big map scrolls inside its panel rather than shrinking its labels to
    # nothing, and a small one stays at its drawn size.
    return HTML(
        f'<div class="notebook-map" style="{PANEL_STYLE}padding:0.5rem 0.6rem 0.35rem;overflow-x:auto;'
        f'margin:0.3rem 0 0.5rem;">'
        f'<svg viewBox="0 0 {svg_width:.0f} {svg_height:.0f}" role="img" aria-label="Discovered room map" '
        f'style="display:block;height:auto;margin-inline:auto;width:100%;max-width:{svg_width:.0f}px;'
        f'min-width:{min(svg_width, 980):.0f}px;">'
        f'<defs><marker id="{map_id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:{SECONDARY};fill-opacity:0.5;" />'
        f"</marker></defs>{''.join(edge_elements)}{''.join(room_elements)}{''.join(edge_captions)}"
        f"{compass}{legend}</svg>"
        f"{caption_markup}</div>"
    )
