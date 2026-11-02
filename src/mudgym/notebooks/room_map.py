"""Room-map rendering for notebook output."""

import html as html_lib
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence

from mudgym.db.directions import DIRECTION_INDEX_BY_NAME
from mudgym.notebooks.map_layout import layout_world_from_edges
from mudgym.notebooks.style import (
    ACCENT,
    BLOCK_MARGIN,
    CORNER_RADIUS,
    HTML,
    INK,
    MONO,
    MUTED,
    PAGE,
    PANEL_STYLE,
    PRIMARY,
    SANS,
    SECONDARY,
    empty_html,
)

# Every drawn map needs its own arrowhead marker id.
MAP_IDS = itertools.count(1)

MAP_LIVE_FILL = f"color-mix(in srgb, {ACCENT} 40%, {PAGE})"
MAP_ROOM_FILL = f"color-mix(in srgb, {SECONDARY} 8%, {PAGE})"

# Where an edge label may sit along its edge, from the middle outwards, and how far it may step to either side at each,
# in label heights.
LABEL_ANCHORS = (0.5, 0.38, 0.62, 0.26, 0.74, 0.14, 0.86)
LABEL_STRIDES = (0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3)
# Where a label may go when nothing beside its edge is clear: points round the middle of the edge, nearest first, out to
# about two rooms either way.
NEARBY_OFFSETS = sorted(
    ((column * 10, row * 10) for column in range(-30, 31) for row in range(-20, 21)),
    key=lambda offset: math.hypot(*offset),
)


def overlap_area(first: Sequence[float], second: Sequence[float]) -> float:
    """How much two ``(x, y, width, height)`` boxes overlap."""
    width = min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0])
    height = min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1])
    return max(0.0, width) * max(0.0, height)


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


def show_room_map(
    edges: Iterable[Sequence[str]] = (),
    *,
    graph: Mapping[str, Mapping] | None = None,
    occupants: Mapping[str, Sequence[str]] | None = None,
    occupants_label: str = "Agents here",
    new_rooms: Iterable[str] = (),
    new_rooms_label: str = "New this turn",
    edge_labels: Mapping[tuple[str, str, str], str] | None = None,
    caption: str | None = None,
) -> HTML:
    """Draw rooms and directed moves as a static SVG map.

    ``edges`` contains ``(source, direction, destination)`` triples. Without ``graph``, the room strings are both
    identifiers and display names. With ``graph``, keys identify rooms and each node's ``name`` supplies its label,
    matching the interactive notebook maps. If no route is supplied, show every graph room and exit.

    ``occupants`` marks rooms containing agents and ``new_rooms`` highlights discoveries. ``edge_labels`` can replace
    each direction caption with a reward or action value. Labels and highlights never change room identity.
    """
    graph = graph or {}
    room_labels = {room_id: str(room.get("name", room_id)) for room_id, room in graph.items()}
    for room in graph.values():
        for destination in room.get("exits", {}).values():
            room_labels.setdefault(destination["id"], str(destination.get("name", destination["id"])))
    edges = list(edges)
    whole_graph = not edges
    if whole_graph:
        edges = [
            (room_id, direction, destination["id"])
            for room_id, room in graph.items()
            for direction, destination in room.get("exits", {}).items()
        ]
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
            [room for edge in clean_edges for room in (edge[0], edge[2])]
            + (list(graph) if whole_graph else [])
            + list(occupants)
            + sorted(new_rooms)
        )
    )
    if not ordered_rooms:
        return empty_html("No rooms discovered yet.")

    # Everything shrinks as the map fills up, so a long mission still fits.
    density = min(1.0, max(0.0, (len(ordered_rooms) - 4) / 28))
    room_width = round(186 - 74 * density)
    room_height = round(78 - 20 * density)
    room_font = 13 - 3 * density
    small_font = 10.5 - 1.5 * density
    side_padding = 40
    top_padding = 78

    # The layout puts a compass step two grid units away. Up, down and a crowded room's exits can land one unit away, so
    # a unit clears a room box both ways.
    grid = layout_world_from_edges(ordered_rooms, clean_edges)
    column_unit = room_width + 8
    row_unit = room_height + 8
    minimum_column = min(cell[0] for cell in grid.values())
    minimum_row = min(cell[1] for cell in grid.values())
    content_width = room_width + (max(cell[0] for cell in grid.values()) - minimum_column) * column_unit
    content_height = room_height + (max(cell[1] for cell in grid.values()) - minimum_row) * row_unit
    svg_width = max(560, 2 * side_padding + content_width)
    svg_height = max(240, top_padding + content_height + 66)
    left_origin = (svg_width - content_width) / 2

    positions = {
        room: (
            left_origin + (column - minimum_column) * column_unit,
            top_padding + (row - minimum_row) * row_unit,
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

    # One arrow per room pair, captioned with every direction that linked them, so a pair walked both ways is drawn once
    # rather than twice on itself.
    labels_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for edge in clean_edges:
        source, direction, destination = edge
        labels_by_pair.setdefault((source, destination), {})[direction] = str(edge_labels.get(edge, direction))

    map_id = f"room-map-{next(MAP_IDS)}"
    line_style = f"stroke:{SECONDARY};stroke-opacity:0.5;stroke-width:2;"
    label_style = f"font-family:{MONO};fill:{SECONDARY};"

    # Each label sits on an opaque plate, so it stays readable where the gap between two rooms is narrower than the
    # word. MONO advances 0.6em per character, which is enough to size the plate without measuring text.
    label_padding_x = 5
    label_padding_y = 2.5
    label_height = small_font + 2 * label_padding_y
    label_separation = label_height + 3

    # Labels collide when a pair is walked both ways, when two unrelated edges cross near the same point, and when an
    # edge passes a room on its way. The rooms, the compass and every label already placed count as taken, and a label
    # has to stay inside the drawing, above the legend.
    compass_x, compass_y = svg_width - 52, 40
    legend_y = svg_height - 24
    # Taken boxes are filed by the cells of a coarse grid they touch, so a spot is tested only against boxes near it and
    # a map of the whole Land stays quick to draw.
    grid_cell = 100
    taken_by_cell: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}

    def cells_of(box):
        return itertools.product(
            range(math.floor(box[0] / grid_cell), math.floor((box[0] + box[2]) / grid_cell) + 1),
            range(math.floor(box[1] / grid_cell), math.floor((box[1] + box[3]) / grid_cell) + 1),
        )

    def take(box) -> None:
        for cell in cells_of(box):
            taken_by_cell.setdefault(cell, []).append(box)

    def inside(box) -> bool:
        return box[0] >= 0 and box[1] >= 0 and box[0] + box[2] <= svg_width and box[1] + box[3] <= legend_y - 16

    def near(box):
        return {other for cell in cells_of(box) for other in taken_by_cell.get(cell, ())}

    def clear(box) -> bool:
        return inside(box) and not any(overlap_area(box, other) for other in near(box))

    def cover_of(box) -> float:
        return sum(overlap_area(box, other) for other in near(box)) if inside(box) else math.inf

    for x, y in positions.values():
        take((x, y, room_width, room_height))
    take((compass_x - 30, compass_y - 30, 60, 62))

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
        edge_elements.append(
            f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            f'style="{line_style}" marker-end="url(#{map_id}-arrow)" />'
        )

        # The first clear spot beside the edge, trying the middle of the edge before points further along it, then the
        # nearest clear spot round the middle. A label never sits on a room's name just to stay by its arrow, and only
        # where nothing nearby is clear does it take the spot that covers least.
        plate_width = 0.62 * small_font * len(label) + 2 * label_padding_x
        span = math.hypot(end[0] - start[0], end[1] - start[1]) or 1.0
        step_x = -(end[1] - start[1]) / span * label_separation
        step_y = (end[0] - start[0]) / span * label_separation
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        beside_edge = [
            (
                start[0] + (end[0] - start[0]) * along + step_x * stride - plate_width / 2,
                start[1] + (end[1] - start[1]) * along + step_y * stride - label_height / 2,
                plate_width,
                label_height,
            )
            for along in LABEL_ANCHORS
            for stride in LABEL_STRIDES
        ]
        round_the_middle = (
            (
                midpoint[0] + offset_x - plate_width / 2,
                midpoint[1] + offset_y - label_height / 2,
                plate_width,
                label_height,
            )
            for offset_x, offset_y in NEARBY_OFFSETS
        )
        plate = next(
            (candidate for candidate in itertools.chain(beside_edge, round_the_middle) if clear(candidate)),
            None,
        ) or min(beside_edge, key=cover_of)
        take(plate)

        # Captions are drawn after the rooms: a label longer than the gap between two rooms would otherwise be painted
        # over by the room it reaches into and read as a truncated word.
        edge_captions.append(
            f'<rect class="edge-label-plate" x="{plate[0]:.1f}" y="{plate[1]:.1f}" '
            f'width="{plate[2]:.1f}" height="{plate[3]:.1f}" rx="{CORNER_RADIUS}" style="fill:{PAGE};" />'
            f'<text class="edge-label" x="{plate[0] + plate_width / 2:.1f}" '
            f'y="{plate[1] + label_height / 2 + 0.36 * small_font:.1f}" '
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
        room_name = room_labels.get(room, room)
        lines = wrap_room_name(room_name, max(12, round(room_width / 7.2)))
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
            f'<g data-room-id="{html_lib.escape(room, quote=True)}"><title>{html_lib.escape(room_name)}</title>'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{room_width}" height="{room_height}" rx="{CORNER_RADIUS}" '
            f'style="fill:{fill};stroke:{stroke};stroke-width:{stroke_width};" />'
            f"{''.join(texts)}</g>"
        )

    compass = (
        f'<g font-size="11" text-anchor="middle" style="font-family:{MONO};font-weight:700;fill:{MUTED};">'
        f'<line x1="{compass_x}" y1="{compass_y - 12}" x2="{compass_x}" y2="{compass_y + 12}" style="{line_style}" />'
        f'<line x1="{compass_x - 12}" y1="{compass_y}" x2="{compass_x + 12}" y2="{compass_y}" style="{line_style}" />'
        f'<text x="{compass_x}" y="{compass_y - 17}">N</text>'
        f'<text x="{compass_x + 20}" y="{compass_y + 4}">E</text>'
        f'<text x="{compass_x}" y="{compass_y + 26}">S</text>'
        f'<text x="{compass_x - 20}" y="{compass_y + 4}">W</text></g>'
    )
    legend_text_style = f"font-family:{SANS};fill:{MUTED};"
    # Start the second entry past the first one's text, so renaming either keeps a gap instead of printing one legend on
    # top of the other.
    second_entry_x = side_padding + 24 + max(70, round(6.4 * len(occupants_label))) + 24
    legend = (
        f'<rect x="{side_padding}" y="{legend_y - 13}" width="16" height="16" rx="{CORNER_RADIUS}" '
        f'style="fill:{MAP_LIVE_FILL};stroke:{PRIMARY};stroke-width:1.5;" />'
        f'<text x="{side_padding + 24}" y="{legend_y}" font-size="11.5" style="{legend_text_style}">'
        f"{html_lib.escape(occupants_label)}</text>"
        f'<rect x="{second_entry_x}" y="{legend_y - 13}" width="16" height="16" rx="{CORNER_RADIUS}" '
        f'style="fill:{MAP_ROOM_FILL};stroke:{ACCENT};stroke-width:3;" />'
        f'<text x="{second_entry_x + 24}" y="{legend_y}" font-size="11.5" style="{legend_text_style}">'
        f"{html_lib.escape(new_rooms_label)}</text>"
    )

    caption_markup = (
        f'<p style="font-family:{SANS};font-size:0.85rem;color:{MUTED};margin:0.55rem 0.2rem 0.1rem;">'
        f"{html_lib.escape(caption)}</p>"
        if caption
        else ""
    )
    # A big map scrolls inside its panel rather than shrinking its labels to nothing, and a small one stays at its drawn
    # size.
    return HTML(
        f'<div class="notebook-map" style="{PANEL_STYLE}padding:0.5rem 0.6rem 0.35rem;overflow-x:auto;'
        f'{BLOCK_MARGIN}">'
        f'<svg viewBox="0 0 {svg_width:.0f} {svg_height:.0f}" role="img" aria-label="Discovered room map" '
        f'style="display:block;height:auto;margin-inline:auto;width:100%;max-width:{svg_width:.0f}px;'
        f'min-width:{min(svg_width, 980):.0f}px;">'
        f'<defs><marker id="{map_id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:{SECONDARY};fill-opacity:0.5;" />'
        f"</marker></defs>{''.join(edge_elements)}{''.join(room_elements)}{''.join(edge_captions)}"
        f"{compass}{legend}</svg>"
        f"{caption_markup}</div>"
    )
