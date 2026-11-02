"""Compass-grid room placement for the static room map."""

import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence

Cell = tuple[int, int]
Step = tuple[int, int] | None

# One compass step is two grid units, so a room can sit between two others. Up and down offset one unit diagonally.
DIRECTION_STEPS: dict[str, tuple[int, int]] = {
    "north": (0, -2),
    "south": (0, 2),
    "east": (2, 0),
    "west": (-2, 0),
    "northeast": (2, -2),
    "northwest": (-2, -2),
    "southeast": (2, 2),
    "southwest": (-2, 2),
    "up": (1, -1),
    "down": (-1, 1),
}

# Where an exit with no compass meaning lands, tried in this order.
NEAR_OFFSETS: tuple[Cell, ...] = (
    (2, 0),
    (0, 2),
    (-2, 0),
    (0, -2),
    (2, -2),
    (2, 2),
    (-2, 2),
    (-2, -2),
    (1, -1),
    (-1, 1),
)

DIRECTION_ALIASES = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "u": "up",
    "d": "down",
}

MOVE_PREFIX = re.compile(r"^move\s+(.+)$")


def compass_step(action: object) -> Step:
    """The grid step for a recorded action, accepting aliases and ``move <direction>``.

    An action with no compass meaning returns None and takes the nearest free cell instead.
    """
    text = "" if action is None else str(action).strip().lower()
    move = MOVE_PREFIX.match(text)
    direction = (move.group(1) if move else text).strip()
    return DIRECTION_STEPS.get(DIRECTION_ALIASES.get(direction, direction))


def candidate_offsets(step: Step) -> Iterator[Cell]:
    """Offsets in the order a new neighbour tries them.

    Along ``step`` first, stretching it before falling back to nearby cells, then an expanding ring that guarantees a
    spot. Every radius is swept so no free cell is skipped.
    """
    if step:
        for multiplier in (1, 2, 4, 8):
            yield step[0] * multiplier, step[1] * multiplier
    yield from NEAR_OFFSETS
    for radius in range(1, 65):
        for delta_x in range(-radius, radius + 1):
            yield from ((delta_x, -radius), (delta_x, radius))
        for delta_y in range(1 - radius, radius):
            yield from ((-radius, delta_y), (radius, delta_y))


def find_free_cell(occupied: set[Cell], origin: Cell, step: Step) -> Cell:
    """The first free cell around ``origin`` in ``candidate_offsets`` order."""
    for delta_x, delta_y in candidate_offsets(step):
        cell = (origin[0] + delta_x, origin[1] + delta_y)
        if cell not in occupied:
            return cell
    raise ValueError("no free map cell within search bounds")


def place_rooms(
    visited_room_ids: Sequence[str],
    exits_for_room: Callable[[str], Iterable[tuple[Step, str]]],
) -> tuple[dict[str, Cell], set[str]]:
    """Breadth-first placement from the first room, reseeding visited rooms the walk never reached.

    Returns the placed cells and the frontier, the rooms an exit names that were never visited. Frontier rooms are
    placed but not expanded.
    """
    visited = set(visited_room_ids)
    placed: dict[str, Cell] = {}
    occupied: set[Cell] = set()
    frontier: set[str] = set()
    if not visited_room_ids:
        return placed, frontier

    def claim(room_id: str, cell: Cell) -> None:
        placed[room_id] = cell
        occupied.add(cell)

    claim(visited_room_ids[0], (0, 0))
    queue = deque([visited_room_ids[0]])
    overflow_row = 6
    while True:
        while queue:
            room_id = queue.popleft()
            origin = placed[room_id]
            for step, destination in exits_for_room(room_id):
                if destination in placed:
                    continue
                claim(destination, find_free_cell(occupied, origin, step))
                if destination in visited:
                    queue.append(destination)
                else:
                    frontier.add(destination)
        unplaced = [room_id for room_id in visited_room_ids if room_id not in placed]
        if not unplaced:
            break
        entrance = next(
            (
                (room_id, step, destination)
                for room_id in unplaced
                for step, destination in exits_for_room(room_id)
                if destination in placed
            ),
            None,
        )
        if entrance is None:
            room_id = unplaced[0]
            claim(room_id, find_free_cell(occupied, (0, overflow_row), None))
            overflow_row += 6
        else:
            room_id, step, destination = entrance
            reverse = (-step[0], -step[1]) if step else None
            claim(room_id, find_free_cell(occupied, placed[destination], reverse))
        queue.append(room_id)
    return placed, frontier


def layout_world_from_edges(room_ids: Sequence[str], edges: Iterable[tuple[str, str, str]]) -> dict[str, Cell]:
    """Place rooms from recorded ``(source, action, destination)`` moves.

    Edges sort by source, action and destination before aliases resolve, so several actions to one room and one action
    to several rooms keep a stable order.
    """
    adjacency: dict[str, list[tuple[Step, str]]] = {}
    for source, action, destination in sorted((str(s), str(a), str(d)) for s, a, d in edges):
        adjacency.setdefault(source, []).append((compass_step(action), destination))
    return place_rooms([str(room_id) for room_id in room_ids], lambda room_id: adjacency.get(room_id, ()))[0]
