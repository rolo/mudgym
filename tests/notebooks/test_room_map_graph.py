import itertools
from xml.etree import ElementTree

from mudgym.notebooks import show_room_map
from mudgym.notebooks.room_map import overlap_area


def svg_from(output):
    start = output.data.index("<svg")
    end = output.data.index("</svg>") + len("</svg>")
    return ElementTree.fromstring(output.data[start:end])


def test_graph_separates_room_ids_from_duplicate_labels():
    graph = {
        "north": {"name": "Forest", "exits": {"south": {"id": "south", "name": "Forest"}}},
        "south": {"name": "Forest", "exits": {}},
        "isolated": {"name": "Island", "exits": {}},
    }
    svg = svg_from(show_room_map(graph=graph, occupants={"north": ["Alpha"]}))
    rooms = svg.findall(".//g[@data-room-id]")
    assert {room.get("data-room-id") for room in rooms} == set(graph)
    labels = [room.find("title").text for room in rooms]
    assert labels.count("Forest") == 2
    assert "Island" in labels
    assert len(svg.findall(".//line[@marker-end]")) == 1


def test_recorded_edges_use_graph_labels_without_adding_other_routes():
    graph = {
        "start": {"name": "Road", "exits": {"north": {"id": "goal", "name": "Hut"}}},
        "goal": {"name": "Hut", "exits": {}},
        "other": {"name": "Other", "exits": {}},
    }
    svg = svg_from(show_room_map([("start", "north", "goal")], graph=graph))
    rooms = svg.findall(".//g[@data-room-id]")
    assert {room.get("data-room-id") for room in rooms} == {"start", "goal"}
    assert {room.find("title").text for room in rooms} == {"Road", "Hut"}


def test_string_routes_keep_their_existing_labels():
    svg = svg_from(show_room_map([("Road", "north", "Hut")]))
    rooms = svg.findall(".//g[@data-room-id]")
    assert {room.get("data-room-id") for room in rooms} == {"Road", "Hut"}
    assert {room.find("title").text for room in rooms} == {"Road", "Hut"}


def test_partial_graph_retains_labels_from_exit_destinations():
    graph = {"road": {"name": "Road", "exits": {"north": {"id": "hut", "name": "Hut"}}}}
    svg = svg_from(show_room_map(graph=graph))
    rooms = {room.get("data-room-id"): room for room in svg.findall(".//g[@data-room-id]")}
    assert rooms["hut"].find("title").text == "Hut"


def test_rooms_crowded_round_one_room_keep_clear_of_each_other():
    # Exits with no compass meaning fill the cells round their room, then cells one grid unit from each other.
    svg = svg_from(show_room_map([("hub", f"portal {index}", f"room {index}") for index in range(12)]))
    boxes = [
        [float(rect.get(name)) for name in ("x", "y", "width", "height")]
        for rect in svg.findall(".//g[@data-room-id]/rect")
    ]
    assert len(boxes) == 13
    assert not any(overlap_area(first, second) for first, second in itertools.combinations(boxes, 2))
