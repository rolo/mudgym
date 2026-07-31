import numpy as np

from mudgym.db.index import room_name_to_index, weather_count
from mudgym.db.levels import WIZARD_POINTS
from mudgym.db.rooms import ROOM_NAMES


def assert_points_are_valid(points):
    assert isinstance(points, (int, np.integer))
    assert 0 <= points <= WIZARD_POINTS


def assert_common_observation_is_valid(obs, subtests):
    with subtests.test(msg="points"):
        assert_points_are_valid(obs["points"])

    with subtests.test(msg="vitals"):
        # check that all are values above 0, except for magic
        vitals = list(obs["vitals"])

        # remove magic from the list
        all_except_magic = vitals[:-2] + vitals[-1:]
        assert all(v > 0 for v in all_except_magic)

    with subtests.test(msg="flags"):
        assert list(obs["flags"]) == [0, 0, 0, 0]  # blind, deaf, crippled, dumb

    with subtests.test(msg="available_exits"):
        # ensure that at least one exit is available
        assert any(obs["available_exits"])

    with subtests.test(msg="weather"):
        # Weather should be a descriptive string
        assert isinstance(obs["weather"], str)
        assert obs["weather"] in ["fair", "cloudy", "overcast", "raining", "stormy", "sunny", "blizzard"]

    with subtests.test(msg="weather_index"):
        assert isinstance(obs["weather_index"], (int, np.integer))
        assert 0 <= obs["weather_index"] <= weather_count


def assert_parsed_observation_is_valid(obs, subtests):
    """
    Use some heuristics to check that the parsed observation is valid.
    The parsed preset includes room information from SuperQuickLookField.
    """
    assert_common_observation_is_valid(obs, subtests)

    with subtests.test(msg="no_room_id"):
        assert "room_id" not in obs

    with subtests.test(msg="no_room_id_index"):
        assert "room_id_index" not in obs

    with subtests.test(msg="room_name"):
        assert "room_name" in obs
        assert obs["room_name"] != "" and obs["room_name"] in ROOM_NAMES

    with subtests.test(msg="room_name_index"):
        assert "room_name_index" in obs
        assert obs["room_name_index"] == room_name_to_index(obs["room_name"])


def assert_cheats_observation_is_valid(obs, subtests):
    """
    Use some heuristics to check that the cheats observation is valid.
    The cheats preset includes MGCheatsField with full room info.
    """
    assert_common_observation_is_valid(obs, subtests)

    with subtests.test(msg="room_id"):
        assert "room_id" in obs
        assert obs["room_id"] != ""

    with subtests.test(msg="room_id_index"):
        assert "room_id_index" in obs

    with subtests.test(msg="room_name"):
        assert obs["room_name"] != "" and obs["room_name"] in ROOM_NAMES

    with subtests.test(msg="room_name_index"):
        assert obs["room_name_index"] == room_name_to_index(obs["room_name"])


def test_env_fields_parsed(live_env_factory, subtests):
    env = live_env_factory(observation="parsed")

    obs, _ = env.reset()
    assert_parsed_observation_is_valid(obs, subtests)

    obs, *_ = env.step("north")
    assert_parsed_observation_is_valid(obs, subtests)


def test_env_fields_cheats(live_env_factory, subtests):
    env = live_env_factory(observation="cheats")

    obs, _ = env.reset()
    assert_cheats_observation_is_valid(obs, subtests)

    obs, *_ = env.step("north")
    assert_cheats_observation_is_valid(obs, subtests)
