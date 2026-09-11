import time

import pytest

from mudgym.connections.registry import available_connections_dict
from mudgym.envs.fields.feinventory import FEInventoryField


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_tea(connection_key, tea_results):
    """
    The tea test checks that the connection can get us into the game and that we are
    actually in the tearoom and able to sip tea.

    This allows us to use the "Time to Tea" metric to evaluate the different ways of
    connecting to the game.
    """
    results: dict[str, dict[str, float]] = {}
    start_time = time.perf_counter()
    last_mark = start_time

    def log_time(name):
        nonlocal last_mark
        now = time.perf_counter()
        results[name] = {
            "step": now - last_mark,
            "total": now - start_time,
        }
        last_mark = now

    connection = available_connections_dict[connection_key]()
    try:
        connection.reset()
        log_time("tea")
        connection.send_line("look,sip tea,fei")
        raw_bytes, terminated, incomplete, _ = connection.read_response(FEInventoryField.end_of_turn_marker)
        assert b"Elizabethan tearoom" in raw_bytes
        assert b"You watch the world go by." in raw_bytes
        assert not terminated and not incomplete

        for _ in range(5):
            connection.send_line("move north,fei")
            _, terminated, incomplete, _ = connection.read_response(FEInventoryField.end_of_turn_marker)
            assert not terminated and not incomplete
        log_time("steps")

        connection.reset()
        log_time("reset")
    finally:
        connection.close()

    log_time("close")

    tea_results[connection_key] = results
