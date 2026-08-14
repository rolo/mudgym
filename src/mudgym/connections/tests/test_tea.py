import time
from collections import OrderedDict

import pytest

from mudgym.connections.prompts import State
from mudgym.connections.registry import available_connections_dict
from mudgym.envs.fields.feinventory import FEInventoryField


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_tea(connection_key, subtests, tea_results, steps=5):
    """
    The tea test checks that the connection can get us into the game and that we are
    actually in the tearoom and able to sip tea.

    This allows us to use the "Time to Tea" metric to evaluate the different ways of
    connecting to the game.
    """
    connection_class = available_connections_dict[connection_key]

    results: OrderedDict[str, dict[str, float]] = OrderedDict()
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

    connection = connection_class()
    try:
        connection.reset()

        with subtests.test(msg="Initial state after first reset"):
            assert connection.sm.state == State.TEA_SIPPED, (
                f"Expected TEA_SIPPED state after connect, got {connection.sm.state}"
            )

        log_time("tea")

        connection.reset()

        with subtests.test(msg="State after second reset"):
            assert connection.sm.state == State.TEA_SIPPED, (
                f"Expected TEA_SIPPED state after second reset, got {connection.sm.state}"
            )

        with subtests.test(msg="Taking game steps"):
            for _ in range(steps):
                lines = ["move north", "fei"]
                for line in lines:
                    connection.send_line(line)
                connection.read_response(lines, FEInventoryField.end_of_turn_marker)
        log_time("steps")

        connection.reset()
        log_time("reset")

        with subtests.test(msg="State after third reset"):
            assert connection.sm.state == State.TEA_SIPPED, (
                f"Expected TEA_SIPPED state after third reset, got {connection.sm.state}"
            )
    finally:
        connection.close()

    log_time("close")

    tea_results[connection_key] = results
