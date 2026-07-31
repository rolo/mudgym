import pytest

from mudgym.connections.prompts import State
from mudgym.connections.registry import available_connections_dict


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_close_releases_the_account_for_the_next_login(connection_key):
    """
    Log in twice in sequence: the second login only works if close() logged the first session out rather than
    abandoning it.
    """
    connection_class = available_connections_dict[connection_key]

    for attempt in ("first", "second"):
        connection = connection_class()
        try:
            connection.reset()
            assert connection.sm.state == State.TEA_SIPPED, f"{attempt} login did not reach the tearoom"
        finally:
            connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_quitting_the_game_terminates_the_step_and_reset_recovers(connection_key):
    """
    A command that ends the game must come back with terminated=True, and reset() must ready up
    again from GAME_OVER.
    """
    connection_class = available_connections_dict[connection_key]

    connection = connection_class()
    try:
        connection.reset()
        _, terminated, _, _ = connection.send_command("quit")
        assert terminated is True
        assert connection.sm.state == State.GAME_OVER

        connection.reset()
        assert connection.sm.state == State.TEA_SIPPED
    finally:
        connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
@pytest.mark.parametrize("player_command", ["say Option:", "Option:", "Not updating persona."])
def test_player_authored_control_text_does_not_close_the_command_window(connection_key, player_command):
    """Known command echoes win over identical input-prompt and game-over text."""
    connection_class = available_connections_dict[connection_key]

    connection = connection_class()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = connection.send_command([player_command, "fei"])

        assert player_command.encode("ascii") in raw_bytes
        assert b"========" in raw_bytes
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()
