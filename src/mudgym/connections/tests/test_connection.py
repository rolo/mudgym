import pytest

from mudgym.connections.prompts import State
from mudgym.connections.registry import available_connections_dict, default_connection


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

        # debug_info names the prompt that closed the window, which is the only thing that tells
        # these two failures apart: TIMEOUT means the marker never arrived in time, while OPTION
        # (or any NO_LONGER_IN_GAME prompt) means the player's text beat its own echo and was read
        # as the game asking for input. Without it a failure here says only "True is not False".
        why = f"closed by {debug_info['matched_prompt']}, tail={raw_bytes[-160:]!r}"

        assert player_command.encode("ascii") in raw_bytes, why
        assert b"========" in raw_bytes, why
        assert terminated is False, why
        assert incomplete is False, why
    finally:
        connection.close()


def test_rejection_before_final_line_echo_is_reported_after_marker_arrives():
    """A rejected first line stays visible after a split batch reaches its final marker."""
    connection = default_connection()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = connection.send_command(["xyzzyfrobnicate", "fei"])

        assert b"xyzzyfrobnicate" in raw_bytes
        assert b"========" in raw_bytes
        assert debug_info["rejected"] is True
        assert debug_info["marker_arrived"] is True
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()


def test_spoken_rejection_text_is_not_reported_as_a_rejected_command():
    """A rejection phrase quoted in player speech is not a front-end response."""
    connection = default_connection()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = connection.send_command(
            ['say I don\'t know the word "frobnicate".', "fei"]
        )

        assert b"says" in raw_bytes
        assert b"========" in raw_bytes
        assert debug_info["rejected"] is False
        assert debug_info["marker_arrived"] is True
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()


def test_command_with_too_many_parts_is_reported_as_rejected():
    """The parser rejects the whole line before its final observation command can run."""
    command_line = ",".join(["n"] * 25 + ["sql", "fes", "fex", "fei"])

    connection = default_connection()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = connection.send_command(command_line)

        assert b"Your command is too long for me, sorry!" in raw_bytes
        assert b"========" not in raw_bytes
        assert debug_info["rejected"] is True
        assert debug_info["marker_arrived"] is False
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()
