import pytest

from mudgym.connections.registry import available_connections_dict
from mudgym.envs.env import TEAROOM_EXIT_NARRATION_END
from mudgym.envs.fields.feinventory import FEInventoryField
from mudgym.session import MudSession


def send_and_read(connection, lines):
    """Send separate lines to exercise completion across a split command batch."""
    for line in lines:
        connection.send_line(line)
    return connection.read_response(FEInventoryField.end_of_turn_marker)


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_tearoom_exit_completes_without_an_observation_probe(connection_key):
    connection = available_connections_dict[connection_key]()
    session = MudSession(connection, observation_line="fei", end_of_turn_marker=FEInventoryField.end_of_turn_marker)
    try:
        session.reset()
        if connection.sm is not None:
            # Exercise the real process reader with the narration and prompt split across reads.
            connection.sm.child.maxread = 1
        session.send("move north")
        raw, terminated, incomplete, transport = session.read_pending_response(TEAROOM_EXIT_NARRATION_END)
        assert not terminated and not incomplete
        assert transport["marker_arrived"]
        assert transport["sent_lines"] == ["move north"]
        narration = TEAROOM_EXIT_NARRATION_END.search(raw)
        assert narration is not None
        assert b"You" in raw[narration.end() :]

        raw, terminated, incomplete, transport = session.receive()
        assert not terminated and not incomplete
        assert transport["sent_lines"] == ["fei"]
        assert b"========" in raw
        assert TEAROOM_EXIT_NARRATION_END.search(raw) is None
    finally:
        session.close()


@pytest.mark.parametrize("connection_key", [key for key in available_connections_dict if key != "wasm"])
def test_close_releases_the_account_for_the_next_login(connection_key):
    """
    Log in twice in sequence: the second login only works if close() logged the first session out rather than
    abandoning it.
    """

    for attempt in ("first", "second"):
        connection = available_connections_dict[connection_key]()
        try:
            connection.reset()
            raw_bytes, terminated, incomplete, _ = send_and_read(connection, ["look,sip tea,fei"])
            assert b"Elizabethan tearoom" in raw_bytes, f"{attempt} login did not reach the tearoom"
            assert b"You watch the world go by." in raw_bytes
            assert not terminated and not incomplete
        finally:
            connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_quitting_the_game_terminates_the_step_and_reset_recovers(connection_key):
    """
    A command that ends the game must come back with terminated=True, and reset() must ready up
    again for another episode.
    """

    connection = available_connections_dict[connection_key]()
    try:
        connection.reset()
        _, terminated, incomplete, _ = send_and_read(connection, ["quit"])
        assert terminated is True
        assert incomplete is False

        connection.reset()
        raw_bytes, terminated, incomplete, _ = send_and_read(connection, ["look,sip tea,fei"])
        assert b"Elizabethan tearoom" in raw_bytes
        assert b"You watch the world go by." in raw_bytes
        assert not terminated and not incomplete
    finally:
        connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
@pytest.mark.parametrize("player_command", ["say Option:", "Option:", "Not updating persona."])
def test_player_authored_control_text_does_not_close_the_command_window(connection_key, player_command):
    """Known command echoes win over identical input-prompt and game-over text."""

    connection = available_connections_dict[connection_key]()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = send_and_read(connection, [player_command, "fei"])

        # Keep the transport's completion details in the failure output.
        why = f"{debug_info=}, tail={raw_bytes[-160:]!r}"

        assert player_command.encode("ascii") in raw_bytes, why
        assert b"========" in raw_bytes, why
        assert terminated is False, why
        assert incomplete is False, why
    finally:
        connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_rejection_before_final_line_echo_is_reported_after_marker_arrives(connection_key):
    """A rejected first line stays visible after a split batch reaches its final marker."""
    connection = available_connections_dict[connection_key]()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = send_and_read(connection, ["xyzzyfrobnicate", "fei"])

        assert b"xyzzyfrobnicate" in raw_bytes
        assert b"========" in raw_bytes
        assert debug_info["rejected"] is True
        assert debug_info["marker_arrived"] is True
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_spoken_rejection_text_is_not_reported_as_a_rejected_command(connection_key):
    """A rejection phrase quoted in player speech is not a front-end response."""
    connection = available_connections_dict[connection_key]()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = send_and_read(
            connection, ['say I don\'t know the word "frobnicate".', "fei"]
        )

        assert b"says" in raw_bytes
        assert b"========" in raw_bytes
        assert debug_info["rejected"] is False
        assert debug_info["marker_arrived"] is True
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()


@pytest.mark.parametrize("connection_key", available_connections_dict)
def test_player_command_with_too_many_parts_does_not_prevent_the_observation_line(connection_key):
    command_line = ",".join(["n"] * 25)

    connection = available_connections_dict[connection_key]()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = send_and_read(connection, [command_line, "fei"])

        assert b"Your command is too long for me, sorry!" in raw_bytes
        assert b"========" in raw_bytes
        assert debug_info["rejected"] is True
        assert debug_info["marker_arrived"] is True
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()
