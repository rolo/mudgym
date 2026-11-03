import pytest

from mudgym.connections.registry import connections
from mudgym.envs.env import command_rejected


def send_and_read(connection, lines):
    """Send separate lines to exercise completion across a split command batch."""
    for line in lines:
        connection.send_line(line)
    return connection.read_response()


@pytest.mark.parametrize("connection_key", connections)
def test_player_command_with_too_many_parts_does_not_prevent_the_observation_line(connection_key):
    command_line = ",".join(["n"] * 25)

    connection = connections[connection_key]()
    try:
        connection.reset()
        raw_bytes, terminated, incomplete, debug_info = send_and_read(connection, [command_line, "fei"])

        assert b"Your command is too long for me, sorry!" in raw_bytes
        assert b"========" in raw_bytes
        assert command_rejected(raw_bytes, debug_info["sent_lines"]) is True
        assert terminated is False
        assert incomplete is False
    finally:
        connection.close()
