import pytest

from mudgym.connections.provider import DockerExecProvider


@pytest.fixture
def provider():
    provider = DockerExecProvider(worlds=4)
    try:
        yield provider
    finally:
        provider.close()


@pytest.mark.parametrize(
    ("count", "expected_slots"),
    [
        pytest.param(4, [0, 1, 2, 3], id="one_per_world"),
        pytest.param(8, [0, 1, 2, 3, 0, 1, 2, 3], id="wraps_modulo"),
    ],
)
def test_create_connection_maps_env_index_to_db_slot(provider, count, expected_slots):
    connections = []
    try:
        for i in range(count):
            connections.append(provider.create_connection(i))

        assert [conn.db_slot for conn in connections] == expected_slots
    finally:
        for conn in connections:
            conn.close()


def test_multiplayer_environments_sharing_worlds():
    provider = DockerExecProvider(worlds=2)
    connections = []
    try:
        # 4 environments sharing 2 worlds
        connections = [provider.create_connection(i) for i in range(4)]

        # verify db_slots sharing pattern
        assert connections[0].db_slot == 0
        assert connections[1].db_slot == 1
        assert connections[2].db_slot == 0  # shares with conn 0
        assert connections[3].db_slot == 1  # shares with conn 1

        # all connections should have unique account_ids
        account_ids = [conn.account_id for conn in connections]
        assert len(account_ids) == len(set(account_ids))

    finally:
        for conn in connections:
            conn.close()
        provider.close()
