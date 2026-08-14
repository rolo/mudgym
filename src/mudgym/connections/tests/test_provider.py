import os
import pickle

import pytest

from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import DockerExecProvider


def test_constructor_is_configuration_only(monkeypatch):
    launched = []
    monkeypatch.setattr(
        DockerExecProvider,
        "prepare_shared_container",
        lambda self, slots: launched.append(slots) or "container",
    )

    provider = DockerExecProvider(worlds=2)

    assert launched == []
    assert provider.containers == []
    provider.close()


def test_omitted_worlds_default_to_batch_count(monkeypatch):
    launched_slots = []
    container_ids = iter(["container-1", "container-2"])

    def launch(self, slots):
        launched_slots.append(slots)
        return next(container_ids)

    monkeypatch.setattr(DockerExecProvider, "prepare_shared_container", launch)
    monkeypatch.setattr("mudgym.connections.provider.subprocess.run", lambda *args, **kwargs: None)

    provider = DockerExecProvider(worlds_per_container=2)
    try:
        connections = provider.create_connections(3)
        assert launched_slots == [2, 1]
        assert [(connection.container_id, connection.db_slot) for connection in connections] == [
            ("container-1", 0),
            ("container-1", 1),
            ("container-2", 0),
        ]
    finally:
        for connection in locals().get("connections", []):
            connection.close()
        provider.close()


@pytest.fixture
def provider():
    provider = DockerExecProvider(worlds=4, container_id="container")
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
def test_create_connections_maps_batch_indices_to_db_slots(provider, count, expected_slots):
    connections = provider.create_connections(count)
    try:
        assert [connection.db_slot for connection in connections] == expected_slots
    finally:
        for connection in connections:
            connection.close()


def test_multiplayer_connections_share_configured_worlds():
    provider = DockerExecProvider(worlds=2, container_id="container")
    connections = provider.create_connections(4)
    try:
        assert [connection.db_slot for connection in connections] == [0, 1, 0, 1]
        account_ids = [connection.account_id for connection in connections]
        assert len(account_ids) == len(set(account_ids))
    finally:
        for connection in connections:
            connection.close()
        provider.close()


def test_multiple_containers_use_local_slots_and_preserve_shared_world_mapping(monkeypatch):
    container_ids = iter(["container-1", "container-2"])
    monkeypatch.setattr(
        DockerExecProvider,
        "prepare_shared_container",
        lambda self, slots: next(container_ids),
    )
    monkeypatch.setattr("mudgym.connections.provider.subprocess.run", lambda *args, **kwargs: None)

    provider = DockerExecProvider(worlds=3, worlds_per_container=2)
    try:
        connections = provider.create_connections(6)
        assert [(connection.container_id, connection.db_slot) for connection in connections] == [
            ("container-1", 0),
            ("container-1", 1),
            ("container-2", 0),
            ("container-1", 0),
            ("container-1", 1),
            ("container-2", 0),
        ]
    finally:
        for connection in locals().get("connections", []):
            connection.close()
        provider.close()


def test_batch_failure_stops_containers_launched_before_a_later_failure(monkeypatch):
    launched = iter(["container-1", "container-2"])
    stopped = []

    def launch(self, slots):
        container_id = next(launched)
        if container_id == "container-2":
            raise RuntimeError("launch failed")
        return container_id

    monkeypatch.setattr(DockerExecProvider, "prepare_shared_container", launch)
    monkeypatch.setattr(
        "mudgym.connections.provider.subprocess.run",
        lambda command, **kwargs: stopped.append(command),
    )

    provider = DockerExecProvider(worlds=2, worlds_per_container=1)
    with pytest.raises(RuntimeError, match="launch failed"):
        provider.create_connections(2)

    assert stopped == [["docker", "stop", "container-1"]]


def test_batch_failure_closes_connections_created_before_a_later_failure(monkeypatch):
    connections = []
    stopped = []

    class FailingConnection(MudConnection):
        def __init__(self, **kwargs):
            if len(connections) == 2:
                raise RuntimeError("connection failed")
            kwargs.pop("container_id")
            super().__init__(**kwargs)
            self.closed = False
            connections.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(DockerExecProvider, "prepare_shared_container", lambda self, slots: "container-1")
    monkeypatch.setattr(
        "mudgym.connections.provider.subprocess.run",
        lambda command, **kwargs: stopped.append(command),
    )

    provider = DockerExecProvider(worlds=3, connection_class=FailingConnection)
    with pytest.raises(RuntimeError, match="connection failed"):
        provider.create_connections(3)

    assert all(connection.closed for connection in connections)
    assert stopped == [["docker", "stop", "container-1"]]


def test_provider_supplies_only_one_batch(provider):
    connections = provider.create_connections(1)
    try:
        with pytest.raises(RuntimeError, match="already created"):
            provider.create_connections(1)
    finally:
        for connection in connections:
            connection.close()


def test_unpickled_unallocated_provider_owns_new_containers(monkeypatch):
    stopped = []
    monkeypatch.setattr(DockerExecProvider, "prepare_shared_container", lambda self, slots: "container-1")
    monkeypatch.setattr(
        "mudgym.connections.provider.subprocess.run",
        lambda command, **kwargs: stopped.append(command),
    )

    original_provider = DockerExecProvider()
    original_provider._owner_pid = -1
    provider = pickle.loads(pickle.dumps(original_provider))
    assert provider._owner_pid == os.getpid()

    connections = provider.create_connections(1)
    for connection in connections:
        connection.close()
    provider.close()

    assert stopped == [["docker", "stop", "container-1"]]


def test_unpickled_allocated_provider_does_not_stop_parent_containers(monkeypatch):
    stopped = []
    monkeypatch.setattr(DockerExecProvider, "prepare_shared_container", lambda self, slots: "container-1")
    monkeypatch.setattr(
        "mudgym.connections.provider.subprocess.run",
        lambda command, **kwargs: stopped.append(command),
    )

    provider = DockerExecProvider()
    connections = provider.create_connections(1)
    copied_provider = pickle.loads(pickle.dumps(provider))

    copied_provider.close()
    assert stopped == []

    for connection in connections:
        connection.close()
    provider.close()
    assert stopped == [["docker", "stop", "container-1"]]


@pytest.mark.parametrize("worlds_per_container", [0, -1])
def test_worlds_per_container_must_be_positive(worlds_per_container):
    with pytest.raises(ValueError, match="worlds_per_container must be at least 1"):
        DockerExecProvider(worlds_per_container=worlds_per_container)


def test_closed_provider_rejects_batch_creation():
    provider = DockerExecProvider(container_id="container")
    provider.close()

    with pytest.raises(RuntimeError, match="Provider is closed"):
        provider.create_connections(1)
