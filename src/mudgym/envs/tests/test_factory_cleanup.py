"""Failure-path factory validation."""

import pytest

from mudgym.connections.connection import MudConnection
from mudgym.envs.factory import make_env, make_parallel_env
from tests.scripted import ScriptedConnection, ScriptedProvider


def test_make_env_invalid_actions_rejected_before_constructing_env():
    conn = ScriptedConnection()
    with pytest.raises(ValueError, match="actions must be one of"):
        make_env(connection=conn, actions="sideways")
    assert conn.sent_lines == []
    assert not conn.closed


def test_make_envconnection_kwargs_with_instance_rejected():
    conn = ScriptedConnection()
    with pytest.raises(ValueError, match="connection_kwargs is not valid"):
        make_env(connection=conn, connection_kwargs={"timeout_ms": 1000})
    assert conn.sent_lines == []
    assert not conn.closed


def test_make_env_resolves_the_registry_default_at_call_time(monkeypatch):
    monkeypatch.setattr("mudgym.envs.factory.registry.default_connection", ScriptedConnection)

    env = make_env(observation="parsed")
    try:
        observation, _ = env.reset()
        assert observation["room_name"]
    finally:
        env.close()


def test_invalid_observation_is_rejected_before_adopting_provider():
    provider = ScriptedProvider()

    with pytest.raises(ValueError, match="observation must be one of"):
        make_parallel_env(1, provider=provider, observation="nope")

    assert provider.requested_count is None
    assert provider.closed is False


def test_provider_teardown_does_not_mask_batch_creation_error():
    closed = []

    class FailingCloseProvider:
        def create_connections(self, count: int) -> list[MudConnection]:
            raise ValueError("connection batch failed")

        def close(self):
            closed.append(True)
            raise RuntimeError("provider close failed")

    with pytest.raises(ValueError, match="connection batch failed"):
        make_parallel_env(1, provider=FailingCloseProvider())

    assert closed == [True]


def test_make_env_constructor_failure_closes_connection():
    connection = ScriptedConnection()

    with pytest.raises(ValueError, match="declare a command"):
        make_env(connection=connection, field_parsers=[])

    assert connection.closed is True


def test_child_constructor_failure_closes_entire_batch_and_provider():
    provider = ScriptedProvider()

    with pytest.raises(ValueError, match="declare a command"):
        make_parallel_env(3, provider=provider, field_parsers=[])

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_parallel_constructor_failure_closes_children_and_provider(monkeypatch):
    provider = ScriptedProvider()

    def failing_parallel_env(children, **kwargs):
        raise RuntimeError("parallel constructor failed")

    monkeypatch.setattr("mudgym.envs.factory.MudParallelEnv", failing_parallel_env)

    with pytest.raises(RuntimeError, match="parallel constructor failed"):
        make_parallel_env(3, provider=provider)

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_wrapper_constructor_failure_closes_children_and_provider(monkeypatch):
    provider = ScriptedProvider()

    def failing_wrapper(env):
        raise RuntimeError("wrapper constructor failed")

    monkeypatch.setattr("mudgym.envs.factory.ParallelDiscreteDirectionsWrapper", failing_wrapper)

    with pytest.raises(RuntimeError, match="wrapper constructor failed"):
        make_parallel_env(3, provider=provider, actions="directions")

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_provider_returning_wrong_batch_size_is_closed_with_its_connections():
    provider = ScriptedProvider(returned_count=2)

    with pytest.raises(RuntimeError, match="returned 2 connections, expected 3"):
        make_parallel_env(3, provider=provider)

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_factory_requests_one_connection_batch():
    provider = ScriptedProvider()

    env = make_parallel_env(3, provider=provider)
    try:
        assert provider.requested_count == 3
    finally:
        env.close()


def test_default_provider_configuration_policy(monkeypatch):
    provider = ScriptedProvider()
    calls = []

    def provider_factory():
        calls.append(True)
        return provider

    monkeypatch.setattr("mudgym.envs.factory.registry.default_parallel_provider_factory", provider_factory)

    env = make_parallel_env(2)
    try:
        assert calls == [True]
    finally:
        env.close()
