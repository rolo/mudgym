"""Failure-path factory validation."""

import pytest

from mudgym.connections.connection import MudConnection
from mudgym.envs.factory import make_env, make_parallel_env, make_vector_env
from tests.scripted import ScriptedConnection, ScriptedProvider


def test_make_env_invalid_actions_rejected_before_constructing_env():
    conn = MudConnection()
    with pytest.raises(ValueError, match="actions must be one of"):
        make_env(connection=conn, actions="sideways")
    assert conn.sm is None


def test_make_envconnection_kwargs_with_instance_rejected():
    conn = MudConnection()
    with pytest.raises(ValueError, match="connection_kwargs is not valid"):
        make_env(connection=conn, connection_kwargs={"account_id": "x"})
    assert conn.sm is None


def test_make_env_resolves_the_registry_default_at_call_time(monkeypatch):
    monkeypatch.setattr("mudgym.envs.factory.registry.default_connection", ScriptedConnection)

    env = make_env(observation="parsed")
    try:
        observation, _ = env.reset()
        assert observation["room_name"]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("factory", "factory_kwargs"),
    [
        (make_vector_env, {"envs": 1}),
        (make_parallel_env, {"agents": 1}),
    ],
)
def test_invalid_observation_is_rejected_before_adopting_provider(factory, factory_kwargs):
    provider = ScriptedProvider()

    with pytest.raises(ValueError, match="observation must be one of"):
        factory(provider=provider, observation="nope", **factory_kwargs)

    assert provider.requested_count is None
    assert provider.closed is False


@pytest.mark.parametrize(
    ("factory", "factory_kwargs"),
    [
        (make_vector_env, {"envs": 1}),
        (make_parallel_env, {"agents": 1}),
    ],
)
def test_provider_teardown_does_not_mask_batch_creation_error(factory, factory_kwargs):
    closed = []

    class FailingCloseProvider:
        def create_connections(self, count: int) -> list[MudConnection]:
            raise ValueError("connection batch failed")

        def close(self):
            closed.append(True)
            raise RuntimeError("provider close failed")

    with pytest.raises(ValueError, match="connection batch failed"):
        factory(provider=FailingCloseProvider(), **factory_kwargs)

    assert closed == [True]


def test_make_env_constructor_failure_closes_connection():
    connection = ScriptedConnection()

    with pytest.raises(ValueError, match="declare a command"):
        make_env(connection=connection, field_parsers=[])

    assert connection.closed is True


@pytest.mark.parametrize(
    ("factory", "factory_kwargs"),
    [
        (make_vector_env, {"envs": 3}),
        (make_parallel_env, {"agents": 3}),
    ],
)
def test_child_constructor_failure_closes_entire_batch_and_provider(factory, factory_kwargs):
    provider = ScriptedProvider()

    with pytest.raises(ValueError, match="declare a command"):
        factory(provider=provider, field_parsers=[], **factory_kwargs)

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_vector_constructor_failure_closes_children_and_provider(monkeypatch):
    provider = ScriptedProvider()

    def failing_vector_env(children, **kwargs):
        raise RuntimeError("vector constructor failed")

    monkeypatch.setattr("mudgym.envs.factory.MudVectorEnv", failing_vector_env)

    with pytest.raises(RuntimeError, match="vector constructor failed"):
        make_vector_env(3, provider=provider)

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_wrapper_constructor_failure_closes_children_and_provider(monkeypatch):
    provider = ScriptedProvider()

    def failing_wrapper(env):
        raise RuntimeError("wrapper constructor failed")

    monkeypatch.setattr("mudgym.envs.factory.VectorDiscreteDirectionsWrapper", failing_wrapper)

    with pytest.raises(RuntimeError, match="wrapper constructor failed"):
        make_vector_env(3, provider=provider, actions="directions")

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


def test_provider_returning_wrong_batch_size_is_closed_with_its_connections():
    provider = ScriptedProvider(returned_count=2)

    with pytest.raises(RuntimeError, match="returned 2 connections, expected 3"):
        make_vector_env(3, provider=provider)

    assert all(connection.closed for connection in provider.connections)
    assert provider.closed is True


@pytest.mark.parametrize(
    ("factory", "factory_kwargs", "expected_count"),
    [
        (make_vector_env, {"envs": 4}, 4),
        (make_parallel_env, {"agents": 3}, 3),
    ],
)
def test_factory_requests_one_connection_batch(factory, factory_kwargs, expected_count):
    provider = ScriptedProvider()

    env = factory(provider=provider, **factory_kwargs)
    try:
        assert provider.requested_count == expected_count
    finally:
        env.close()


@pytest.mark.parametrize(
    ("factory", "factory_kwargs", "registry_factory_name"),
    [
        (make_vector_env, {"envs": 2}, "default_provider_factory"),
        (make_parallel_env, {"agents": 2}, "default_parallel_provider_factory"),
    ],
)
def test_default_provider_configuration_policy(monkeypatch, factory, factory_kwargs, registry_factory_name):
    provider = ScriptedProvider()
    calls = []

    def provider_factory():
        calls.append(True)
        return provider

    monkeypatch.setattr(f"mudgym.envs.factory.registry.{registry_factory_name}", provider_factory)

    env = factory(**factory_kwargs)
    try:
        assert calls == [True]
    finally:
        env.close()
