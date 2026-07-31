"""Failure-path factory validation."""

import pytest

from mudgym.connections.connection import MudConnection
from mudgym.envs.factory import make_env, make_parallel_env, make_vector_env
from tests.scripted import ScriptedConnection


def test_make_env_invalid_actions_rejected_before_constructing_env():
    """An invalid actions value is rejected before the env (and its connection) is acquired."""
    conn = MudConnection()
    with pytest.raises(ValueError, match="actions must be one of"):
        make_env(connection=conn, actions="sideways")
    # Validation happened before adoption, so make_env never owned (or opened) the connection.
    assert conn.sm is None


def test_make_env_connection_kwargs_with_instance_rejected():
    """connection_kwargs can't be combined with an already-constructed connection instance (there's nothing
    to bind them to), so it fails fast rather than silently ignoring them."""
    conn = MudConnection()
    with pytest.raises(ValueError, match="connection_kwargs is not valid"):
        make_env(connection=conn, connection_kwargs={"account_id": "x"})
    # Rejected before adoption: the passed-in connection was never opened.
    assert conn.sm is None


def test_make_parallel_env_teardown_does_not_mask_the_original_error():
    """If make_env fails and the provider's own close() then fails during teardown, the caller must still
    see the original error, not the cleanup one."""
    closed = []

    class FailingCloseProvider:
        def __init__(self, **kwargs):
            pass

        def create_connection(self, index: int) -> MudConnection:
            return MudConnection()

        def close(self):
            closed.append(True)
            raise RuntimeError("provider close failed")

    # an invalid observation preset makes make_env raise; teardown then hits the failing close
    with pytest.raises(ValueError, match="observation must be one of"):
        make_parallel_env(agents=1, provider_factory=FailingCloseProvider, make_env_kwargs={"observation": "nope"})

    assert closed == [True]


def test_make_vector_env_teardown_does_not_mask_the_original_error():
    closed = []

    class FailingCloseProvider:
        def __init__(self, **kwargs):
            pass

        def create_connection(self, index: int) -> MudConnection:
            return MudConnection()

        def close(self):
            closed.append(True)
            raise RuntimeError("provider close failed")

    with pytest.raises(ValueError, match="observation must be one of"):
        make_vector_env(1, provider_factory=FailingCloseProvider, make_env_kwargs={"observation": "nope"})

    assert closed == [True]


def test_make_env_closes_connection_when_wrapper_construction_fails():
    """Once MudEnv owns a connection, wrapper failures must close it."""
    conn = ScriptedConnection()

    def broken_wrapper(env):
        raise RuntimeError("wrapper failed")

    with pytest.raises(RuntimeError, match="wrapper failed"):
        make_env(connection=conn, wrappers=[broken_wrapper])

    assert conn.closed is True
