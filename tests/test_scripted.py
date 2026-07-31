import pytest

from tests.scripted import ScriptedConnection, make_scripted_env


def test_make_scripted_env_rejects_connection_and_responses():
    with pytest.raises(ValueError, match="either connection or responses"):
        make_scripted_env(connection=ScriptedConnection(), responses={})
