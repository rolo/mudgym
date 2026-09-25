import gymnasium as gym
import pytest
from gymnasium.envs.registration import registry

import mudgym  # noqa: F401 - importing the package registers its Gymnasium envs
from mudgym.envs.registration import env_configs, register_envs


@pytest.mark.parametrize(("env_id", "expected_kwargs"), env_configs.items(), ids=list(env_configs))
def test_import_registers_each_id_with_the_expected_factory_spec(env_id, expected_kwargs):
    assert env_id in registry
    spec = gym.spec(env_id)
    assert spec.entry_point == "mudgym.envs.factory:make_env"
    assert spec.kwargs == expected_kwargs


def test_register_envs_is_idempotent():
    """Calling register_envs() again must not raise (guarded against duplicate registration)."""
    register_envs()
    register_envs()
