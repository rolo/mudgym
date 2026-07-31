import gymnasium as gym
from gymnasium.envs.registration import registry

import mudgym  # noqa: F401 - importing the package registers its Gymnasium environments
from mudgym.envs.registration import default_env_id, env_configs, register_envs


def test_import_mudgym_registers_all_ids():
    """``import mudgym`` registers every configured env ID into the gymnasium registry."""
    for env_id in env_configs:
        assert env_id in registry, f"{env_id} not registered after `import mudgym`"


def test_specs_resolve_to_the_factory():
    """Each registered ID resolves to make_env with the expected observation kwarg, lazily."""
    for env_id, expected_kwargs in env_configs.items():
        spec = gym.spec(env_id)
        assert spec.entry_point == "mudgym.envs.factory:make_env"
        assert spec.kwargs == expected_kwargs


def test_default_env_id_is_registered():
    assert gym.spec(default_env_id).kwargs == {"observation": "parsed"}


def test_register_envs_is_idempotent():
    """Calling register_envs() again must not raise (guarded against duplicate registration)."""
    register_envs()
    register_envs()
