import gymnasium
from gymnasium.envs.registration import registry

env_configs = {
    "MUD2/Parsed-v0": {
        "observation": "parsed",
    },
    "MUD2/Bytes-v0": {
        "observation": "bytes",
    },
    "MUD2/Text-v0": {
        "observation": "text",
    },
    "MUD2/Cheats-v0": {
        "observation": "cheats",
    },
}

default_env_id = "MUD2/Parsed-v0"


def register_envs():
    for env_id, env_kwargs in env_configs.items():
        if env_id not in registry:
            gymnasium.register(
                id=env_id,
                entry_point="mudgym.envs.factory:make_env",
                kwargs=env_kwargs,
                nondeterministic=True,
            )
