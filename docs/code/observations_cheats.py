# ruff: noqa: PLC0415
"""The `cheats` observation table and display shown on the observations page."""


def example(fragments) -> None:
    from mudgym import make_env

    env = make_env(observation="cheats")
    observation, info = env.reset()
    env.close()

    fragments.write_fragment("observations-cheats", fragments.observation_table(observation))
    fragments.write_ansi("observations-cheats", info["render_bytes"])
