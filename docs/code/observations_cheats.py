# ruff: noqa: PLC0415
"""The `cheats` observation table and display shown on the observations page."""


def example(fragments) -> None:
    from mudgym import make_env

    env = make_env(observation="cheats", render_mode="ansi")
    observation, info = env.reset()
    rendered = env.render()
    env.close()

    fragments.write_fragment("observations-cheats", fragments.observation_table(observation))
    fragments.write_ansi("observations-cheats", rendered.encode("latin-1"))
