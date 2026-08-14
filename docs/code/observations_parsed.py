# ruff: noqa: PLC0415
"""The `parsed` observation table and display shown on the observations page."""


def example(fragments) -> None:
    from mudgym import make_env

    env = make_env(observation="parsed", render_mode="ansi")
    observation, info = env.reset()
    rendered = env.render()
    env.close()

    fragments.write_fragment("observations-parsed", fragments.observation_table(observation))
    fragments.write_ansi("observations-parsed", rendered.encode("latin-1"))
