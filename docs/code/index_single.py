# ruff: noqa: PLC0415
"""Single-agent quickstart shown on the docs index page."""


def example(fragments) -> None:
    # --8<-- [start:index-single]
    from mudgym import make_env

    env = make_env(render_mode="ansi")
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step("l,howl")
    rendered = env.render()
    env.close()
    # --8<-- [end:index-single]
    fragments.write_ansi("index-single", rendered.encode("latin-1"))
    fragments.write_ansi("index-single-raw", info["raw_bytes"])
