# ruff: noqa: PLC0415
"""Single-agent quickstart shown on the docs index page."""


def example(fragments) -> None:
    # --8<-- [start:index-single]
    from mudgym import make_env

    env = make_env()
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step("l,howl")
    env.close()
    # --8<-- [end:index-single]
    fragments.write_ansi("index-single", info["render_bytes"])
    fragments.write_ansi("index-single-raw", info["raw_bytes"])
