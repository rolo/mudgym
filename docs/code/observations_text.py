# ruff: noqa: PLC0415
"""The `text` observation example shown on the observations page."""


def example(fragments) -> None:
    with fragments.captured_stdout() as stdout:
        # --8<-- [start:observations-text]
        from mudgym import make_env

        env = make_env(observation="text")
        observation, info = env.reset()
        env.close()

        print(observation["text"])
        # --8<-- [end:observations-text]
    fragments.write_fenced("observations-text", stdout.getvalue())
