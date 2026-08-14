# ruff: noqa: PLC0415
"""The `text` action example shown on the actions page."""


def example(fragments) -> None:
    with fragments.captured_stdout() as stdout:
        # --8<-- [start:actions-text]
        from mudgym import make_env

        env = make_env(observation="parsed", render_mode="ansi")
        env.reset()
        observation, reward, terminated, truncated, info = env.step("look")
        rendered = env.render()
        env.close()

        print(f"room_name  {observation['room_name']}")
        print(f"reward     {reward}")
        print(f"terminated {terminated}")
        print(f"truncated  {truncated}")
        # --8<-- [end:actions-text]
    fragments.write_fenced("actions-text", stdout.getvalue())
    fragments.write_ansi("actions-text", rendered.encode("latin-1"))
