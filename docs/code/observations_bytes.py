# ruff: noqa: PLC0415
"""The `bytes` observation literal and display shown on the observations page."""


def example(fragments) -> None:
    from mudgym import make_env

    env = make_env(observation="bytes", render_mode="ansi")
    observation, info = env.reset()
    rendered = env.render()
    env.close()

    returned_bytes = bytes(observation["raw_bytes"][: len(info["raw_bytes"])])
    fragments.write_fenced("observations-bytes", f"{returned_bytes[:320]!r}\n...", language="python")
    fragments.write_ansi("observations-bytes", rendered.encode("latin-1"))
