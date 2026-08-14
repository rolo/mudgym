# ruff: noqa: PLC0415
"""The `bytes` observation literal and display shown on the observations page."""


def example(fragments) -> None:
    from mudgym import make_env

    env = make_env(observation="bytes")
    observation, info = env.reset()
    env.close()

    returned_bytes = bytes(observation["raw_bytes"][: len(info["raw_bytes"])])
    fragments.write_fenced("observations-bytes", f"{returned_bytes[:320]!r}\n...", language="python")
    fragments.write_ansi("observations-bytes", info["render_bytes"])
