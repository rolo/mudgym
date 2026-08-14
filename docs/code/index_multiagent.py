# ruff: noqa: PLC0415
"""Multi-agent quickstart shown on the docs index page."""


def example(fragments) -> None:
    with fragments.captured_stdout() as stdout:
        # --8<-- [start:index-multiagent]
        from mudgym import make_parallel_env

        env = make_parallel_env(agents=2, render_mode="ansi")
        observations, infos = env.reset()
        actions = dict.fromkeys(env.agents, "yodel")
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent in sorted(observations):
            print(agent, observations[agent]["room_name"], rewards[agent])
        rendered = {agent: env.envs[agent].render() for agent in env.agents}
        env.close()
        # --8<-- [end:index-multiagent]
    fragments.write_fenced("index-multiagent", stdout.getvalue())
    for agent in sorted(rendered):
        fragments.write_ansi(f"index-multiagent-{agent}", rendered[agent].encode("latin-1"))
