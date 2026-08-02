# MudGym

A reinforcement learning environment for MUD2.

## Setup

The game runs in Docker, so you'll need a Docker engine. The image `ghcr.io/rolo/mudgym` is pulled automatically on first use.

Install the `mudgym` package with your Python package manager of choice.

=== "uv"

    ```bash
    uv add mudgym
    ```

=== "pip"

    ```bash
    pip install mudgym
    ```

## Quickstart

### Single Agent

```python exec="true" source="material-block" result="text" session="single"
from mudgym import make_env

env = make_env(observation="parsed")
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step("howl")
print(observation["room_name"], reward)
env.close()
```

```python exec="true" html="true" session="single"
from mudgym.notebooks import show_ansi

print(show_ansi(info["render_bytes"]).data)
```

### Multi-agent (MARL)

Two agents in the same world.

```python exec="true" source="material-block" result="text" session="multiagent"
from mudgym import make_parallel_env

env = make_parallel_env(agents=2)
observations, infos = env.reset()
actions = {agent: "yodel" for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
for agent in sorted(observations):
    print(agent, observations[agent]["room_name"], rewards[agent])
env.close()
```

```python exec="true" session="multiagent"
from mudgym.notebooks import show_ansi

for agent in sorted(infos):
    print(f"**{agent}**\n")
    print(show_ansi(infos[agent]["render_bytes"]).data + "\n")
```

## License

The Python code and tooling in this repository are MIT licensed.

The MUD2 game, name, source code, and associated story remain the intellectual property of Richard Bartle and Roy Trubshaw and may not be used for commercial purposes.

Permission to use the MUD2 game for research purposes has been provided by Richard Bartle, with kind thanks.
