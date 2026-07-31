# MudGym

A Gymnasium reinforcement learning environment for MUD2. One of the first online multiplayer text adventure games.

## Setup 

A Docker engine is required to run the game. The image `ghcr.io/rolo/mudgym` is pulled automatically on first use.

Install MudGym from PyPI:

```bash
uv add mudgym
```

## Quickstart

```python
from mudgym import make_env

env = make_env(observation="parsed")
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step("howl")
print(observation["room_name"], reward)
env.close()
```

## Docs

See the [docs](https://rolo.github.io/mudgym/) for more examples.

## License

The Python code and tooling in this repository are made available under the MIT License.

Permission to use the MUD2 game for research purposes has been provided by Richard Bartle, with kind thanks. The MUD2 game, name, source code, and associated story remain the intellectual property of Richard Bartle and Roy Trubshaw and may not be used for commercial purposes.
