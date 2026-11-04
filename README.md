# MudGym

A Gymnasium reinforcement learning environment for MUD2. One of the first online multiplayer text adventure games.

Run in your browser:
- [Quickstart](https://rolo.github.io/mudgym/lite/notebooks/index.html?path=quickstart.ipynb).
- [Goal room DQN](https://rolo.github.io/mudgym/lite/notebooks/index.html?path=goal_room_dqn.ipynb).

## Local Setup 

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

See the [docs](https://rolo.github.io/mudgym/).

## Citation

```
@misc{Mawlabaux2026MudGym,
	author = {Mawlabaux, Rolo},
	year = {2026},
	month = {nov 4},
	title = {MudGym},
	url = {https://rolo.github.io/mudgym/},
	howpublished = {https://rolo.github.io/mudgym/},
}
```

Or use the GitHub sidebar link.

## License

The Python code and tooling in this repository are made available under the MIT License.

Permission to use the MUD2 game for research purposes has been provided by Richard Bartle, with kind thanks. The MUD2 game, name, source code, and associated story remain the intellectual property of Richard Bartle and Roy Trubshaw and may not be used for commercial purposes.
