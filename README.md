# MudGym

A Gymnasium reinforcement learning environment for MUD2. One of the first online multiplayer text adventure games.

## Browser Quickstart

No install needed, through the magic of WebAssembly.

See the game and env in action at the [MudGym Playground](https://mudgym.kingrolo.com/).

Or run and train against the game in JupyterLite notebooks:
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
	title = {MudGym},
	doi = {10.5281/zenodo.22730678},
	url = {https://rolo.github.io/mudgym/},
}
```

To cite a specific release, use its version DOI from [Zenodo](https://doi.org/10.5281/zenodo.22730678). You can also use GitHub's sidebar link.

## License

The Python code and tooling in this repository are made available under the MIT License.

Permission to use the MUD2 game for research purposes has been provided by Richard Bartle, with kind thanks. The MUD2 game, name, source code, and associated story remain the intellectual property of Richard Bartle and Roy Trubshaw and may not be used for commercial purposes.
