# Actions

We use a `text` action space by default, but wrappers can change this to a discrete one. Use [`DiscreteActionSpaceWrapper(env, commands=[...])`](api.md#mudgym.envs.actions.discrete.DiscreteActionSpaceWrapper) to specify your own commands, or [`DiscreteDirectionsWrapper(env)`](api.md#mudgym.envs.actions.discrete.DiscreteDirectionsWrapper) for the game's movement directions. Passing `actions="directions"` to [`make_env()`](api.md#mudgym.envs.factory.make_env) applies `DiscreteDirectionsWrapper` for you.

```python
from mudgym import make_env

env = make_env(actions="text")        # step("get axe")
env = make_env(actions="directions")  # step(3)
```

## `text`

`step()` takes a non-empty string of up to 64 characters.

```python
--8<-- "docs/code/actions_text.py:actions-text"
```

--8<-- "docs/recordings/actions-text.md"

```python exec="true" html="true"
from pathlib import Path

from mudgym.notebooks import show_ansi

print(show_ansi(Path("docs/recordings/actions-text.ansi").read_bytes()).data)
```

## `directions`

`Discrete(14)` mapped onto `move <direction>` commands in the game's canonical exit order.

```python exec="true"
from mudgym.actions import DIRECTIONS

print("| Index | Command |")
print("|---|---|")
for index, direction in enumerate(DIRECTIONS):
    print(f"| {index} | `move {direction}` |")
```

`observation["available_exits"][index]` says whether the corresponding command is not known to be blocked in the current room, so you can use the [parsed](observations.md#parsed) output as an action mask to avoid directions known to be unavailable.

```python
import numpy as np

from mudgym import make_env
from mudgym.actions import DIRECTIONS

rng = np.random.default_rng(1)
env = make_env(observation="parsed", actions="directions")
observation, info = env.reset()

candidate_actions = np.flatnonzero(observation["available_exits"])
action = (
    int(rng.choice(candidate_actions))
    if len(candidate_actions)
    else int(env.action_space.sample())
)

observation, reward, terminated, truncated, info = env.step(action)
print(DIRECTIONS[action])
env.close()
```

!!! warning "Dark rooms report every exit"

    A dark room returns no exit line, and `FEXitsField` then reports all exits as available rather than none. A `True` means "not known to be blocked", not "known to be open".
