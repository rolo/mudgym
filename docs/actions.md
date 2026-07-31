# Actions

We use a `text` action space by default, but wrappers can change this to a discrete one. Use [`DiscreteActionSpaceWrapper(env, commands=[...])`](api.md#mudgym.envs.actions.discrete.DiscreteActionSpaceWrapper) to specify your own commands, or [`DiscreteDirectionsWrapper(env)`](api.md#mudgym.envs.actions.discrete.DiscreteDirectionsWrapper) for the game's movement directions. Passing `actions="directions"` to [`make_env()`](api.md#mudgym.envs.factory.make_env) applies `DiscreteDirectionsWrapper` for you.

```python
from mudgym import make_env

env = make_env(actions="text")        # step("get axe")
env = make_env(actions="directions")  # step(3)
```

## `text`

`step()` takes a empty string of up to 64 characters.

<!-- transcript: actions-step -->
```text
room_name  beaten track
reward     0.0
terminated False
truncated  False
```

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #00aa00">Beaten track</span><span style="color: #F5F1DE">.
</span><span style="color: #00aa00; background-color: #000316">You're on a rough east-west track with a dense forest to the north and pasture to the south. </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: actions-step -->

## `directions`

`Discrete(14)` mapped onto `move <direction>` commands in the game's canonical exit order.

<!-- transcript: actions-directions -->
| Index | Command | Available here |
|---|---|---|
| 0 | `move north` | yes |
| 1 | `move east` | yes |
| 2 | `move south` | yes |
| 3 | `move west` | yes |
| 4 | `move northeast` | yes |
| 5 | `move southeast` | yes |
| 6 | `move southwest` | yes |
| 7 | `move northwest` | yes |
| 8 | `move up` | yes |
| 9 | `move down` | yes |
| 10 | `move in` | yes |
| 11 | `move out` | yes |
| 12 | `move over` | no |
| 13 | `move swampward` | yes |
<!-- /transcript: actions-directions -->

You can use the [parsed](observations.md#parsed) `available_exits` output as an action mask to avoid directions known to be unavailable.

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
