# Multi-agent (MARL)

Multi-agent support is new and has only had superficial testing compared to the single-agent API.

Use [`make_parallel_env()`](api.md#factories) to create a PettingZoo `ParallelEnv`.

```python
from mudgym import make_parallel_env

env = make_parallel_env(agents=4)
observations, infos = env.reset()

observations, rewards, terminations, truncations, infos = env.step(
    {agent: "look" for agent in env.agents}
)
env.close()
```

Agents are named `player_0`, `player_1`, and so on. Finished agents are removed from `env.agents`, so only submit actions for agents still listed there.

!!! note "Parallel API, serial game"

    Actions arrive together, but MUD2 processes them one at a time. `step_order` controls the order: `"rotate"` (default) rotates the first mover each step, `"fixed"` keeps the same order, and `"shuffle"` randomises it.
