# Multi-agent (MARL)

Multi-agent support is new and has only had superficial testing compared to the single-agent API.

`make_parallel_env()` provides the supported PettingZoo `ParallelEnv` facade.

```python
from mudgym import make_parallel_env

env = make_parallel_env(agents=4)
observations, infos = env.reset(seed=123)
observations, rewards, terminations, truncations, infos = env.step(
    {agent: "look" for agent in env.agents}
)
env.close()
```

`possible_agents` is the fixed list `player_0` through `player_3`. Each ID represents one player life during the current environment episode. A finished ID is removed from `agents` and is never revived within that episode. Its final observation and raw response are returned on the terminating or truncating step. A full `reset()` replaces the world and starts a new lifetime context for all IDs. Observation and action spaces retain their object identity across lives and resets.

This follows the [Parallel API's fixed possible_agents contract](https://pettingzoo.farama.org/api/parallel/) and its [official no-revival test](https://github.com/Farama-Foundation/PettingZoo/blob/main/pettingzoo/test/parallel_test.py). MudGym does not generate an unbounded sequence of PettingZoo agent IDs.
