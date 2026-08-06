# Goal room reward

A wrapper that replaces the game's reward with a fixed navigation task. Full notebook: [`examples/goal_room_task.py`](https://github.com/rolo/mudgym/blob/main/examples/goal_room_task.py).

```python
import gymnasium as gym

--8<-- "examples/goal_room_task.py:goal-room-wrapper"
```

Reaching the goal pays `success_reward` and ends the episode; every other step costs `step_penalty`. The two are exclusive, so the goal step is not also charged the penalty and a shorter route scores higher. Success is decided by room name, and MUD2 has many rooms sharing one, so this treats all of them as the goal.

## One attempt

```python
import marimo as mo
import numpy as np

from mudgym import make_env
from mudgym.notebooks import show_game_tabs

--8<-- "examples/goal_room_task.py:goal-room-attempt"
```

Each entry in `attempt` is a standard transition, which is the shape offline learning code expects.
