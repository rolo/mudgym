# Shared world

Two agents, one MUD2 world. Full notebook: [`examples/marl.py`](https://github.com/rolo/mudgym/blob/main/examples/marl.py).

`make_parallel_env` takes a dictionary of actions and returns a dictionary of every result. `step_order="rotate"` varies which agent reaches the game first, so none is permanently advantaged by turn order.

```python
from mudgym import make_parallel_env

--8<-- "examples/marl.py:marl-run"
```

Transporting both onto one mobile and looking again is the plainest evidence the world is shared: each agent's `players` field then lists the other.
