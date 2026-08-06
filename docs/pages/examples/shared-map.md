# Shared map

Several scouts chart one world between them. Full notebook: [`examples/shared_map.py`](https://github.com/rolo/mudgym/blob/main/examples/shared_map.py).

Agents sharing a room are the wasteful case, so the policy groups them and hands each a different, least-tried exit.

```python
--8<-- "examples/shared_map.py:scout-commands"
```

## Rewarding coverage

```python
--8<-- "examples/shared_map.py:coverage-reward"
```

Every agent receives the same number, which is what makes splitting up worth more than following each other. It is also the weakness: nothing distinguishes the agent that opened a new wing from the one that stood still.
