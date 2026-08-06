# Exploration

A masked random walk. Full notebook: [`examples/exploring.py`](https://github.com/rolo/mudgym/blob/main/examples/exploring.py).

`make_env(actions="directions")` gives a `Discrete(14)` action space, and `available_exits` is a `MultiBinary(14)` mask over the same directions. Keep the set bits and sample among those.

```python
import numpy as np

--8<-- "examples/exploring.py:masked-policy"
```

The mask changes every step, so it is worth seeing one directly.

```python
from mudgym.actions import DIRECTIONS
from mudgym.notebooks import show_table

--8<-- "examples/exploring.py:direction-mask"
```
