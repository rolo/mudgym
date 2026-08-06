# Quickstart

One environment, one step. Full notebook: [`examples/quickstart.py`](https://github.com/rolo/mudgym/blob/main/examples/quickstart.py).

```python
import numpy as np

from mudgym import make_env

--8<-- "examples/quickstart.py:quickstart-step"
```

`available_exits` is a mask over the `Discrete(14)` direction space, so taking a set bit is a legal move rather than a guess.
