# Sorcerer war

A free-for-all between personas promoted with `mgsorcerise`. Combat kills are worth points, so the game's own scoring is the objective. Full notebook: [`examples/sorc_war.py`](https://github.com/rolo/mudgym/blob/main/examples/sorc_war.py).

```python
import numpy as np

from mudgym.db.directions import DIRECTIONS

--8<-- "examples/sorc_war.py:sorc-policy"
```

`kill player` names no one in particular, so the policy stays a single branch and the game picks the opponent. A bare `kill` is rejected. The parsed observation separates room contents by kind, and the env filters your own persona out of `players` -- matching the engine-side exclusion `mgcheats` makes -- so any entry at all is a rival.
