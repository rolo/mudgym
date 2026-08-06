# Sorcerer team

A band of sorcerers promoted with `mgsorcerise` gathers on one mobile and hunts it as a pack. The gather mobile doubles as the quarry: it arrives as a leading `mgtransport <mobile> me` start command, so the episode opens with the party already assembled on its mark. Full notebook: [`examples/sorc_team.py`](https://github.com/rolo/mudgym/blob/main/examples/sorc_team.py).

```python
import numpy as np

from mudgym.db.directions import DIRECTIONS

--8<-- "examples/sorc_team.py:team-policy"
```

The quarry shows up in the parsed `here` field under its full display name -- the goat is listed as `billy goat` -- so the check is a substring match rather than an exact one. Fights resolve on the game's own real-time ticks while steps run at wire speed, so a hunt ends one of three ways: the quarry is slain and the game's points flow as reward, a sorcerer falls to a swing gone wide in the crowded room, or the step budget runs out with the quarry still standing.
