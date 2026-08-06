# Observations

Choose an observation mode with the `observation` argument to `make_env()`:

```python
from mudgym import make_env

env = make_env(observation="parsed")
observation, info = env.reset()
env.close()
```

## `text`

Just `{"text": str}`.

```python
--8<-- "docs/code/observations_text.py:observations-text"
```

--8<-- "docs/recordings/observations-text.md"

## `parsed`

Adds keyed data parsed from game output.

--8<-- "docs/recordings/observations-parsed.md"

```python exec="true" html="true"
from pathlib import Path

from mudgym.notebooks import show_ansi

print(show_ansi(Path("docs/recordings/observations-parsed.ansi").read_bytes()).data)
```

`available_exit_names` is the available subset of `DIRECTIONS`, in the same game-native order as the set bits in `available_exits`.
`over` and `swampward` are MudGym's public names for the game's internal `jump` and `swamp` directions.

`vitals` is `[stamina, max_stamina, effective_strength, strength, effective_dexterity, dexterity, magic, max_magic]`; `flags` is `[blind, deaf, crippled, dumb]`.

## `cheats`

Adds hidden state output from the `mgcheats` command, some of which wouldn't typically be known to a player. Most notably `room_id`.

--8<-- "docs/recordings/observations-cheats.md"

```python exec="true" html="true"
from pathlib import Path

from mudgym.notebooks import show_ansi

print(show_ansi(Path("docs/recordings/observations-cheats.ansi").read_bytes()).data)
```

## `bytes`

Adds `raw_bytes`, a fixed-size `uint8` NumPy array, zero-padded to 16,384 bytes by default. The unpadded bytes value is available as `info["raw_bytes"]`.

```python
raw = observation["raw_bytes"][: info["bytes_length"]].tobytes()
```

Shown as a bytes literal here for readability:

--8<-- "docs/recordings/observations-bytes.md"

```python exec="true" html="true"
from pathlib import Path

from mudgym.notebooks import show_ansi

print(show_ansi(Path("docs/recordings/observations-bytes.ansi").read_bytes()).data)
```

## Creating your own keys

You can customise the observation output further with `exclude_keys`, or specify the parser components you want to include with `field_parsers`:

```python
from mudgym import make_env
from mudgym.envs.fields import FEScoreField, FEXitsField

env = make_env(observation="parsed", exclude_keys=["weather_index"])
env.close()

env = make_env(field_parsers=[FEScoreField, FEXitsField])
env.close()
```

You can add your own field parsers in the same way by creating an [`ObservationField`](api.md#observation-fields) subclass. Take a look at the fields in `mudgym/envs/fields/` to use as a reference.

## Auto-commands

Observation fields can declare commands for output to consume. MudGym appends those commands after the player's action, and the final one also acts as an end-of-step marker, so we know when the step's response is complete.
