# Observations

Choose an observation mode with the `observation` argument to `make_env()`:

```python
from mudgym import make_env

env = make_env(observation="parsed")
observation, info = env.reset()
env.close()
```

```python exec="true" session="observations"
import numpy as np

from mudgym import make_env
from mudgym.notebooks import show_ansi


def format_value(value):
    if isinstance(value, np.ndarray):
        return f"`{np.array2string(value, separator=', ', threshold=24)}`"
    if isinstance(value, tuple):
        return ", ".join(f"`{item}`" for item in value) if value else "_empty_"
    return f"`{value}`"


def print_observation_table(observation):
    print("| Key | Value |")
    print("|---|---|")
    for key, value in observation.items():
        if key != "text":
            print(f"| `{key}` | {format_value(value)} |")
```

## `text`

Just `{"text": str}`.

```python exec="true" source="material-block" result="text" session="observations"
env = make_env(observation="text")
observation, info = env.reset()
env.close()

print(observation["text"])
```

## `parsed`

Adds keyed data parsed from game output.

```python exec="true" session="observations"
env = make_env(observation="parsed")
observation, info = env.reset()
env.close()

print_observation_table(observation)
```

```python exec="true" html="true" session="observations"
print(show_ansi(info["render_bytes"]).data)
```

`available_exit_names` is the available subset of `DIRECTIONS`, in the same game-native order as the set bits in `available_exits`.
`over` and `swampward` are MudGym's public names for the game's internal `jump` and `swamp` directions.

`vitals` is `[stamina, max_stamina, effective_strength, strength, effective_dexterity, dexterity, magic, max_magic]`; `flags` is `[blind, deaf, crippled, dumb]`.

## `cheats`

Adds hidden state output from the `mgcheats` command, some of which wouldn't typically be known to a player. Most notably `room_id`.

```python exec="true" session="observations"
env = make_env(observation="cheats")
observation, info = env.reset()
env.close()

print_observation_table(observation)
```

```python exec="true" html="true" session="observations"
print(show_ansi(info["render_bytes"]).data)
```

## `bytes`

Adds `raw_bytes`, a fixed-size `uint8` NumPy array, zero-padded to 16,384 bytes by default. The unpadded bytes value is available as `info["raw_bytes"]`.

```python
raw = observation["raw_bytes"][: info["bytes_length"]].tobytes()
```

Shown as a bytes literal here for readability:

```python exec="true" result="python" session="observations"
env = make_env(observation="bytes")
observation, info = env.reset()
env.close()

returned_bytes = bytes(observation["raw_bytes"][: info["bytes_length"]])
print(repr(returned_bytes[:320]))
print("...")
```

```python exec="true" html="true" session="observations"
print(show_ansi(info["render_bytes"]).data)
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
