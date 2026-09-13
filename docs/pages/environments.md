# Environments

`make_env()` is the usual entry point. It builds a Gymnasium environment with one of the observation presets and either text or discrete direction actions.

```python
from mudgym import make_env

env = make_env(
    observation="parsed",   # bytes | text | parsed | cheats
    actions="text",         # text | directions
)
env.close()
```

See [Observations](observations.md) and [Actions](actions.md) for what each mode gives you. The full argument list is in the [API reference](api.md).

The presets are also registered with Gymnasium on import: `MUD2/Parsed-v0`, `MUD2/Text-v0`, `MUD2/Bytes-v0` and `MUD2/Cheats-v0`.

## The episode lifecycle

Games of MUD2 begin in The Elizabethan Tearoom. `reset()` begins a new episode by issuing a "north" command to step out of the tearoom into The Land. All bytes up to and including the tearoom exit message are trimmed and then everything which follows belongs to the episode's first observation.

## Text, bytes, and ANSI

Player-visible output is ASCII plus ANSI escape sequences.

Every reset and step exposes all three forms below, regardless of observation mode.

| Key | What it is |
|---|---|
| `info["raw_bytes"]` | The unmodified transition bytestring, including command echoes and any opt-in observation-command output. |
| `info["render_bytes"]` | Player-visible output, with ANSI retained. |
| `observation["text"]` | Player-visible plain text, with ANSI stripped. |

## Rendering

Pass `render_mode="human"` to print the player-visible output after each reset and step. With `"ansi"`, `env.render()` returns it as an ANSI string.

## Independent worlds

`SyncVectorEnv` collects scalar environments sequentially:

```python
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from mudgym import make_env

envs = SyncVectorEnv(
    [lambda: make_env(observation="parsed") for _ in range(8)],
    autoreset_mode=AutoresetMode.DISABLED,
)
observations, infos = envs.reset(seed=123)
envs.close()
```

## Shared worlds

Use `make_parallel_env()` to control multiple players in one shared world through PettingZoo. Actions and observations are dictionaries keyed by agent ID. See [Multi-agent](multiagent.md) for the step and reset contracts.
