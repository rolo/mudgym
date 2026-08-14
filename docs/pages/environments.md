# Environments

`make_env()` is the usual entry point. It builds a Gymnasium environment with one of four observation presets and either text or discrete direction actions.

```python
from mudgym import make_env

env = make_env(
    observation="parsed",   # bytes | text | parsed | cheats
    actions="text",         # text | directions
)
env.close()
```

See [Observations](observations.md) and [Actions](actions.md) for what each mode gives you. The full argument list is in the [API reference](api.md).

The presets are also registered with Gymnasium on import: `MUD2/Parsed-v0`, `MUD2/Text-v0`, `MUD2/Bytes-v0`, `MUD2/Cheats-v0`.

## The episode lifecycle

Games of MUD2 begin in The Elizabethan Tearoom. MudGym spins up a Docker container running the game, navigates the menus, and chooses a random name for your persona.

`reset()` begins a new episode by issuing a "north" command to step out of the tearoom into The Land. All bytes up to and including the tearoom exit message are trimmed and then everything which follows belongs to the episode's first observation.

!!! note "Seeding"

    `reset(seed=...)` does not yet make the MUD2 world fully reproducible.

## Text, bytes, and ANSI

Player-visible output is ASCII plus ANSI escape sequences.

Every reset and step puts all three forms below in `info`, regardless of observation mode.

| Key | What it is |
|---|---|
| `info["raw_bytes"]` | The unmodified bytestring, including prompts and command echo. |
| `env.render()` with `render_mode="ansi"` | Player visible output, with ANSI kept. |
| `observation["text"]` | Player visible plain text, with ANSI stripped. |

## Rendering

Pass `render_mode="human"` to print the player-visible output after each reset and step. With `"ansi"`, `env.render()` returns it as an ANSI string.

## Vector environments

```python
from mudgym import make_vector_env
from mudgym.connections.provider import DockerExecProvider

envs = make_vector_env(envs=8, provider=DockerExecProvider(worlds=2))
obs, info = envs.reset()
envs.close()
```

See [Multi-agent (MARL)](multiagent.md) for multi-agent support.
