# MudGym

A reinforcement learning environment for MUD2.

## Setup

The game runs in Docker, so you'll need a Docker engine. The image `ghcr.io/rolo/mudgym` is pulled automatically on first use.

Install the `mudgym` package with your Python package manager of choice.

=== "uv"

    ```bash
    uv add mudgym
    ```

=== "pip"

    ```bash
    pip install mudgym
    ```

## Quickstart

### Single Agent

```python
--8<-- "docs/code/index_single.py:index-single"
```

```python exec="true" html="true"
from pathlib import Path

from mudgym.notebooks import show_ansi

print(show_ansi(Path("docs/recordings/index-single.ansi").read_bytes()).data)
```

### Multi-agent (MARL)

Two agents in the same world.

```python
--8<-- "docs/code/index_multiagent.py:index-multiagent"
```

--8<-- "docs/recordings/index-multiagent.md"

```python exec="true"
from pathlib import Path

from mudgym.notebooks import show_ansi

for path in sorted(Path("docs/recordings").glob("index-multiagent-*.ansi")):
    print(f"**{path.stem.removeprefix('index-multiagent-')}**\n")
    print(show_ansi(path.read_bytes()).data + "\n")
```

## License

The Python code and tooling in this repository are MIT licensed.

The MUD2 game, name, source code, and associated story remain the intellectual property of Richard Bartle and Roy Trubshaw and may not be used for commercial purposes.

Permission to use the MUD2 game for research purposes has been provided by Richard Bartle, with kind thanks.
