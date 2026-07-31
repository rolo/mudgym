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
from mudgym import make_env

env = make_env(observation="parsed")
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step("howl")
print(observation["room_name"], reward)
env.close()
```

<!-- transcript: quickstart-single -->
```text
beaten track near cliff 0.0
```

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #aa5500; background-color: #000316">You howl.</span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: quickstart-single -->

### Multi-agent (MARL)

Two agents in the same world.

```python
from mudgym import make_parallel_env

env = make_parallel_env(agents=2)
observations, infos = env.reset()
actions = {agent: "yodel" for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
for agent in sorted(observations):
    print(agent, observations[agent]["room_name"], rewards[agent])
env.close()
```

<!-- transcript: quickstart-multiagent -->
```text
player_0 narrow road between lands 0.0
player_1 dally lane 0.0
```

**player_0**

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #aa5500; background-color: #000316">You yodel.</span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>

**player_1**

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #aa5500; background-color: #000316">A male voice in the distance yodels.</span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">

</span><span style="color: #aa5500; background-color: #000316">You yodel.</span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: quickstart-multiagent -->


## License

The Python code and tooling in this repository are MIT licensed.

The MUD2 game, name, source code, and associated story remain the intellectual property of Richard Bartle and Roy Trubshaw and may not be used for commercial purposes.

Permission to use the MUD2 game for research purposes has been provided by Richard Bartle, with kind thanks.
