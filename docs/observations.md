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

<!-- transcript: observation-text -->
```text
Dally Lane.
You are standing on a dusty road with rising ground both to the north and south. Though dilapidated and disused, the route north of where you stand, with a building at the far end, looks as if it once formed a grand driveway. To the south, the road twists up the hill where, at the summit, an ancient walled monastery dominates the scene. Open fields lie to the west, and east is a flat area of lawn.
```

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #00aa00">Dally Lane</span><span style="color: #F5F1DE">.
</span><span style="color: #00aa00; background-color: #000316">You are standing on a dusty road with rising ground both to the north and south. Though dilapidated and disused, the route north of where you stand, with a building at the far end, looks as if it once formed a grand driveway. To the south, the road twists up the hill where, at the summit, an ancient walled monastery dominates the scene. Open fields lie to the west, and east is a flat area of lawn. </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: observation-text -->

## `parsed`

Adds keyed data parsed from game output.

<!-- transcript: observation-parsed -->
| Key | Value |
|---|---|
| `room_name` | `badly-paved road` |
| `room_name_index` | `25` |
| `here` | `road`, `wall`, `Stephen the protector`, `gap` |
| `features` | `road`, `wall`, `gap` |
| `mobiles` | _empty_ |
| `players` | `Stephen the protector` |
| `points` | `200` |
| `vitals` | `[54, 54, 61, 61, 65, 65,  0, 54]` |
| `flags` | `[0, 0, 0, 0]` |
| `reset_minutes` | `105` |
| `weather` | `fair` |
| `weather_index` | `1` |
| `available_exits` | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]` |
| `available_exit_names` | `north`, `east`, `south`, `west`, `northeast`, `southeast`, `southwest`, `northwest`, `up`, `down`, `in`, `out`, `swampward` |
| `portables` | _empty_ |
| `inventory` | _empty_ |

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #00aa00">Badly-paved road</span><span style="color: #F5F1DE">.
</span><span style="color: #00aa00; background-color: #000316">You are standing on a badly-paved road, which runs from the east to stop at a large wall constructed to the west. There is a narrow gap in this blockage, but it is so tight that if you wanted to go that way you'd have to drop everything to get through. North and south are the foothills of a pair of majestic mountains, and northeast is a deep valley. </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: observation-parsed -->

`available_exit_names` is the available subset of `DIRECTIONS`, in the same game-native order as the set bits in `available_exits`.
`over` and `swampward` are MudGym's public names for the game's internal `jump` and `swamp` directions.

`vitals` is `[stamina, max_stamina, effective_strength, strength, effective_dexterity, dexterity, magic, max_magic]`; `flags` is `[blind, deaf, crippled, dumb]`.

## `cheats`

Adds hidden state output from the `mgcheats` command, some of which wouldn't typically be known to a player. Most notably `room_id`.

<!-- transcript: observation-cheats -->
| Key | Value |
|---|---|
| `points` | `200` |
| `vitals` | `[52, 52, 65, 65, 63, 63,  0, 52]` |
| `flags` | `[0, 0, 0, 0]` |
| `reset_minutes` | `105` |
| `weather` | `fair` |
| `weather_index` | `1` |
| `available_exits` | `[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1]` |
| `available_exit_names` | `north`, `east`, `south`, `west`, `northeast`, `southeast`, `southwest`, `northwest`, `out`, `swampward` |
| `room_id` | `mroad2` |
| `room_id_index` | `596` |
| `room_name` | `narrow road` |
| `room_name_index` | `344` |
| `fighting` | `0` |
| `dark` | `0` |
| `glowing` | `0` |
| `asleep` | `0` |
| `gifted` | `0` |
| `here` | `road` |
| `ticks` | `0` |
| `portables` | _empty_ |
| `inventory` | _empty_ |

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #00aa00">Narrow road</span><span style="color: #F5F1DE">.
</span><span style="color: #00aa00; background-color: #000316">You are on a narrow east-west road with forest to the north and gorse scrub to the south. </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: observation-cheats -->

## `bytes`

Adds `raw_bytes`, a fixed-size `uint8` NumPy array, zero-padded to 16,384 bytes by default. The unpadded bytes value is available as `info["raw_bytes"]`.

```python
raw = observation["raw_bytes"][: info["bytes_length"]].tobytes()
```

Shown as a bytes literal here for readability:

<!-- transcript: observation-bytes -->
```python
b'move north,fes\r\n\x1b[32mBeaten track near cliff\x1b[37m.\r\n\x1b[0;32;40mYou are at the end of a rough track. There is a dangerous cliff to the west marked "Lovers\' Leap". \x1b[1;37;40m\x1b[0;32;40mIt is raining. \x1b[1;37;40m\x1b[36mA streetsign has fallen here. \x1b[37m\r\n\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m\x1b[0;37;40m\x1b[1;32;40m63\x1b[0;37;40'
...
```

<div class="game-frame" style="background:var(--notebook-code-background,#0b0d0c);color:#eee;font-family:'Notebook JetBrains Mono','JetBrains Mono','Fira Code','Consolas','Monaco',monospace;font-size:14px;line-height:1.5;padding:1.1em 1.2em;margin:0;white-space:pre-wrap;border:1px solid var(--notebook-primary,#18352f);border-left:4px solid var(--notebook-accent,#b58a2a);border-radius:12px;box-shadow:0 12px 28px rgba(24,53,47,0.12);max-height:60vh;overflow:auto;"><span style="color: #00aa00">Beaten track near cliff</span><span style="color: #F5F1DE">.
</span><span style="color: #00aa00; background-color: #000316">You are at the end of a rough track. There is a dangerous cliff to the west marked "Lovers' Leap". </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316"></span><span style="color: #00aa00; background-color: #000316">It is raining. </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316"></span><span style="font-weight: bold; color: #00aaaa; background-color: #000316">A streetsign has fallen here. </span><span style="font-weight: bold; color: #F5F1DE; background-color: #000316">
</span></div>
<!-- /transcript: observation-bytes -->

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
