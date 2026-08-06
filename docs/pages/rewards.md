# Rewards

The reward is the game's points delta for that step, parsed from points change events in the output, which take the form `(+100 = 300)`.

- `reward` is the delta, or `0.0` if nothing scored.
- `info["points"]` is the absolute total, when the step revealed it.
- The episode terminates at 204,800 upon reaching wizard rank.