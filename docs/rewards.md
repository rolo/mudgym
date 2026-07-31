# Rewards

The reward is the game's points delta for that step, parsed from points change events in the output, which take the form `(+100 = 300)`.

- `reward` is the delta, or `0.0` if nothing scored.
- `info["points"]` is the absolute total, when the step revealed it.
- The episode terminates at 204,800 upon reaching wizard rank.

Rewards are sparse and long-horizon, so you may well want to customise this.

## Reward shaping example

Layer your own with a standard [Gymnasium wrapper](https://gymnasium.farama.org/api/wrappers/):

```python
import gymnasium as gym

from mudgym import make_env


class ReachRoomWrapper(gym.Wrapper):
    """Reward arriving at a target room, then end the episode."""

    def __init__(self, env, target_room_id):
        super().__init__(env)
        self.target_room_id = target_room_id

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        if obs["room_id"] == self.target_room_id:
            reward += 1.0
            terminated = True
        else:
            reward -= 0.01

        return obs, reward, terminated, truncated, info


env = make_env(
    observation="cheats",
    wrappers=[lambda env: ReachRoomWrapper(env, "vroad4")],
)
env.close()
```
