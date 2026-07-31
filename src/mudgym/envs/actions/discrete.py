import gymnasium as gym
from gymnasium.core import ActionWrapper

from mudgym.db.directions import DIRECTIONS


class DiscreteActionSpaceWrapper(ActionWrapper):
    """
    Sets the action space to a discrete categorical multiple choice space.
    Maps discrete actions (ints) to string commands for the underlying env.

    We use `command` to refer to the text sent to the game and `action` as the
    RL/gymnasium side concept.
    """

    def __init__(self, env, commands):
        super().__init__(env)
        self.commands = tuple(commands)
        if not self.commands:
            raise ValueError("commands must contain at least one command.")
        if len(set(self.commands)) != len(self.commands):
            raise ValueError("commands must be unique.")

        self.action_count = len(self.commands)
        self.action_space = gym.spaces.Discrete(self.action_count)

        # for reverse action mapping
        self._command_to_index = {command: index for index, command in enumerate(self.commands)}

    def action(self, index):
        """
        Map discrete index to command string.
        """
        if not self.action_space.contains(index):
            raise ValueError(f"Invalid discrete action {index!r}; expected {self.action_space}.")
        return self.commands[index]

    def reverse_action(self, command):
        """
        Map command string back to index.
        """
        return self._command_to_index[command]


class DiscreteDirectionsWrapper(DiscreteActionSpaceWrapper):
    """
    Set the action space to include the movement directions.
    """

    _DIRECTION_ACTIONS = [f"move {direction}" for direction in DIRECTIONS]

    def __init__(self, env):
        super().__init__(env, commands=self._DIRECTION_ACTIONS)
