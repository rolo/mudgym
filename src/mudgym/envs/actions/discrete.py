import gymnasium as gym
from gymnasium.core import ActionWrapper
from gymnasium.vector import VectorActionWrapper, VectorEnv
from gymnasium.vector.utils import batch_space
from pettingzoo import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

from mudgym.db.directions import DIRECTIONS


class DiscreteActions:
    """Map a fixed set of text commands to a discrete action space."""

    def __init__(self, commands):
        self.commands = tuple(commands)
        if not self.commands:
            raise ValueError("commands must contain at least one command.")
        if len(set(self.commands)) != len(self.commands):
            raise ValueError("commands must be unique.")

        self.space = self.make_space()
        self.command_indices = {command: index for index, command in enumerate(self.commands)}

    def make_space(self):
        return gym.spaces.Discrete(len(self.commands))

    def command(self, index):
        if not self.space.contains(index):
            raise ValueError(f"Invalid discrete action {index!r}; expected {self.space}.")
        return self.commands[index]

    def index(self, command):
        return self.command_indices[command]


class DiscreteActionSpaceWrapper(ActionWrapper):
    """
    Sets the action space to a discrete categorical multiple choice space.
    Maps discrete actions (ints) to string commands for the underlying env.

    We use `command` to refer to the text sent to the game and `action` as the
    RL/gymnasium side concept.
    """

    def __init__(self, env, commands):
        super().__init__(env)
        self.discrete_actions = DiscreteActions(commands)
        self.commands = self.discrete_actions.commands
        self.action_count = len(self.commands)
        self.action_space = self.discrete_actions.space

    def action(self, index):
        """
        Map discrete index to command string.
        """
        return self.discrete_actions.command(index)


class ParallelDiscreteActionSpaceWrapper(BaseParallelWrapper):
    """Map every agent's discrete action before forwarding one parallel step."""

    def __init__(self, env: ParallelEnv, commands):
        super().__init__(env)
        self.discrete_actions = DiscreteActions(commands)
        self.commands = self.discrete_actions.commands
        self.action_count = len(self.commands)
        # Each agent owns a separate space, and therefore a separate sampling RNG stream.
        self.action_spaces = {agent: self.discrete_actions.make_space() for agent in env.possible_agents}

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions):
        commands = {agent: self.discrete_actions.command(action) for agent, action in actions.items()}
        return self.env.step(commands)


class VectorDiscreteActionSpaceWrapper(VectorActionWrapper):
    """Map every discrete vector action to a text command."""

    def __init__(self, env: VectorEnv, commands):
        super().__init__(env)
        self.discrete_actions = DiscreteActions(commands)
        self.commands = self.discrete_actions.commands
        self.action_count = len(self.commands)
        self.single_action_space = self.discrete_actions.space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def actions(self, actions):
        return tuple(self.discrete_actions.command(action) for action in actions)


DIRECTION_COMMANDS = tuple(f"move {direction}" for direction in DIRECTIONS)


class DiscreteDirectionsWrapper(DiscreteActionSpaceWrapper):
    """
    Set the action space to include the movement directions.
    """

    def __init__(self, env):
        super().__init__(env, commands=DIRECTION_COMMANDS)


class ParallelDiscreteDirectionsWrapper(ParallelDiscreteActionSpaceWrapper):
    """Set every agent's action space to the movement directions."""

    def __init__(self, env):
        super().__init__(env, commands=DIRECTION_COMMANDS)


class VectorDiscreteDirectionsWrapper(VectorDiscreteActionSpaceWrapper):
    """Set every vector entry's action space to movement directions."""

    def __init__(self, env):
        super().__init__(env, commands=DIRECTION_COMMANDS)
