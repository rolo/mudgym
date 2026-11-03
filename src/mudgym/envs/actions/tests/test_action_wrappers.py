import numpy as np
import pytest
from gymnasium.spaces import Discrete

from mudgym.db.directions import DIRECTIONS
from mudgym.envs.actions.discrete import (
    DiscreteActionSpaceWrapper,
    DiscreteDirectionsWrapper,
    ParallelDiscreteDirectionsWrapper,
)
from mudgym.envs.zoo import MudParallelEnv
from tests.scripted import NoOpProvider


def test_discrete_action_space_wrapper_action_and_mapping(scripted_env):
    actions = ["look", "jump", "howl"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)

    assert isinstance(wrapped.action_space, Discrete)
    assert wrapped.action_space.n == 3

    assert wrapped.action(0) == "look"
    assert wrapped.action(1) == "jump"
    assert wrapped.action(2) == "howl"


def test_discrete_directions_wrapper_actions(scripted_env):
    wrapped = DiscreteDirectionsWrapper(scripted_env)
    assert wrapped.commands == tuple(f"move {direction}" for direction in DIRECTIONS)
    assert wrapped.action_space.n == len(wrapped.commands)


def test_parallel_directions_wrapper_has_matching_api_and_independent_spaces(scripted_env_factory):
    children = {
        "player_0": scripted_env_factory(),
        "player_1": scripted_env_factory(),
    }
    wrapped = ParallelDiscreteDirectionsWrapper(MudParallelEnv(children, provider=NoOpProvider()))

    try:
        assert wrapped.commands == tuple(f"move {direction}" for direction in DIRECTIONS)
        assert wrapped.action_space("player_0").n == len(wrapped.commands)
        assert wrapped.action_space("player_0") is not wrapped.action_space("player_1")
        assert wrapped.unwrapped.envs["player_0"] is children["player_0"]
    finally:
        wrapped.close()


@pytest.mark.parametrize("action", [-1, 2])
def test_discrete_action_wrapper_invalid_action_index(scripted_env, action):
    actions = ["a", "b"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)
    with pytest.raises(ValueError, match="Invalid discrete action"):
        wrapped.action(action)


def test_discrete_action_wrapper_numpy_array_input(scripted_env):
    actions = ["look", "dance"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)
    action_index = np.array(1)
    assert wrapped.action(action_index) == "dance"


def test_discrete_action_wrapper_rejects_empty_commands(scripted_env):
    with pytest.raises(ValueError, match="at least one"):
        DiscreteActionSpaceWrapper(scripted_env, [])


def test_discrete_action_wrapper_rejects_duplicate_commands(scripted_env):
    with pytest.raises(ValueError, match="unique"):
        DiscreteActionSpaceWrapper(scripted_env, ["look", "look"])
