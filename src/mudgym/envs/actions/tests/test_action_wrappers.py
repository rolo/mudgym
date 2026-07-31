import numpy as np
import pytest
from gymnasium.spaces import Discrete

from mudgym.db.directions import DIRECTIONS
from mudgym.envs.actions.discrete import DiscreteActionSpaceWrapper, DiscreteDirectionsWrapper


def test_discrete_action_space_wrapper_action_and_reverse(scripted_env):
    actions = ["look", "jump", "howl"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)

    assert isinstance(wrapped.action_space, Discrete)
    assert wrapped.action_space.n == 3

    assert wrapped.action(0) == "look"
    assert wrapped.action(1) == "jump"
    assert wrapped.action(2) == "howl"

    assert wrapped.reverse_action("look") == 0
    assert wrapped.reverse_action("jump") == 1
    assert wrapped.reverse_action("howl") == 2

    with pytest.raises(KeyError):
        wrapped.reverse_action("non_existent_action")


def test_discrete_action_space_wrapper_step(scripted_env):
    actions = ["yodel", "jump", "howl"]
    discrete_env = DiscreteActionSpaceWrapper(scripted_env, actions)
    discrete_env.reset()
    obs, reward, terminated, truncated, info = discrete_env.step(1)
    assert info["last_command"] == "jump"


def test_discrete_directions_wrapper_actions(scripted_env):
    wrapped = DiscreteDirectionsWrapper(scripted_env)
    assert wrapped.commands == tuple(f"move {direction}" for direction in DIRECTIONS)
    assert wrapped.action_space.n == len(wrapped.commands)


def test_discrete_action_wrapper_invalid_action_index(scripted_env):
    actions = ["a", "b"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)
    with pytest.raises(ValueError, match="Invalid discrete action"):
        wrapped.action(2)


def test_discrete_action_wrapper_rejects_negative_action_index(scripted_env):
    actions = ["a", "b"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)
    with pytest.raises(ValueError, match="Invalid discrete action"):
        wrapped.action(-1)


def test_discrete_action_wrapper_numpy_array_input(scripted_env):
    actions = ["look", "dance"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)
    action_index = np.array(1)
    assert wrapped.action(action_index) == "dance"


def test_discrete_action_wrapper_single_action(scripted_env):
    wrapped = DiscreteActionSpaceWrapper(scripted_env, ["inventory"])
    assert wrapped.action_space.n == 1
    assert wrapped.action(0) == "inventory"


def test_discrete_action_wrapper_step_with_invalid_action(scripted_env):
    actions = ["a", "b"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)
    wrapped.reset()
    with pytest.raises(ValueError, match="Invalid discrete action"):
        wrapped.step(2)


def test_discrete_action_wrapper_rejects_empty_commands(scripted_env):
    with pytest.raises(ValueError, match="at least one"):
        DiscreteActionSpaceWrapper(scripted_env, [])


def test_discrete_action_wrapper_rejects_duplicate_commands(scripted_env):
    with pytest.raises(ValueError, match="unique"):
        DiscreteActionSpaceWrapper(scripted_env, ["look", "look"])


def test_discrete_action_wrapper_copies_commands(scripted_env):
    actions = ["look", "dance"]
    wrapped = DiscreteActionSpaceWrapper(scripted_env, actions)

    actions[0] = "changed"
    actions.append("howl")

    assert wrapped.commands == ("look", "dance")
    assert wrapped.action_space.n == 2
    assert wrapped.action(0) == "look"
    assert wrapped.reverse_action("dance") == 1
