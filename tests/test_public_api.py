from mudgym import make_env, make_parallel_env
from mudgym.actions import DIRECTIONS, direction_index
from mudgym.envs.factory import make_env as factory_make_env
from mudgym.envs.factory import make_parallel_env as factory_make_parallel_env


def test_environment_factories_are_available_from_the_public_package():
    assert make_env is factory_make_env
    assert make_parallel_env is factory_make_parallel_env


def test_direction_index_matches_the_public_direction_order():
    assert [direction_index(direction) for direction in DIRECTIONS] == list(range(len(DIRECTIONS)))
