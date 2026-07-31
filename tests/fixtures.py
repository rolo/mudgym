from collections.abc import Callable, Iterator
from typing import Any

import pytest

from mudgym.envs.factory import make_env, make_parallel_env, make_vector_env
from mudgym.envs.fields.tests.payloads import BYTES_CASES
from mudgym.featurizers.responses import split_on_prompt
from tests.scripted import make_scripted_env


def tracked_factory(maker: Callable[..., Any]) -> Iterator[Callable[..., Any]]:
    """Yield a maker that records every env it builds, then closes them all at teardown."""
    created: list[Any] = []

    def make(*args: Any, **kwargs: Any):
        env = maker(*args, **kwargs)
        created.append(env)
        return env

    yield make
    for env in reversed(created):
        env.close()


@pytest.fixture
def scripted_env_factory():
    """Build deterministic envs over ScriptedConnection and close them at teardown."""
    yield from tracked_factory(make_scripted_env)


@pytest.fixture
def scripted_env(scripted_env_factory):
    return scripted_env_factory()


@pytest.fixture
def live_env_factory():
    """Build live Docker-backed envs and close them at teardown."""
    yield from tracked_factory(make_env)


@pytest.fixture
def live_env(live_env_factory):
    return live_env_factory()


@pytest.fixture
def live_vector_env_factory():
    yield from tracked_factory(make_vector_env)


@pytest.fixture
def live_parallel_env_factory():
    yield from tracked_factory(make_parallel_env)


@pytest.fixture(params=list(BYTES_CASES))
def bytes_case(request):
    """A captured real game payload: its expected parsed values plus the prompt-split chunks."""
    case = BYTES_CASES[request.param]
    return {**case, "chunks": split_on_prompt(case["raw"])}
