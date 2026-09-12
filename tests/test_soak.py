"""Soak WASM session creation and post-death resets in isolated and shared worlds.

Run with ``uv run pytest -m soak tests/test_soak.py``. Set MUDGYM_SOAK_ITERATIONS to change the default 25 iterations.
"""

import os
import time
from functools import partial

import pytest

from mudgym.connections.wasm import WasmtimeProvider, create_connection

pytestmark = pytest.mark.soak

SOAK_ITERATIONS = int(os.getenv("MUDGYM_SOAK_ITERATIONS", "25"))

# one step each: a real death (the swearing kill), a clean quit (Cheerio), and the disconnect cheat
DEATH_ACTIONS = ["fuck", "quit", "mgquit"]


@pytest.fixture
def connection_factory(wasm_runtime):
    return partial(create_connection, runtime=wasm_runtime)


def summarise(name: str, durations: list[float]) -> None:
    if durations:
        print(
            f"\n{name}: {len(durations)} resets, mean {sum(durations) / len(durations):.2f}s, max {max(durations):.2f}s"
        )


def test_soak_post_death_resets_survive_live(connection_factory, live_env_factory, subtests):
    env = live_env_factory(connection=connection_factory)
    env.reset(seed=0)
    reset_durations: list[float] = []

    for iteration in range(SOAK_ITERATIONS):
        action = DEATH_ACTIONS[iteration % len(DEATH_ACTIONS)]
        with subtests.test(iteration=iteration, action=action):
            # a speech step every few iterations keeps the split wire format under soak pressure
            if iteration % 5 == 0:
                obs, reward, terminated, truncated, info = env.step(f"say soak iteration {iteration}")
                assert truncated is False, "speech step lost the auto command batch"
                assert terminated is False

            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated is True, f"{action!r} did not terminate the episode"

            started = time.monotonic()
            obs, info = env.reset(seed=iteration + 1)
            reset_durations.append(time.monotonic() - started)
            assert obs["text"], "post-death reset returned an empty observation"

    summarise("post-death resets", reset_durations)


def test_soak_parallel_post_death_resets_survive_live(wasm_runtime, live_parallel_env_factory, subtests):
    provider = WasmtimeProvider(worlds=1, runtime=wasm_runtime)
    env = live_parallel_env_factory(agents=3, provider=provider)
    rounds = max(SOAK_ITERATIONS // 2, 5)
    observations, infos = env.reset(seed=0)
    reset_durations: list[float] = []

    for round_index in range(rounds):
        with subtests.test(round=round_index):
            commands = {agent: DEATH_ACTIONS[round_index % len(DEATH_ACTIONS)] for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(commands)
            assert all(terminations.values()), f"round {round_index}: not all agents terminated"

            started = time.monotonic()
            observations, infos = env.reset(seed=round_index + 1)
            reset_durations.append(time.monotonic() - started)
            assert env.agents, "parallel reset came back with no agents"

    summarise("parallel post-death resets", reset_durations)


def test_soak_fresh_logins_survive_live(connection_factory, live_env_factory, subtests):
    login_durations: list[float] = []

    for iteration in range(SOAK_ITERATIONS):
        with subtests.test(iteration=iteration):
            env = live_env_factory(connection=connection_factory)
            started = time.monotonic()
            obs, info = env.reset(seed=iteration)
            login_durations.append(time.monotonic() - started)
            assert obs["text"], "fresh login returned an empty observation"
            env.close()

    summarise("fresh logins", login_durations)
