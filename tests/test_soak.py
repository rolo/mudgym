"""Long-running live-game soaks for the reset paths that have historically flaked.

Excluded from the default test run via the ``soak`` marker; run explicitly with::

    uv run pytest -m soak tests/test_soak.py

Iteration counts scale with ``MUDGYM_SOAK_ITERATIONS`` (default 25).
Scalar soaks use every enabled connection in the registry. Shared-world soaks use the
Docker Exec and WASM providers. Docker Run owns a separate container per connection.

Covered failure shapes, both previously seen in the wild:
- the post-death ``env.reset()`` timing out at TEA_SIPPED after the quit/relogin sequence,
  singly and with several players sharing a world
- a fresh container refusing its first login with "All registration slots used up" after an
  unplanned reset
"""

import os
import subprocess
import time
from functools import partial

import pexpect
import pytest

from mudgym.connections.provider import DockerExecProvider
from mudgym.connections.registry import available_connections_dict
from mudgym.connections.wasm import WasmtimeProvider

pytestmark = pytest.mark.soak

SOAK_ITERATIONS = int(os.getenv("MUDGYM_SOAK_ITERATIONS", "25"))
SHARED_PROVIDERS = {"docker_exec": DockerExecProvider, "wasm": WasmtimeProvider}

# one step each: a real death (the swearing kill), a clean quit (Cheerio), and the disconnect cheat
DEATH_ACTIONS = ["fuck", "quit", "mgquit"]

REGISTRATION_REFUSED_MARKERS = (b"All registration slots used up", b"unplanned reset")


@pytest.fixture(params=available_connections_dict)
def connection_factory(request, wasm_runtime):
    factory = available_connections_dict[request.param]
    return partial(factory, runtime=wasm_runtime) if request.param == "wasm" else factory


def drain_wire(state_machine, seconds: float = 5.0) -> bytes:
    """Read whatever is still arriving on the wire, to tell 'late' apart from 'never coming'."""
    drained: list[bytes] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            drained.append(state_machine.child.read_nonblocking(size=65536, timeout=0.5))
        except pexpect.TIMEOUT:
            continue
        except (pexpect.EOF, OSError):
            break
    return b"".join(drained)


def state_machine_diagnostics(state_machine) -> str:
    if state_machine is None:
        return "no process state machine available"
    before = state_machine.get_buffer()
    drained = drain_wire(state_machine)
    lines = [
        f"state={state_machine.state.name}",
        f"history={' -> '.join(state.name for state in state_machine.history)}",
        f"buffer={before!r}",
        f"drained={drained!r}",
    ]
    if any(marker in before + drained for marker in REGISTRATION_REFUSED_MARKERS):
        lines.append("registration refusal detected: the game hit the fe_init failure path on login")
    return "\n".join(lines)


def single_env_diagnostics(env) -> str:
    return state_machine_diagnostics(env.unwrapped.session.connection.sm)


def container_forensics(env) -> str:
    """Capture the game-side evidence (waiter and mmud logs, shm state, processes) before the
    container's --rm teardown destroys it. The registration-refusal flake has stayed undiagnosed
    because these logs never left the container."""
    connection = env.unwrapped.session.connection
    container_name = getattr(connection, "container_name", None) or getattr(connection, "container_id", None)
    if container_name is None:
        return "no container name on the connection"
    capture_script = (
        "echo '--- reset directories ---'; ls /home/mudgod/LOGS 2>&1; "
        'for f in /home/mudgod/LOGS/w.*.log; do echo "--- $f ---"; tail -20 "$f" 2>&1; done; '
        "echo '--- latest mmud log ---'; tail -40 /home/mudgod/LOGS/reset.*/mmud 2>&1; "
        "echo '--- shm and semaphores ---'; ipcs -a 2>&1; "
        "echo '--- processes ---'; ps -ef 2>&1"
    )
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "sh", "-c", capture_script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"forensics capture failed: {error}"


def parallel_env_diagnostics(env) -> str:
    parts = []
    for name, child_env in env.envs.items():
        parts.append(f"--- agent {name} ---")
        parts.append(single_env_diagnostics(child_env))
    return "\n".join(parts)


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
            try:
                obs, info = env.reset(seed=iteration + 1)
            except (RuntimeError, ValueError) as error:
                pytest.fail(f"post-death reset {iteration} failed: {error}\n{single_env_diagnostics(env)}")
            reset_durations.append(time.monotonic() - started)
            assert obs["text"], "post-death reset returned an empty observation"

    summarise("post-death resets", reset_durations)


@pytest.mark.parametrize("connection_key", [key for key in available_connections_dict if key in SHARED_PROVIDERS])
def test_soak_parallel_post_death_resets_survive_live(
    connection_key, wasm_runtime, live_parallel_env_factory, subtests
):
    options = {"runtime": wasm_runtime} if connection_key == "wasm" else {}
    provider = SHARED_PROVIDERS[connection_key](worlds=1, **options)
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
            try:
                observations, infos = env.reset(seed=round_index + 1)
            except (RuntimeError, ValueError) as error:
                pytest.fail(f"parallel reset round {round_index} failed: {error}\n{parallel_env_diagnostics(env)}")
            reset_durations.append(time.monotonic() - started)
            assert env.agents, "parallel reset came back with no agents"

    summarise("parallel post-death resets", reset_durations)


def test_soak_fresh_logins_survive_live(connection_factory, live_env_factory, subtests):
    login_durations: list[float] = []

    for iteration in range(SOAK_ITERATIONS):
        with subtests.test(iteration=iteration):
            env = live_env_factory(connection=connection_factory)
            started = time.monotonic()
            try:
                obs, info = env.reset(seed=iteration)
            except (RuntimeError, ValueError) as error:
                pytest.fail(
                    f"fresh login {iteration} failed: {error}\n"
                    f"{single_env_diagnostics(env)}\n{container_forensics(env)}"
                )
            login_durations.append(time.monotonic() - started)
            assert obs["text"], "fresh login returned an empty observation"
            env.close()

    summarise("fresh logins", login_durations)
