"""Provider lifecycle checks use real worlds, with faults at the host boundary."""

import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import numpy as np
import pytest

from mudgym import make_env, make_vector_env
from mudgym.connections.wasm import WasmtimeProvider, WasmtimeRuntime
from mudgym.connections.wasm.wasmtime_provider import WorldAdvancementFailed
from mudgym.connections.wasm.wasmtime_runtime import WasmtimeTerminalTick, WasmtimeWorld


class InstrumentedRuntime(WasmtimeRuntime):
    def __init__(self, world_class):
        super().__init__()
        self.world_class = world_class
        self.created_worlds = []

    def create_world(self, **options):
        world = self.world_class(self, **options)
        self.created_worlds.append(world)
        return world


@pytest.mark.parametrize("seed", [99, [99, None]])
@pytest.mark.parametrize("departed", [False, True])
def test_masked_reset_reseeds_only_the_selected_independent_world(wasm_runtime, seed, departed):
    provider = WasmtimeProvider(runtime=wasm_runtime)
    with make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}, observation="text") as scalar:
        expected, expected_info = scalar.reset(seed=99)
    with closing(make_vector_env(2, provider=provider, observation="text")) as vector:
        vector.reset(seed=10)
        vector.step(["mgquit" if departed else "look", "look"])
        previous_worlds = list(provider._ordered_worlds)
        actual, info = vector.reset(seed=seed, options={"reset_mask": np.array([True, False])})
        assert provider._world_seeds == (99, 11)
        assert provider._ordered_worlds[0] is not previous_worlds[0]
        assert provider._ordered_worlds[1] is previous_worlds[1]
        assert actual["text"][0] == expected["text"]
        assert info["raw_bytes"][0] == expected_info["raw_bytes"]
        assert provider.advance_worlds(0) == {0: 0, 1: 1}
        vector.reset()
        assert provider._world_seeds == (99, 11)


def test_provider_allows_one_valid_batch_after_rejecting_an_invalid_topology(wasm_runtime):
    with closing(WasmtimeProvider(runtime=wasm_runtime, worlds=2)) as provider:
        with pytest.raises(ValueError, match="worlds cannot exceed connections"):
            provider.create_connections(1)
        first, independent, peer = provider.create_connections(3)
        with pytest.raises(RuntimeError, match="already created connections"):
            provider.create_connections(1)
        provider.reset(seed=123)
        for connection in (first, independent, peer):
            connection.reset()
            connection.send_line("look")
            raw, terminated, incomplete, _ = connection.read_response(None)
            assert b"Elizabethan tearoom" in raw
            assert not terminated and not incomplete
        assert first.advance_world_ticks(1) == 1
        assert peer.advance_world_ticks(0) == 1
        assert independent.advance_world_ticks(0) == 0


def test_shared_partial_reset_rejects_reseeding_without_replacing_any_session(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1)
    with closing(make_vector_env(2, provider=provider, observation="text")) as vector:
        vector.reset(seed=10)
        sessions = [connection._session for connection in provider._connections]
        with pytest.raises(ValueError, match="shared world"):
            vector.reset(seed=99, options={"reset_mask": np.array([True, False])})
        assert [connection._session for connection in provider._connections] == sessions
        assert provider._world_seeds == (10,)
        vector.reset(seed=99)
        assert provider._world_seeds == (99,)
        assert not vector.step(["look", "look"])[2].any()


def test_provider_timeout_reaches_native_ticks_and_session_admission():
    calls = []

    class TracedWorld(WasmtimeWorld):
        def _call(self, name, *arguments):
            if name in {
                "mud2_shared_tick",
                "mud2_shared_add_session",
                "mud2_shared_remove_session",
                "mud2_shared_step_session",
            }:
                calls.append((name, arguments[-1]))
            return super()._call(name, *arguments)

    with closing(WasmtimeProvider(runtime=InstrumentedRuntime(TracedWorld), worlds=1, timeout_ms=123)) as provider:
        connection, _ = provider.create_connections(2)
        connection.reset()
        provider.advance_worlds(1)
        connection.advance_world_ticks(1)
        connection.reset()
        assert {name for name, _ in calls} == {
            "mud2_shared_tick",
            "mud2_shared_add_session",
            "mud2_shared_remove_session",
            "mud2_shared_step_session",
        }
        assert all(timeout == 123 for _, timeout in calls), calls


def test_failed_reset_cleans_every_new_world_and_keeps_the_build_failure():
    construction_error = OSError("world construction failed")
    shutdown_error = RuntimeError("world shutdown failed")

    class FailingWorld(WasmtimeWorld):
        def __init__(self, runtime, *, seed, **options):
            if seed == 32:
                raise construction_error
            super().__init__(runtime, seed=seed, **options)
            self.fail_shutdown = seed == 30

        def shutdown(self):
            super().shutdown()
            if self.fail_shutdown:
                raise shutdown_error

    runtime = InstrumentedRuntime(FailingWorld)
    provider = WasmtimeProvider(runtime=runtime)
    try:
        connections = provider.create_connections(3)
        provider.reset(seed=10)
        previous_worlds = list(provider._ordered_worlds)
        with pytest.raises(Exception) as raised:
            provider.reset(seed=30)
        replacements = [world for world in runtime.created_worlds if world not in previous_worlds]
        assert len(replacements) == 2
        assert all(world.closed for world in replacements)
        assert raised.value.exceptions == (construction_error, shutdown_error)
        assert raised.value.__cause__ is construction_error
        assert provider._ordered_worlds == previous_worlds
        for connection in connections:
            connection.reset()
            connection.send_line("look")
            assert b"Elizabethan" in connection.read_response(None)[0]
    finally:
        provider.close()
        for world in runtime.created_worlds:
            WasmtimeWorld.shutdown(world)


def test_failed_admission_preserves_the_native_error_when_shutdown_also_fails():
    shutdown_error = RuntimeError("world shutdown failed")

    class OverfilledWorld(WasmtimeWorld):
        def add_session(self, persona_name, *arguments, **options):
            super().add_session(persona_name, *arguments, **options)
            return super().add_session(persona_name, *arguments, **options)

        def shutdown(self):
            super().shutdown()
            raise shutdown_error

    runtime = InstrumentedRuntime(OverfilledWorld)
    with closing(WasmtimeProvider(runtime=runtime)) as provider:
        provider.create_connections(1)
        with pytest.raises(ExceptionGroup) as raised:
            provider.reset()
        admission_error, cleanup_error = raised.value.exceptions
        assert "add-player-failed" in str(admission_error)
        assert cleanup_error is shutdown_error
        assert raised.value.__cause__ is admission_error
        assert all(world.closed for world in runtime.created_worlds)


def interrupted_tick():
    started, release, completed, returned = (threading.Event() for _ in range(4))

    class PausedWorld(WasmtimeWorld):
        def tick(self, *arguments, **options):
            started.set()
            assert release.wait(5), "tick was never released"
            result = super().tick(*arguments, **options)
            completed.set()
            return result

    def interrupt():
        assert started.wait(5), "tick was never started"
        os.kill(os.getpid(), signal.SIGINT)
        returned.wait(0.2)
        release.set()

    with closing(WasmtimeProvider(runtime=InstrumentedRuntime(PausedWorld))) as provider:
        provider.create_connections(1)
        provider.reset()
        interrupter = threading.Thread(target=interrupt)
        interrupter.start()
        try:
            with pytest.raises(BaseException) as raised:
                provider.advance_worlds(1)
            settled_when_raised = completed.is_set()
        finally:
            returned.set()
            interrupter.join(timeout=5)
        assert settled_when_raised, "advance_worlds returned while its native tick was still pending"
        assert isinstance(raised.value, KeyboardInterrupt)
        assert provider._ordered_worlds[0].current_tick() == 1


def test_sigint_settles_the_started_tick_before_escaping():
    result = subprocess.run([sys.executable, "-I", __file__], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("operation", ["reset", "advance_worlds"])
def test_interrupted_submission_settles_started_work_and_rolls_back_reset(operation):
    started, interrupted, returned, release = (threading.Event() for _ in range(4))
    interruption = KeyboardInterrupt("second submission interrupted")

    class InterruptedExecutor(ThreadPoolExecutor):
        submitted = None

        def submit(self, function, *arguments, **options):
            if self.submitted is not None:
                assert started.wait(5), "first task never started"
                interrupted.set()
                raise interruption

            def task():
                started.set()
                assert release.wait(5), "first task was never released"
                return function(*arguments, **options)

            self.submitted = super().submit(task)
            return self.submitted

    def release_worker():
        assert interrupted.wait(5), "second submission was never interrupted"
        returned.wait(0.2)
        release.set()

    runtime = InstrumentedRuntime(WasmtimeWorld)
    provider = WasmtimeProvider(runtime=runtime)
    try:
        provider.create_connections(2)
        provider.reset(seed=10)
        previous_worlds = list(provider._ordered_worlds)
        provider._executor.shutdown()
        provider._executor = executor = InterruptedExecutor(max_workers=2)
        releaser = threading.Thread(target=release_worker)
        releaser.start()
        try:
            with pytest.raises(KeyboardInterrupt) as raised:
                if operation == "reset":
                    provider.reset(seed=20)
                else:
                    provider.advance_worlds(1)
            settled_when_raised = executor.submitted.done()
        finally:
            returned.set()
            releaser.join(timeout=5)
            executor.shutdown()

        assert raised.value is interruption
        assert provider._ordered_worlds == previous_worlds
        if operation == "reset":
            replacements = [world for world in runtime.created_worlds if world not in previous_worlds]
            assert len(replacements) == 1
            assert (settled_when_raised, replacements[0].closed) == (True, True)
            assert provider._world_seeds == (10, 11)
        else:
            assert settled_when_raised, "tick was still running when submission interruption escaped"
            assert [world.current_tick() for world in previous_worlds] == [1, 0]
    finally:
        provider.close()
        for world in runtime.created_worlds:
            world.shutdown()


def test_closing_a_connection_releases_its_session_in_the_shared_world(wasm_runtime):
    with closing(WasmtimeProvider(runtime=wasm_runtime, worlds=1)) as provider:
        player, peer = provider.create_connections(2)
        player.reset()
        peer.reset()
        session = player._session
        world = session.world

        player.close()
        replacement = world.add_session(session.persona_name)

        assert replacement.player_id == session.player_id
        assert replacement.generation > session.generation
        assert b"Elizabethan tearoom" in replacement.send("look", 5000).output
        peer.send_line("look")
        raw, terminated, incomplete, _ = peer.read_response(None)
        assert b"Elizabethan tearoom" in raw
        assert not terminated and not incomplete


@pytest.mark.parametrize("advance_directly", [False, True])
def test_a_terminal_world_keeps_its_final_output_while_its_sibling_advances(wasm_runtime, advance_directly):
    provider = WasmtimeProvider(runtime=wasm_runtime)
    with closing(make_vector_env(2, provider=provider, observation="text")) as vector:
        vector.reset(seed=[51, 123])
        provider._connections[0].advance_world_ticks(3208)
        if advance_directly:
            with pytest.raises(WorldAdvancementFailed) as raised:
                provider.advance_worlds(1)
            assert raised.value.successes == {1: 1}
            assert set(raised.value.failures) == {0}
            terminal = raised.value.failures[0]
            assert isinstance(terminal, WasmtimeTerminalTick)
            assert terminal.tick_failure["requested_ticks"] == 1
            assert terminal.tick_failure["completed_ticks"] == 0
        _, rewards, terminated, truncated, info = vector.step(["look", "look"])
        assert terminated.tolist() == [True, False]
        assert not truncated.any()
        assert rewards.tolist() == [300, 0]
        assert info["raw_bytes"][0].count(b"Auto-reset initiated") == 1
        assert provider._ordered_worlds[1].current_tick() == 1 + advance_directly
        assert info["transport"]["sent_lines"].tolist() == [["look"], ["look"]]


@pytest.mark.parametrize("advance_connection", [False, True])
def test_step_clock_does_not_suppress_runtime_errors(wasm_runtime, advance_connection):
    with closing(WasmtimeProvider(runtime=wasm_runtime)) as provider:
        connection, _ = provider.create_connections(2)
        provider.reset(seed=123)
        provider._ordered_worlds[0].shutdown()

        with pytest.raises(RuntimeError, match="tick-failed"):
            if advance_connection:
                connection.tick_for_step()
            else:
                provider.tick_for_step()
        assert provider._ordered_worlds[1].current_tick() == (0 if advance_connection else 1)


if __name__ == "__main__":
    interrupted_tick()
