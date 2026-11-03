"""Exercise the installed WASM engine through the public env API."""

import importlib.util
import shutil
import threading
from datetime import UTC, datetime, timedelta, timezone
from importlib import metadata
from importlib.resources import files

import pytest

from mudgym import make_env, make_parallel_env
from mudgym.connections.persona import Persona
from mudgym.connections.recording import RecordingProvider
from mudgym.connections.wasm import WasmProvider, WasmtimeRuntime


@pytest.mark.parametrize("recording", [False, True])
def test_parallel_env_rejects_multiple_wasm_worlds_and_closes_provider(wasm_runtime, tmp_path, recording):
    provider = WasmProvider(runtime=wasm_runtime, worlds=2)
    wrapped = provider
    if recording:
        wrapped = RecordingProvider(provider, lambda index: tmp_path / f"player{index}.jsonl")

    with pytest.raises(ValueError, match="connections must share one world"):
        make_parallel_env(2, provider=wrapped)

    assert provider._closed
    assert provider._connections == []
    assert provider._executor is None


@pytest.mark.parametrize(
    ("command", "content"),
    [
        ("help combat", (b"KILL",)),
        ("faq 28", (b"Q:", b"A:", b"de-mystifier")),
    ],
)
def test_installed_engine_serves_runtime_help_and_faq_content(wasm_runtime, command, content):
    env = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    try:
        env.reset(seed=123)
        _, _, terminated, truncated, info = env.step(command)
        assert not terminated and not truncated
        for expected in content:
            assert expected in info["raw_bytes"]
    finally:
        env.close()


def test_pending_peer_output_precedes_the_observation_command_echo(wasm_runtime):
    provider = WasmProvider(
        runtime=wasm_runtime,
        worlds=1,
        personas=(("Aaron", "male"), ("Abbie", "male")),
    )
    sender, observer = provider.create_connections(2)
    try:
        sender.reset()
        observer.reset()
        observer.send_line("look")
        sender.send_line('tell "queued before probe" to abbie')
        sender.read_response()
        observer.send_line("fes")
        raw_bytes, terminated, incomplete, info = observer.read_response()
        assert info["sent_lines"] == ["look", "fes"]
        assert not terminated and not incomplete
        assert raw_bytes.index(b"look\r\n") < raw_bytes.index(b"queued before probe")
        assert raw_bytes.index(b"queued before probe") < raw_bytes.index(b"fes\r\n")
        assert raw_bytes.count(b"queued before probe") == 1
    finally:
        provider.close()


def test_civil_anchor_controls_halloween_startup_and_seeded_replay(wasm_runtime):
    offset = timezone(timedelta(hours=-4))
    halloween = b"It's Hallowe'en, the time of magic!"
    for anchor, seasonal in (
        (datetime(2026, 10, 30, 23, 59, 59, tzinfo=offset), False),
        (datetime(2026, 10, 31, tzinfo=offset), True),
    ):
        runs = []
        for seed in (123, 456, 123):
            world = wasm_runtime.create_world(max_players=1, seed=seed, civil_time_anchor=anchor)
            try:
                session = world.add_session(Persona("Ada", "f"))
                initial = session.receive()
                assert (halloween in initial) is seasonal
                world.tick(7)
                runs.append((initial, session.send("time", 5000).output, session.send("fes", 5000).output))
            finally:
                world.shutdown()
        assert runs[0] == runs[2]
        assert runs[0] != runs[1]


def test_civil_anchor_offset_and_ticks_are_retained_across_resets(wasm_runtime):
    anchor = datetime(2026, 1, 15, 1, 1, 58, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    env = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime, "civil_time_anchor": anchor})
    try:
        runs = []
        for seed in (123, 456, 123):
            env.reset(seed=seed)
            first = env.step("time")[-1]["raw_bytes"]
            second = env.step("time")[-1]["raw_bytes"]
            assert b"It feels about one o'clock." in first
            assert b"It feels about five past one." in second
            runs.append((first, second))
        assert runs[0] == runs[2]
        assert runs[0] != runs[1]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        (datetime(2026, 1, 1), "must include a UTC offset"),  # noqa: DTZ001 - deliberately test a naive anchor
        (datetime(2026, 1, 1, microsecond=1, tzinfo=UTC), "whole-second precision"),
    ],
)
def test_civil_anchor_refuses_ambiguous_or_lossy_values(wasm_runtime, anchor, message):
    with pytest.raises(ValueError, match=message):
        wasm_runtime.create_world(max_players=1, seed=123, civil_time_anchor=anchor)


def test_installation_contains_only_the_binary_engine_and_no_native_binding():
    for module in ("mudlib", "cffi", "_cffi_backend"):
        assert importlib.util.find_spec(module) is None
    distribution = metadata.distribution("mudgym-wasm-engine")
    assert not distribution.requires


@pytest.mark.parametrize("mode", ["scalar", "parallel"])
def test_explicit_world_ticker_overrides_the_backend_clock(wasm_runtime, mode):
    provider = WasmProvider(runtime=wasm_runtime, worlds=1)

    def ticker():
        provider.advance_worlds(3)

    if mode == "scalar":
        env = make_env(connection=provider.create_connections(1)[0], world_ticker=ticker)
        action = "look"
    else:
        env = make_parallel_env(2, provider=provider, world_ticker=ticker)
        action = {"player_0": "look", "player_1": "look"}
    try:
        env.reset(seed=123)
        env.step(action)
        assert provider.advance_worlds(0) == {0: 3}
    finally:
        env.close()
        provider.close()


@pytest.mark.parametrize("mode", ["scalar", "parallel"])
def test_each_step_runs_the_provider_s_ticks_per_step(wasm_runtime, mode):
    if mode == "scalar":
        env = make_env(connection_kwargs={"runtime": wasm_runtime, "ticks_per_step": 4})
        provider = env.unwrapped.session.connection.provider
        action = "look"
    else:
        provider = WasmProvider(runtime=wasm_runtime, worlds=1, ticks_per_step=4)
        env = make_parallel_env(2, provider=provider)
        action = {"player_0": "look", "player_1": "look"}
    try:
        env.reset(seed=123)
        env.step(action)
        env.step(action)
        assert provider.advance_worlds(0) == {0: 8}
    finally:
        env.close()


@pytest.mark.parametrize("ticks_per_step", [0, True, 1.5])
def test_ticks_per_step_must_be_a_positive_integer(wasm_runtime, ticks_per_step):
    with pytest.raises(ValueError, match="ticks_per_step must be a positive integer"):
        WasmProvider(runtime=wasm_runtime, ticks_per_step=ticks_per_step)


@pytest.mark.parametrize(
    ("command", "rejected"),
    [
        ("xyzzyfrobnicate", True),
        ("dance,get the", True),
        ("dance,get goat", True),
        ("say I don't know the word frobnicate.", False),
        ("Not updating persona.", True),
    ],
)
def test_command_rejection_comes_from_game_output(wasm_runtime, command, rejected):
    env = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    try:
        env.reset(seed=123)
        _, _, terminated, truncated, info = env.step(command)
        assert not terminated and not truncated
        assert info["action_rejected"] is rejected
    finally:
        env.close()


def test_closing_the_wasm_envs_releases_their_workers(wasm_runtime):
    before = set(threading.enumerate())
    for env in (
        make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}),
        make_parallel_env(2, provider=WasmProvider(runtime=wasm_runtime, worlds=1)),
    ):
        try:
            env.reset(seed=211)
        finally:
            env.close()
            env.close()
    assert not [
        thread for thread in threading.enumerate() if thread not in before and thread.name.startswith("wasm-world")
    ]


def test_a_copied_module_needs_no_neighbouring_game_data(tmp_path):
    module = tmp_path / "mud2-wasi.wasm"
    shutil.copyfile(str(files("mudgym_wasm_engine").joinpath("wasi/mud2-wasi.wasm")), module)
    runtime = WasmtimeRuntime(wasm_path=module)
    provider = WasmProvider(runtime=runtime, worlds=1)
    env = make_parallel_env(1, provider=provider)
    try:
        env.reset(seed=123)
        _, _, terminates, truncates, infos = env.step({"player_0": "faq 1"})
        assert not any(terminates.values()) and not any(truncates.values())
        assert b"Elizabethan Tearoom" in infos["player_0"]["raw_bytes"]
        assert list(tmp_path.iterdir()) == [module]
    finally:
        env.close()


def test_natural_world_reset_returns_final_transition_and_can_restart(wasm_runtime):
    env = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    try:
        env.reset(seed=51)
        assert env.session.connection.advance_world_ticks(3208) == 3208
        _, reward, terminated, truncated, info = env.step("look")
        assert terminated
        assert not truncated
        # The seeded journey earns survival bonuses during the explicitly advanced interval.
        assert reward == 300
        assert b"Auto-reset initiated" in info["raw_bytes"]
        env.reset(seed=51)
        assert env.session.connection.advance_world_ticks(0) == 0
        _, _, terminated, truncated, _ = env.step("look")
        assert not terminated and not truncated
    finally:
        env.close()
