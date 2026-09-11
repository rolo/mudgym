"""Exercise the installed WASM engine through the public environment API."""

import importlib.util
import shutil
import threading
from datetime import UTC, datetime, timedelta, timezone
from importlib import metadata
from importlib.resources import files

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from mudgym import make_env, make_parallel_env, make_vector_env
from mudgym.connections.wasm import WasmtimeProvider, WasmtimeRuntime


@pytest.mark.parametrize(
    ("command", "content"),
    [
        ("help movement", (b"NORTHEAST", b"HELP MOVEMENT")),
        ("help combat", (b"KILL",)),
        ("faq 1", (b"Q:", b"A:", b"Elizabethan Tearoom")),
        ("faq 5", (b"Q:", b"A:", b"How do I move around?")),
        ("faq 12", (b"Q:", b"A:", b"Where will I find treasure?")),
        ("faq 28", (b"Q:", b"A:", b"de-mystifier")),
    ],
)
def test_installed_engine_serves_runtime_help_and_faq_content(wasm_runtime, command, content):
    environment = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    try:
        environment.reset(seed=123)
        _, _, terminated, truncated, info = environment.step(command)
        assert not terminated and not truncated
        for expected in content:
            assert expected in info["raw_bytes"]
    finally:
        environment.close()


def test_default_civil_clock_starts_at_midnight_on_every_scalar_reset(wasm_runtime):
    environment = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    try:
        for seed in (123, 456, 123):
            environment.reset(seed=seed)
            _, _, terminated, truncated, info = environment.step("time")
            assert not terminated and not truncated
            assert b"It feels about twelve o'clock." in info["raw_bytes"]
    finally:
        environment.close()


def test_pending_peer_output_precedes_the_observation_command_echo(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1)
    sender, observer = provider.create_connections(2)
    try:
        sender.reset()
        observer.reset()
        observer.send_line("look")
        sender.send_line('tell "queued before probe" to alba')
        sender.read_response(None)
        observer.send_line("fes")
        raw_bytes, terminated, incomplete, info = observer.read_response(None)
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
                session = world.add_session("Ada")
                initial = session.receive()
                assert (halloween in initial) is seasonal
                world.tick(7)
                runs.append((initial, session.send("time", 5000).output, session.send("fes", 5000).output))
            finally:
                world.shutdown()
        assert runs[0] == runs[2]
        assert runs[0] != runs[1]


@pytest.mark.parametrize("mode", ["scalar", "vector"])
def test_civil_anchor_offset_and_ticks_are_retained_across_resets(wasm_runtime, mode):
    anchor = datetime(2026, 1, 15, 1, 1, 58, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    if mode == "scalar":
        environment = make_env(
            connection="wasm", connection_kwargs={"runtime": wasm_runtime, "civil_time_anchor": anchor}
        )
        action = "time"
    else:
        environment = make_vector_env(1, provider=WasmtimeProvider(runtime=wasm_runtime, civil_time_anchor=anchor))
        action = ["time"]
    try:
        runs = []
        for seed in (123, 456, 123):
            environment.reset(seed=seed)
            _, _, _, _, info = environment.step(action)
            first = info["raw_bytes"] if mode == "scalar" else info["raw_bytes"][0]
            _, _, _, _, info = environment.step(action)
            second = info["raw_bytes"] if mode == "scalar" else info["raw_bytes"][0]
            assert b"It feels about one o'clock." in first
            assert b"It feels about five past one." in second
            runs.append((first, second))
        assert runs[0] == runs[2]
        assert runs[0] != runs[1]
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("anchor", "message"),
    [
        (datetime(2026, 1, 1), "must include a UTC offset"),
        (datetime(2026, 1, 1, microsecond=1, tzinfo=UTC), "whole-second precision"),
        (datetime(2026, 1, 1, tzinfo=timezone(timedelta(microseconds=1))), "whole-second precision"),
    ],
)
def test_civil_anchor_refuses_ambiguous_or_lossy_values(wasm_runtime, anchor, message):
    with pytest.raises(ValueError, match=message):
        wasm_runtime.create_world(max_players=1, seed=123, civil_time_anchor=anchor)


def test_installation_contains_only_the_binary_engine_and_no_native_binding():
    for module in ("mudlib", "cffi", "_cffi_backend"):
        assert importlib.util.find_spec(module) is None
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.distribution("mud2-embedded")
    distribution = metadata.distribution("mudgym-wasm-engine")
    assert not distribution.requires


@pytest.mark.parametrize("observation", ["text", "bytes", "parsed", "cheats"])
def test_existing_observation_presets_reset_and_step_with_real_engine_bytes(wasm_runtime, observation):
    environment = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}, observation=observation)
    try:
        initial, info = environment.reset(seed=123)
        assert environment.observation_space.contains(initial)
        assert info["persona"] == "Ada"
        assert initial["points"] == 200
        result, reward, terminated, truncated, info = environment.step("look")
        assert environment.observation_space.contains(result)
        assert result["text"]
        assert reward == 0 and not terminated and not truncated
        assert b"look\r\n" in info["raw_bytes"]
        if observation == "bytes":
            assert result["raw_bytes"].tobytes()[: len(info["raw_bytes"])] == info["raw_bytes"]
    finally:
        environment.close()


@pytest.mark.parametrize("mode", ["scalar", "vector", "parallel"])
def test_explicit_world_ticker_overrides_the_backend_clock(wasm_runtime, mode):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1)

    def ticker():
        provider.advance_worlds(3)

    if mode == "scalar":
        environment = make_env(connection=provider.create_connections(1)[0], world_ticker=ticker)
        action = "look"
    elif mode == "vector":
        environment = make_vector_env(2, provider=provider, world_ticker=ticker)
        action = ["look", "look"]
    else:
        environment = make_parallel_env(2, provider=provider, world_ticker=ticker)
        action = {"player_0": "look", "player_1": "look"}
    try:
        environment.reset(seed=123)
        environment.step(action)
        assert provider.advance_worlds(0) == {0: 3}
    finally:
        environment.close()
        provider.close()


def test_partial_vector_reset_preserves_the_other_world_and_observation(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime)
    environment = make_vector_env(2, provider=provider)
    try:
        environment.reset(seed=[123, 456])
        previous, *_ = environment.step(["look", "look"])
        observation, info = environment.reset(options={"reset_mask": np.array([True, False])})
        assert provider.advance_worlds(0) == {0: 0, 1: 1}
        assert observation["text"][1] == previous["text"][1]
        assert info["_raw_bytes"].tolist() == [True, False]
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("command", "rejected"),
    [
        ("xyzzyfrobnicate", True),
        ("say I don't know the word frobnicate.", False),
        ("say Not updating persona.", False),
        ("Not updating persona.", True),
    ],
)
def test_command_rejection_comes_from_game_output(wasm_runtime, command, rejected):
    environment = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    try:
        environment.reset(seed=123)
        _, _, terminated, truncated, info = environment.step(command)
        assert not terminated and not truncated
        assert info["action_rejected"] is rejected
    finally:
        environment.close()


def test_wasm_vector_next_step_autoreset_preserves_the_terminal_transition(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime)
    environment = make_vector_env(2, provider=provider, observation="bytes", autoreset_mode=AutoresetMode.NEXT_STEP)
    try:
        environment.reset(seed=211)
        provider.advance_worlds(7)
        _, rewards, terminated, truncated, info = environment.step(["mgquit", "look"])
        assert rewards.tolist() == [0, 0]
        assert terminated.tolist() == [True, False]
        assert truncated.tolist() == [False, False]
        assert info["raw_bytes"][0]
        terminal_bytes = info["raw_bytes"][0]
        _, rewards, terminated, truncated, info = environment.step(["ignored reset action", "look"])
        assert rewards.tolist() == [0, 0]
        assert not terminated.any() and not truncated.any()
        assert info["raw_bytes"][0] != terminal_bytes
        assert b"ignored reset action" not in info["raw_bytes"][0]
        # Relogin after a player's departure retains even an independent slot's existing world and clock.
        assert provider.advance_worlds(0) == {0: 9, 1: 9}
        _, _, terminated, truncated, _ = environment.step(["look", "look"])
        assert not terminated.any() and not truncated.any()
    finally:
        environment.close()


def test_eight_wasm_worlds_can_reset_step_and_reset_again(wasm_runtime):
    environment = make_vector_env(8, provider=WasmtimeProvider(runtime=wasm_runtime))
    try:
        for seed in (41, 42):
            environment.reset(seed=seed)
            observations, rewards, terminated, truncated, _ = environment.step(["look"] * 8)
            assert len(observations["text"]) == 8
            assert all(observations["text"])
            assert rewards.tolist() == [0] * 8
            assert not terminated.any() and not truncated.any()
    finally:
        environment.close()


def test_wasm_vector_worlds_do_not_deliver_messages_to_each_other(wasm_runtime):
    environment = make_vector_env(2, observation="bytes", provider=WasmtimeProvider(runtime=wasm_runtime))
    try:
        environment.reset(seed=211)
        _, _, terminated, truncated, info = environment.step(['tell "isolated trial" to alba', "look"])
        assert not terminated.any() and not truncated.any()
        assert b"isolated trial" not in info["raw_bytes"][1]
    finally:
        environment.close()


def test_closing_the_wasm_environments_releases_their_workers(wasm_runtime):
    before = set(threading.enumerate())
    for environment in (
        make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}),
        make_vector_env(2, provider=WasmtimeProvider(runtime=wasm_runtime)),
        make_parallel_env(2, provider=WasmtimeProvider(runtime=wasm_runtime, worlds=1)),
    ):
        try:
            environment.reset(seed=211)
        finally:
            environment.close()
            environment.close()
    assert not [
        thread for thread in threading.enumerate() if thread not in before and thread.name.startswith("wasmtime-world")
    ]


def test_a_copied_module_needs_no_neighbouring_game_data(tmp_path):
    module = tmp_path / "mud2-wasi.wasm"
    shutil.copyfile(str(files("mudgym_wasm_engine").joinpath("wasi/mud2-wasi.wasm")), module)
    runtime = WasmtimeRuntime(wasm_path=module)
    provider = WasmtimeProvider(runtime=runtime)
    environment = make_vector_env(1, provider=provider)
    try:
        environment.reset(seed=123)
        _, _, terminated, truncated, info = environment.step(["faq 1"])
        assert not terminated.any() and not truncated.any()
        assert b"Elizabethan Tearoom" in info["raw_bytes"][0]
        assert list(tmp_path.iterdir()) == [module]
    finally:
        environment.close()


@pytest.mark.parametrize("mode", ["scalar", "vector", "parallel"])
def test_each_joint_step_advances_worlds_once_and_observation_does_not(wasm_runtime, mode):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1 if mode == "parallel" else None)
    if mode == "scalar":
        connection = provider.create_connections(1)[0]
        environment = make_env(connection=connection)
        action = "look"
        initial_ticks = {0: 0}
        next_ticks = {0: 1}
    elif mode == "vector":
        environment = make_vector_env(2, provider=provider)
        action = ["look", "look"]
        initial_ticks = {0: 0, 1: 0}
        next_ticks = {0: 1, 1: 1}
    else:
        environment = make_parallel_env(2, provider=provider)
        action = {"player_0": "look", "player_1": "look"}
        initial_ticks = {0: 0}
        next_ticks = {0: 1}
    try:
        environment.reset(seed=123)
        assert provider.advance_worlds(0) == initial_ticks
        environment.step(action)
        assert provider.advance_worlds(0) == next_ticks
    finally:
        environment.close()
        provider.close()


def test_scalar_and_vector_reset_apply_the_same_engine_seed(wasm_runtime):
    scalar = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime})
    vector = make_vector_env(1, provider=WasmtimeProvider(runtime=wasm_runtime))
    try:
        for seed in (123, 456, 123):
            scalar_observation, _ = scalar.reset(seed=seed)
            vector_observation, _ = vector.reset(seed=seed)
            assert scalar_observation["room_name_index"] == vector_observation["room_name_index"][0]
            scalar_observation, *_ = scalar.step("look")
            vector_observation, *_ = vector.step(["look"])
            assert scalar_observation["text"] == vector_observation["text"][0]
    finally:
        scalar.close()
        vector.close()


def test_bytes_observation_retains_peer_output(wasm_runtime):
    environment = make_parallel_env(2, observation="bytes", provider=WasmtimeProvider(runtime=wasm_runtime, worlds=1))
    try:
        environment.reset(seed=211)
        observations, _, terminated, truncated, info = environment.step(
            {"player_0": 'tell "packaging trial" to alba', "player_1": "look"}
        )
        assert not any(terminated.values()) and not any(truncated.values())
        assert b"packaging trial" in info["player_1"]["raw_bytes"]
        for agent, observation in observations.items():
            raw = info[agent]["raw_bytes"]
            assert observation["raw_bytes"].tobytes()[: len(raw)] == raw
            assert info[agent]["transport"]["sent_lines"] == (
                ['tell "packaging trial" to alba'] if agent == "player_0" else ["look"]
            )
    finally:
        environment.close()


@pytest.mark.parametrize("observation", ["parsed", "bytes"])
def test_natural_world_reset_returns_final_transition_and_can_restart(wasm_runtime, observation):
    environment = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}, observation=observation)
    try:
        environment.reset(seed=51)
        assert environment.session.connection.advance_world_ticks(3208) == 3208
        _, reward, terminated, truncated, info = environment.step("look")
        assert terminated
        assert not truncated
        # The seeded journey earns survival bonuses during the explicitly advanced interval.
        assert reward == 300
        assert b"Auto-reset initiated" in info["raw_bytes"]
        environment.reset(seed=51)
        assert environment.session.connection.advance_world_ticks(0) == 0
        _, _, terminated, truncated, _ = environment.step("look")
        assert not terminated and not truncated
    finally:
        environment.close()
