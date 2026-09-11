"""Exercise the shipped module without any WASI filesystem preopens."""

from importlib import metadata
from importlib.resources import files

import pytest
from packaging.requirements import Requirement
from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig

from mudgym.connections.persona import PERSONA_NAMES


def test_all_shared_persona_names_can_join_the_same_world(wasm_runtime):
    world = wasm_runtime.create_world(max_players=len(PERSONA_NAMES), seed=123)
    try:
        players = [world.add_session(name) for name in PERSONA_NAMES]
        assert len({player.player_id for player in players}) == len(PERSONA_NAMES)
        for player in players:
            assert b"Elizabethan tearoom" in player.send("look", timeout_ms=5000).output
    finally:
        world.shutdown()


@pytest.mark.parametrize("privilege", [1, 2, 3], ids=["wizard", "arch", "god"])
def test_distributed_engine_rejects_privileged_admission_through_the_raw_abi(wasm_runtime, privilege):
    world = wasm_runtime.create_world(max_players=1, seed=123)
    try:
        result = world._call_with_text(
            "mud2_shared_add_session",
            "Ada",
            arguments_after_text=(0, 0, 0, 0, 0, 0, 0, privilege, 5000),
        )
        assert result == -1
        assert world._call("mud2_shared_player_count") == 0
        player = world.add_session("Ada")
        assert b"Elizabethan tearoom" in player.send("look", timeout_ms=5000).output
    finally:
        world.shutdown()


def test_required_wasm_engine_accepts_patch_releases_only():
    requirements = [Requirement(value) for value in metadata.requires("mudgym")]
    (engine,) = [requirement for requirement in requirements if requirement.name == "mudgym-wasm-engine"]
    assert engine.url is None
    assert engine.marker is None
    for version in ("0.1.0", "0.1.1", "0.1.99"):
        assert version in engine.specifier
    for version in ("0.0.9", "0.2.0", "1.0.0"):
        assert version not in engine.specifier


def test_engine_wheel_contains_only_namespace_and_wasm():
    distribution = metadata.distribution("mudgym-wasm-engine")
    payload = {
        str(path)
        for path in distribution.files
        if str(path).startswith("mudgym_wasm_engine/") and "__pycache__" not in str(path)
    }
    assert payload == {"mudgym_wasm_engine/__init__.py", "mudgym_wasm_engine/wasi/mud2-wasi.wasm"}


def test_module_boots_and_serves_help_without_any_preopened_directory():
    config = Config()
    config.wasm_exceptions = True
    engine = Engine(config)
    module = Module(engine, files("mudgym_wasm_engine").joinpath("wasi/mud2-wasi.wasm").read_bytes())
    store = Store(engine)
    wasi = WasiConfig()
    wasi.inherit_stderr()
    store.set_wasi(wasi)
    linker = Linker(engine)
    linker.define_wasi()
    exports = linker.instantiate(store, module).exports(store)
    exports["_initialize"](store)
    assert exports["mud2_seed_world"](store, 123.0) == 1
    assert exports["mud2_shared_init_with_options"](store, 1, 0, 1767225600.0, 0, 1) == 1
    try:
        name = b"Ada\0"
        pointer = exports["malloc"](store, len(name))
        exports["memory"].write(store, name, pointer)
        player = exports["mud2_shared_add_session"](store, pointer, 0, 0, 0, 0, 0, 0, 0, 0, 5000)
        exports["free"](store, pointer)
        assert player >= 0
        generation = exports["mud2_shared_last_session_generation"](store)
        for command, expected in (("help movement", b"NORTHEAST"), ("faq 1", b"Elizabethan Tearoom")):
            text = command.encode("latin-1") + b"\0"
            pointer = exports["malloc"](store, len(text))
            exports["memory"].write(store, text, pointer)
            length = exports["mud2_shared_step_session"](store, player, generation, pointer, 5000)
            exports["free"](store, pointer)
            assert length > 0
            output = exports["mud2_shared_output"](store)
            assert expected in bytes(exports["memory"].read(store, output, output + length))
    finally:
        assert exports["mud2_shared_shutdown"](store) == 0
