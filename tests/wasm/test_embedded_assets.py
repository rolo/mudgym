"""Exercise the packaged module without any WASI filesystem preopens."""

from importlib import metadata
from itertools import batched

import pytest

from mudgym.connections.persona import DEFAULT_PERSONA_POOL, Persona
from mudgym.connections.wasm.engine_contract import MAX_WORLD_SESSIONS


@pytest.mark.parametrize("personas", tuple(batched(DEFAULT_PERSONA_POOL, MAX_WORLD_SESSIONS)))
def test_all_shared_personas_can_join_in_supported_world_batches(wasm_runtime, personas):
    world = wasm_runtime.create_world(max_players=len(personas), seed=123)
    try:
        players = [world.add_session(persona) for persona in personas]
        assert len({player.player_id for player in players}) == len(personas)
        for player in players:
            assert b"Elizabethan tearoom" in player.send("look", timeout_ms=5000).output
    finally:
        world.shutdown()


def test_distributed_engine_rejects_privileged_admission_through_the_raw_abi(wasm_runtime):
    world = wasm_runtime.create_world(max_players=1, seed=123)
    try:
        result = world._call_with_text(
            "mud2_shared_add_session",
            "Ada",
            arguments_after_text=(0, 0, 0, 0, 0, 0, 0, 1, 5000),
        )
        assert result == -1
        assert world._call("mud2_shared_player_count") == 0
        player = world.add_session(Persona("Ada", "f"))
        assert b"Elizabethan tearoom" in player.send("look", timeout_ms=5000).output
    finally:
        world.shutdown()


def test_engine_wheel_contains_only_namespace_and_wasm():
    distribution = metadata.distribution("mudgym-wasm-engine")
    payload = {
        str(path)
        for path in distribution.files
        if str(path).startswith("mudgym_wasm_engine/") and "__pycache__" not in str(path)
    }
    assert payload == {"mudgym_wasm_engine/__init__.py", "mudgym_wasm_engine/wasi/mud2-wasi.wasm"}
