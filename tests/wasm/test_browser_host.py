"""The WASI shim a browser hosts the engine with runs the real engine module in V8, Chrome's WebAssembly engine."""

import json
import shutil
import subprocess
from importlib import resources

import pytest

from mudgym.connections.wasm.engine_contract import WorldStatus
from mudgym.connections.wasm.world import engine_module_path

# Evaluates the shim as Pyodide's run_js does and drives one world through the same exports WasmWorld calls.
PLAY_ONE_COMMAND = """
const { readFileSync } = require("node:fs")
const [shimPath, modulePath] = process.argv.slice(1)
const instantiate = (0, eval)(readFileSync(shimPath, "utf8"))
const instance = instantiate(new WebAssembly.Module(readFileSync(modulePath)))
const engine = instance.exports
const text = (value) => {
  const bytes = Buffer.from(value + "\\0", "latin1")
  const pointer = engine.malloc(bytes.length)
  new Uint8Array(engine.memory.buffer, pointer, bytes.length).set(bytes)
  return pointer
}
const output = (length) => Buffer.from(engine.memory.buffer, engine.mud2_shared_output(), length).toString("latin1")
engine._initialize()
const started = engine.mud2_seed_world(7) === 1 && engine.mud2_shared_init_with_options(1, 0, 1767225600, 0, 1) === 1
const name = text("Freya")
const player = engine.mud2_shared_add_session(name, 0, 1, 0, 0, 0, 0, 0, 0, 5000)
engine.free(name)
const generation = engine.mud2_shared_last_session_generation()
const arrival = output(engine.mud2_shared_receive_session(player, generation))
const command = text("look")
const look = output(engine.mud2_shared_step_session(player, generation, command, 5000))
engine.free(command)
const tick = engine.mud2_shared_tick(3, 5000) === 0 ? engine.mud2_shared_current_tick() : null
engine.mud2_shared_shutdown()
console.log(JSON.stringify({ started, player, arrival, look, tick }))
"""

# Ticks a fresh world one cycle at a time until it stops, which a world first does at its natural reset.
TICK_TO_THE_RESET = """
const { readFileSync } = require("node:fs")
const [shimPath, modulePath] = process.argv.slice(1)
const instantiate = (0, eval)(readFileSync(shimPath, "utf8"))
const engine = instantiate(new WebAssembly.Module(readFileSync(modulePath))).exports
engine._initialize()
engine.mud2_seed_world(7)
engine.mud2_shared_init_with_options(1, 0, 1767225600, 0, 1)
let status = 0
let ticks = 0
while (status === 0 && ticks < 10000) {
  status = engine.mud2_shared_tick(1, 5000)
  if (status === 0) ticks += 1
}
console.log(JSON.stringify({ status, ticks, world: engine.mud2_shared_world_status() }))
"""


# Asks the shim whether this V8 can run the engine, and checks the answer against compiling the engine itself.
ASK_FOR_SUPPORT = """
const { readFileSync } = require("node:fs")
const [shimPath, modulePath] = process.argv.slice(1)
const instantiate = (0, eval)(readFileSync(shimPath, "utf8"))
let compiles = true
try {
  new WebAssembly.Module(readFileSync(modulePath))
} catch {
  compiles = false
}
console.log(JSON.stringify({ reason: instantiate.unsupportedReason(), compiles }))
"""


def run_in_node(script, *v8_flags):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    shim = resources.files("mudgym.connections.wasm").joinpath("wasi_shim.js")
    result = subprocess.run(
        [node, *v8_flags, "-e", script, str(shim), str(engine_module_path())],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_browser_wasi_shim_starts_a_world_and_plays_a_command():
    played = run_in_node(PLAY_ONE_COMMAND)
    assert played["started"] and played["player"] >= 0
    assert "tearoom" in played["arrival"].lower()
    assert "Elizabethan tearoom" in played["look"]
    assert played["tick"] == 3


def test_a_world_under_the_browser_wasi_shim_halts_at_its_natural_reset_instead_of_waiting_forever():
    # The engine pauses before a natural reset, which wasmtime sleeps through and the shim must end at once.
    ended = run_in_node(TICK_TO_THE_RESET)
    assert ended["status"] != 0
    assert ended["world"] == WorldStatus.HALTED
    assert ended["ticks"] > 3000


def test_the_browser_wasi_shim_finds_nothing_missing_where_the_engine_compiles():
    assert run_in_node(ASK_FOR_SUPPORT) == {"reason": None, "compiles": True}


def test_the_browser_wasi_shim_names_the_browsers_to_use_where_the_engine_cannot_compile():
    # V8 without exnref is an older browser's WebAssembly, which rejects the engine with a bare compile error.
    answer = run_in_node(ASK_FOR_SUPPORT, "--no-experimental-wasm-exnref")
    assert not answer["compiles"]
    assert "too old for the MUD2 engine" in answer["reason"]
    assert "Chrome 137, Firefox 131, Safari 18.4 or later" in answer["reason"]


def test_the_browser_wasi_shim_names_the_browsers_to_use_where_there_is_no_webassembly():
    # V8 without its JIT exposes no WebAssembly at all.
    answer = run_in_node(ASK_FOR_SUPPORT, "--jitless")
    assert not answer["compiles"]
    assert "This browser has no WebAssembly" in answer["reason"]
    assert "Chrome 137, Firefox 131, Safari 18.4 or later" in answer["reason"]
