from __future__ import annotations

from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

# Pyodide's own modules: this runtime is only ever loaded in a browser, by default_runtime or by name.
from js import Uint8Array, WebAssembly  # type: ignore[import-not-found]
from pyodide.code import run_js  # type: ignore[import-not-found]
from pyodide.ffi import to_js  # type: ignore[import-not-found]

from .futures import InlineExecutor
from .world import WasmWorld, engine_module_path


class BrowserInstance:
    """One instance of the engine module and its linear memory, reached through Pyodide's JavaScript bridge."""

    def __init__(self, instance: Any) -> None:
        self.exports = instance.exports
        self.memory = self.exports.memory
        # Looking an export up makes a new proxy each time, so each is looked up once.
        self.functions: dict[str, Any] = {}

    def call(self, name: str, *arguments: Any) -> Any:
        function = self.functions.get(name)
        if function is None:
            function = self.functions[name] = getattr(self.exports, name)
        return function(*arguments)

    def write(self, pointer: int, data: bytes) -> None:
        # A grown memory replaces its buffer, so a view is taken afresh for every copy.
        Uint8Array.new(self.memory.buffer, pointer, len(data)).assign(data)

    def read(self, pointer: int, length: int) -> bytes:
        return Uint8Array.new(self.memory.buffer, pointer, length).to_bytes()


class BrowserRuntime:
    """Compile one WASI module with the browser's WebAssembly, then instantiate any number of isolated worlds."""

    backend = "browser"

    def __init__(self, *, wasm_path: Path | None = None) -> None:
        """Load the self-contained engine module from its wheel or an explicit module path."""
        self.instantiate = run_js(resources.files(__package__).joinpath("wasi_shim.js").read_text())
        # An older browser would reject the module with a compile error that names no browser. JavaScript's null, for
        # a browser that can run it, arrives as None or Pyodide's jsnull, and both are false.
        if unsupported := self.instantiate.unsupportedReason():
            raise RuntimeError(unsupported)
        self.wasm_path = engine_module_path(wasm_path)
        # Python in a browser runs in a worker, where compiling synchronously has no size limit.
        self.module = WebAssembly.Module.new(to_js(self.wasm_path.read_bytes()))
        if "mud2_embedded_assets_version" not in {export.name for export in WebAssembly.Module.exports(self.module)}:
            raise RuntimeError("The WASI module has no embedded assets. Install a compatible mudgym-wasm-engine wheel.")

    def create_world(
        self,
        *,
        max_players: int,
        seed: int,
        civil_time_anchor: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    ) -> WasmWorld:
        return WasmWorld(
            BrowserInstance(self.instantiate(self.module)),
            max_players=max_players,
            seed=seed,
            civil_time_anchor=civil_time_anchor,
        )

    def executor(self, worlds: int) -> InlineExecutor:
        """Build and advance worlds one after another, since Pyodide cannot start threads."""
        return InlineExecutor()
