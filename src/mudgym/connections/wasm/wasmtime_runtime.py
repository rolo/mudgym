"""Wasmtime execution of the separately packaged MUD2 WASI engine, for native Python."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig

from .world import WasmWorld, engine_module_path


class WasmtimeInstance:
    """One Store and Instance of the engine module, with the guest's stderr passed through."""

    def __init__(self, runtime: WasmtimeRuntime) -> None:
        self.store = Store(runtime.engine)
        wasi = WasiConfig()
        wasi.inherit_stderr()
        self.store.set_wasi(wasi)
        linker = Linker(runtime.engine)
        linker.define_wasi()
        self.exports = linker.instantiate(self.store, runtime.module).exports(self.store)
        self.memory = self.exports["memory"]

    def call(self, name: str, *arguments: Any) -> Any:
        return self.exports[name](self.store, *arguments)

    def write(self, pointer: int, data: bytes) -> None:
        self.memory.write(self.store, data, pointer)

    def read(self, pointer: int, length: int) -> bytes:
        return bytes(self.memory.read(self.store, pointer, pointer + length))


class WasmtimeRuntime:
    """Compile one WASI module, then instantiate any number of isolated worlds."""

    backend = "wasmtime"

    def __init__(
        self,
        *,
        wasm_path: Path | None = None,
    ) -> None:
        """Load the self-contained engine module from its wheel or an explicit module path."""
        self.wasm_path = engine_module_path(wasm_path)
        config = Config()
        config.wasm_exceptions = True
        self.engine = Engine(config)
        self.module = Module.from_file(self.engine, str(self.wasm_path))
        if "mud2_embedded_assets_version" not in {export.name for export in self.module.exports}:
            raise RuntimeError("The WASI module has no embedded assets. Install a compatible mudgym-wasm-engine wheel.")

    def create_world(
        self,
        *,
        max_players: int,
        seed: int,
        civil_time_anchor: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    ) -> WasmWorld:
        return WasmWorld(
            WasmtimeInstance(self), max_players=max_players, seed=seed, civil_time_anchor=civil_time_anchor
        )

    def executor(self, worlds: int) -> ThreadPoolExecutor:
        """Build and advance worlds in parallel, since each has its own Store."""
        return ThreadPoolExecutor(max_workers=min(worlds, os.process_cpu_count() or 1), thread_name_prefix="wasm-world")
