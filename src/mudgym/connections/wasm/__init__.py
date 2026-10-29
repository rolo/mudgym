"""The MUD2 WASI engine as a MudGym connection.

Runs on wasmtime in native Python and on the browser's own WebAssembly in Pyodide. The runtimes load what their
host has on first use.
"""

from importlib import import_module

__all__ = ["BrowserRuntime", "WasmProvider", "WasmtimeRuntime", "create_connection", "default_runtime"]

EXPORTS = {
    "BrowserRuntime": "browser_runtime",
    "WasmProvider": "provider",
    "WasmtimeRuntime": "wasmtime_runtime",
    "create_connection": "provider",
    "default_runtime": "provider",
}


def __getattr__(name):
    if name not in EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{EXPORTS[name]}"), name)
