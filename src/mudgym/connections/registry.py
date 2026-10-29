from collections.abc import Callable

from mudgym.connections import wasm
from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import ConnectionProvider


# These reach the engine through the wasm package's lazy exports, so importing the registry loads no runtime.
def create_wasm_connection(**kwargs) -> MudConnection:
    """Create the built-in WASM engine connection."""
    return wasm.create_connection(**kwargs)


connections: dict[str, Callable[..., MudConnection]] = {"wasm": create_wasm_connection}

# The env factory resolves these defaults at call time so recording tools can replace them.
default_connection = create_wasm_connection


def default_parallel_provider_factory(**options) -> ConnectionProvider:
    """Create the default provider for players who share one world."""
    return wasm.WasmProvider(worlds=1, **options)
