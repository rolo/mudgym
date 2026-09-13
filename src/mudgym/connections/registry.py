from collections.abc import Callable

from mudgym.connections.connection import MudConnection
from mudgym.connections.wasm import WasmtimeProvider, create_connection

connections: dict[str, Callable[..., MudConnection]] = {"wasm": create_connection}

# The env factory resolves these defaults at call time so recording tools can replace them.
default_connection = create_connection


def default_parallel_provider_factory() -> WasmtimeProvider:
    """Create the default provider for players who share one world."""
    return WasmtimeProvider(worlds=1)
