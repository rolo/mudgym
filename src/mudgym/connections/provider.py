from typing import Protocol

from mudgym.connections.connection import MudConnection


class ConnectionProvider(Protocol):
    """Provide connections backed by shared resources.

    The provider determines how players are arranged across worlds. Once create_connections returns, the caller owns the connections. If creation fails, the provider cleans up that call's resources. The owning environment closes the provider after its connections.
    """

    def create_connections(self, count: int) -> list[MudConnection]:
        """Create exactly ``count`` connections, cleaning up this call if it fails."""
        ...

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        """Reset managed resources, interpreting seeds according to the provider's topology."""
        ...

    def close(self) -> None:
        """Close the provider and clean up shared resources."""
        ...
