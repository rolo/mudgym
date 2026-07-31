import os

from mudgym.connections.config import AVAILABLE_CONNECTIONS
from mudgym.connections.connection import MudConnection
from mudgym.connections.docker_exec import DockerExecConnection
from mudgym.connections.docker_run import DockerRunConnection
from mudgym.logs import get_logger

logger = get_logger(__name__)

connections: dict[str, type[MudConnection]] = {
    "docker_run": DockerRunConnection,
    "docker_exec": DockerExecConnection,
}

# connections available in the current environment
connection_slugs = [conn.strip() for conn in AVAILABLE_CONNECTIONS.split(",")]
available_connections: list[type[MudConnection]] = []
available_connections_dict: dict[str, type[MudConnection]] = {}

for slug in connection_slugs:
    conn_cls = connections.get(slug)
    if conn_cls is not None and conn_cls.is_available():
        available_connections.append(conn_cls)
        available_connections_dict[slug] = conn_cls

# default connection can be specified as slug via env var, or defaults to first in available_connections
_default_slug = os.getenv("MUDGYM_DEFAULT_CONNECTION")
if _default_slug:
    default_connection = available_connections_dict.get(_default_slug, available_connections[0])
else:
    default_connection = available_connections[0]


def list_connections() -> list[dict[str, str | bool]]:
    """
    Get information about all registered connections.

    Returns:
        A list of dictionaries containing:
            - name: Connection slug/name
            - class_name: Connection class name
            - available: Whether the connection is available
            - is_default: Whether this is the default connection
    """
    result = []
    requested = {slug: idx for idx, slug in enumerate(connection_slugs)}

    for slug, conn_class in connections.items():
        is_requested = slug in requested
        result.append(
            {
                "name": slug,
                "class_name": conn_class.__name__,
                "available": is_requested and conn_class.is_available(),
                "is_default": conn_class == default_connection,
            }
        )
    return result


def show_connections() -> None:
    """Display a structured list of all connections and their status."""
    conn_list = list_connections()

    logger.info("connections.list")
    for conn in conn_list:
        default_text = " (default)" if conn["is_default"] else ""
        print(
            f"{conn['name']}: {conn['class_name']} ({'available' if conn['available'] else 'unavailable'}){default_text}"
        )


if __name__ == "__main__":
    show_connections()
