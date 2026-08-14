import os

from mudgym.connections.config import AVAILABLE_CONNECTIONS
from mudgym.connections.connection import MudConnection
from mudgym.connections.docker_exec import DockerExecConnection
from mudgym.connections.docker_run import DockerRunConnection
from mudgym.connections.provider import DockerExecProvider

connections: dict[str, type[MudConnection]] = {
    "docker_run": DockerRunConnection,
    "docker_exec": DockerExecConnection,
}

# Connections enabled in the current environment, preserving configured order.
available_connections_dict: dict[str, type[MudConnection]] = {}
for configured_slug in AVAILABLE_CONNECTIONS.split(","):
    slug = configured_slug.strip()
    conn_cls = connections.get(slug)
    if conn_cls is not None:
        available_connections_dict[slug] = conn_cls

# default connection can be specified as a slug, or defaults to the first configured connection
_default_slug = os.getenv("MUDGYM_DEFAULT_CONNECTION")
default_connection = available_connections_dict.get(_default_slug) if _default_slug else None
if default_connection is None:
    default_connection = next(iter(available_connections_dict.values()))


# The env factory resolves both defaults through this module at call time so tooling can replace them.
default_provider_factory = DockerExecProvider


def default_parallel_provider_factory() -> DockerExecProvider:
    """Create the default provider for players who have explicitly asked to share one world."""
    return DockerExecProvider(worlds=1)
