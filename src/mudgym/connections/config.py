import os

DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "ghcr.io/rolo/mudgym:v0.3")
CONTAINER_PREFIX = "mudgym"
DEFAULT_DOCKER_EXEC_CONTAINER_NAME = "mud2-boot"

# connection defaults
DEFAULT_ACCOUNT_ID = "W00000001"
DEFAULT_PASSWORD = "password"

# first listed in AVAILABLE_CONNECTIONS is default
AVAILABLE_CONNECTIONS = os.getenv("AVAILABLE_CONNECTIONS", "docker_run,docker_exec")


def configured_docker_exec_container_name() -> str:
    """Return the shared-container name configured for this process."""
    return os.getenv("MUDGYM_DOCKER_EXEC_CONTAINER_NAME", DEFAULT_DOCKER_EXEC_CONTAINER_NAME)
