import subprocess
import uuid
from collections.abc import Callable

import pexpect

from mudgym.connections.config import (
    CONTAINER_PREFIX,
    DEFAULT_ACCOUNT_ID,
    DEFAULT_PASSWORD,
    DOCKER_IMAGE,
)
from mudgym.connections.connection import MudConnection
from mudgym.connections.docker_image import ensure_docker_image
from mudgym.connections.prompts import Prompt, PromptSpec
from mudgym.logs import get_logger

logger = get_logger(__name__)


class DockerRunConnection(MudConnection):
    """
    Docker run connection.

    Spins up a new container via `docker run` for each connection.
    """

    initial_prompt: PromptSpec = Prompt.OPTION

    def __init__(
        self,
        container_name: str | None = None,
        image_name: str | None = DOCKER_IMAGE,
        use_tty: bool = True,
        *,
        account_id: str = DEFAULT_ACCOUNT_ID,
        password: str = DEFAULT_PASSWORD,
        persona_slot: int | None = None,
        db_slot: int | None = None,
        name_generator: Callable[[], str] | None = None,
    ):
        super().__init__(
            account_id=account_id,
            password=password,
            persona_slot=persona_slot,
            db_slot=db_slot,
            name_generator=name_generator,
        )

        # generate unique container name if not provided
        self.container_name = container_name or f"{CONTAINER_PREFIX}_{uuid.uuid4().hex}"
        self.image_name = image_name
        self.use_tty = use_tty

        self.command = self.build_command()

    def build_command(self) -> list[str]:
        return [
            "docker",
            "run",
            "--name",
            self.container_name,
            "--init",
            "--rm",
            "-it" if self.use_tty else "-i",
            "--ipc=private",
            "--shm-size=100mb",
            "-e",
            f"L0={self.account_id}",
            "-e",
            f"L1={self.password}",
            self.image_name,
        ]

    def cleanup_container(self) -> None:
        """Remove any existing container with our name to avoid conflicts.

        This handles the case where a previous container wasn't properly cleaned up,
        e.g., if the process was killed abruptly or marimo re-ran a cell.
        """
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                check=False,  # Don't raise if container doesn't exist
            )
            logger.debug("docker.container.cleanup", container_name=self.container_name)
        except Exception as e:
            logger.debug("docker.container.cleanup.failed", container_name=self.container_name, error=str(e))

    def spawn(self) -> pexpect.spawn:
        """Spawn the Docker container, ensuring any stale container is cleaned up first."""
        ensure_docker_image(self.image_name)
        self.cleanup_container()
        return super().spawn()
