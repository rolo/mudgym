import subprocess
from collections.abc import Callable

from mudgym.connections.config import (
    DEFAULT_ACCOUNT_ID,
    DEFAULT_PASSWORD,
    DOCKER_IMAGE,
    configured_docker_exec_container_name,
)
from mudgym.connections.connection import MudConnection
from mudgym.connections.docker_image import ensure_docker_image
from mudgym.connections.prompts import Prompt, PromptSpec
from mudgym.logs import get_logger

logger = get_logger(__name__)


class DockerExecConnection(MudConnection):
    """
    MUD2 connection that execs into an existing Docker container.

    Typically for using a single container running with multiple game slots, or when connecting
    multiple clients to the same game slot.
    """

    initial_prompt: PromptSpec = [
        Prompt.OPTION,
        Prompt.SUPERSEDE,
        Prompt.SESSION_DYING,
    ]

    def __init__(
        self,
        container_name: str | None = None,
        container_id: str | None = None,
        start_if_missing: bool = True,
        container_image: str = DOCKER_IMAGE,
        *,
        use_tty: bool = True,
        account_id: str = DEFAULT_ACCOUNT_ID,
        password: str = DEFAULT_PASSWORD,
        persona_slot: int | None = None,
        db_slot: int | None = None,
        name_generator: Callable[[], str] | None = None,
    ):
        container_name = container_name if container_name is not None else configured_docker_exec_container_name()
        # if container_id is not provided, find the first container with the given prefix
        self.container_image = container_image
        self.container_name = container_name
        # track whether we started the container and own the lifecycle
        self._started_container = False
        self.container_id = container_id or self.find_container_id(container_name, start_if_missing)
        self.use_tty = use_tty

        super().__init__(
            account_id=account_id,
            password=password,
            persona_slot=persona_slot,
            db_slot=db_slot,
            name_generator=name_generator,
        )
        self.command = self.build_command()

    def find_container_id(self, name: str, start_if_missing: bool = True) -> str:
        """Find container ID by name."""
        out = subprocess.check_output(["docker", "ps", "-qf", f"name={name}"], text=True).strip()
        if out:
            # if multiple lines, pick first:
            return out.splitlines()[0]
        if start_if_missing:
            logger.debug("docker.container.not_found", name=name, start_if_missing=start_if_missing)
            return self.start_container()
        raise RuntimeError(f"No running container matches name={name}; start_if_missing={start_if_missing}")

    def start_container(self, slots: int = 1) -> str:
        """Start a new container and return the container ID."""
        ensure_docker_image(self.container_image)
        output = subprocess.check_output(
            [
                "docker",
                "run",
                "--init",
                "--rm",
                "-d",
                #'--ipc="private"',
                "--name",
                self.container_name,
                self.container_image,
                "/bin/sh",
                "-lc",
                f"/app/bin/boot -n {slots} -f -k",
            ],
            text=True,
        ).strip()

        # docker run -d prints the container ID on stdout
        container_id = output.splitlines()[-1] if output else ""
        if not container_id:
            raise RuntimeError(f"docker run did not return a container id for {self.container_name}")

        logger.debug("docker.container.started", container_id=container_id)
        self.container_id = container_id
        self._started_container = True
        return self.container_id

    def build_command(self) -> list[str]:
        cmd = [
            "docker",
            "exec",
        ]

        # only add TTY flag if we have a TTY (needed for GitHub Actions)
        if self.use_tty:
            cmd.append("-it")
        else:
            cmd.append("-i")

        cmd.extend(
            [
                "-e",
                f"L0={self.account_id}",
                "-e",
                f"L1={self.password}",
            ]
        )

        cmd.extend([self.container_id, "/app/bin/mudlogin", "-n"])

        return cmd

    def close(self):
        """Close the exec session, and remove the container if this connection started it.

        We only tear down the container we own (one we started because none was running). A
        pre-existing, shared container is left alone so other clients exec'd into it survive.
        """
        super().close()

        if not self._started_container:
            return

        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                check=False,  # Don't raise if the container is already gone
            )
            logger.debug("docker.container.removed", container_name=self.container_name)
        except Exception as e:
            logger.debug("docker.container.remove.failed", container_name=self.container_name, error=str(e))
        finally:
            self._started_container = False
