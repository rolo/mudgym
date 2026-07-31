"""Fail-fast Docker preflight shared by the Docker-backed connections.

An uncached image pull can never fit inside the game's first-prompt timeout,
so the image must be present before any container spawns. This module makes
each failure mode explicit instead of surfacing a raw pexpect timeout: no
Docker binary, an unreachable daemon, a registry that refuses access, and a
tag that does not exist.
"""

import shutil
import subprocess

from mudgym.logs import get_logger

logger = get_logger(__name__)

INSPECT_TIMEOUT_SECONDS = 30
PULL_TIMEOUT_SECONDS = 600

# Images verified once per process; repeated spawns skip the inspect subprocess.
_verified_images: set[str] = set()


class DockerSetupError(RuntimeError):
    """Docker or the game image is unusable; the message says how to fix it."""


def ensure_docker_image(image_name: str) -> None:
    """Make sure ``image_name`` is usable locally, pulling it on first use."""
    if image_name in _verified_images:
        return

    docker = shutil.which("docker")
    if docker is None:
        raise DockerSetupError(
            "Docker is required to run the MudGym game but no `docker` executable was found on PATH. "
            "Install Docker (https://docs.docker.com/get-docker/) and try again."
        )

    try:
        inspect = subprocess.run(
            [docker, "image", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=INSPECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise DockerSetupError(
            f"Docker did not answer `docker image inspect` within {INSPECT_TIMEOUT_SECONDS}s. "
            "Check that the Docker daemon is healthy and try again."
        ) from error

    if inspect.returncode != 0:
        inspect_error = inspect.stderr.strip()
        lowered = inspect_error.lower()
        # A missing image also mentions the daemon ("Error response from daemon:
        # No such image"), so only treat connection failures as daemon-down.
        if "cannot connect" in lowered or "is the docker daemon running" in lowered:
            raise DockerSetupError(
                f"Docker is installed but its daemon is not reachable: {inspect_error} Start Docker and try again."
            )
        _pull_image(docker, image_name)

    _verified_images.add(image_name)


def _pull_image(docker: str, image_name: str) -> None:
    logger.info("docker.image.pull.start", image=image_name)
    try:
        pull = subprocess.run(
            [docker, "pull", image_name],
            capture_output=True,
            text=True,
            timeout=PULL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise DockerSetupError(
            f"Pulling the MudGym game image {image_name!r} did not finish within "
            f"{PULL_TIMEOUT_SECONDS}s. Check your network connection and try again, or run "
            f"`docker pull {image_name}` yourself to watch its progress."
        ) from error

    if pull.returncode == 0:
        logger.info("docker.image.pull.done", image=image_name)
        return

    pull_error = pull.stderr.strip()
    lowered = pull_error.lower()
    if "denied" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        detail = (
            "The registry refused access to it. If this is a released MudGym version the image is "
            "public and no login is needed, so check for a stale `docker login ghcr.io` credential."
        )
    elif "manifest unknown" in lowered or "not found" in lowered:
        detail = "That tag does not exist on the registry, so check the image name and tag."
    else:
        detail = "Check your network connection and the Docker daemon logs."
    raise DockerSetupError(f"Could not pull the MudGym game image {image_name!r}. {detail}\nDocker said: {pull_error}")
