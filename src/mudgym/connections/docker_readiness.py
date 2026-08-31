import subprocess
import time

from mudgym.logs import get_logger

logger = get_logger(__name__)

WORLD_READINESS_TIMEOUT_SECONDS = 60.0
WORLD_READINESS_POLL_SECONDS = 0.01


class DockerWorldReadinessError(RuntimeError):
    """A Docker-backed MUD world did not finish resetting before login."""


def wait_for_docker_worlds_ready(
    container_id: str,
    slots: int,
    *,
    timeout_seconds: float = WORLD_READINESS_TIMEOUT_SECONDS,
) -> None:
    """Wait until every requested mmud slot has completed its initial reset."""
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    probe_script = (
        f'slot=0; while [ "$slot" -lt {slots} ]; do '
        'grep -qF "[Reset took" "/home/mudgod/LOGS/reset.$slot/mmud" 2>/dev/null '
        '|| { printf "%s\\n" "$slot"; exit 1; }; '
        "slot=$((slot + 1)); done"
    )
    command = ["docker", "exec", container_id, "/bin/sh", "-c", probe_script]

    while True:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(
                "docker.container.ready",
                container_id=container_id,
                slots=slots,
                elapsed_seconds=time.monotonic() - started_at,
            )
            return

        docker_error = result.stderr.strip()
        if "No such container" in docker_error or "is not running" in docker_error:
            raise DockerWorldReadinessError(
                f"Docker container {container_id} stopped before {slots} MUD world(s) became ready: {docker_error}"
            )

        if time.monotonic() >= deadline:
            first_unready_slot = result.stdout.strip() or "unknown"
            raise DockerWorldReadinessError(
                f"Docker container {container_id} did not make {slots} MUD world(s) ready within "
                f"{timeout_seconds:g} seconds; first unready slot: {first_unready_slot}"
            )

        time.sleep(WORLD_READINESS_POLL_SECONDS)
