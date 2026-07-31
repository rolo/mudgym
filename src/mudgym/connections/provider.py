import os
import subprocess
import time
from threading import Lock
from typing import Protocol, runtime_checkable
from uuid import uuid4

from mudgym.connections.config import DOCKER_IMAGE
from mudgym.connections.connection import MudConnection
from mudgym.connections.docker_exec import DockerExecConnection
from mudgym.logs import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ConnectionProvider(Protocol):
    """
    Protocol for providers that create connections to the game.

    Manages shared infrastructure (containers, slot bootstrapping) that may be
    needed by multiple connections.

    Lifecycle contract:
    - `create_connection(env_index)` is expected to be called once per env (per process).
      The returned connection is owned by the environment.
    - The provider creates connections but doesn't manage their lifecycle.
      Connections are closed by environments when `env.close()` is called.
    - The provider must be closed after all its environments are closed, since it owns the shared infrastructure
    (e.g. containers) they depend on.
    """

    def create_connection(self, env_index: int) -> MudConnection:
        """Create a connection for the given environment index."""
        ...

    def close(self) -> None:
        """Close the provider and clean up shared infrastructure."""
        ...


class DockerExecProvider(ConnectionProvider):
    """
    Uses shared containers with multiple game worlds via docker exec.
    """

    def __init__(
        self,
        worlds: int = 128,
        image: str = DOCKER_IMAGE,
        container_id: str | None = None,
        worlds_per_container: int = 128,
        *,
        connection_class: type[MudConnection] = DockerExecConnection,
        connection_kwargs: dict | None = None,
    ):
        if container_id and worlds > worlds_per_container:
            raise ValueError("Single container mode only supports up to 'worlds_per_container' worlds")

        self.worlds = worlds
        self.image = image
        self.worlds_per_container = max(1, worlds_per_container)
        self._connection_class = connection_class
        self._connection_kwargs = dict(connection_kwargs or {})

        reserved = {"account_id", "db_slot", "container_id"}
        bad = reserved & set(self._connection_kwargs)
        if bad:
            raise ValueError(f"connection_kwargs must not set provider-managed keys: {sorted(bad)}")
        self._lock = Lock()
        self._closed = False
        self._owner_pid = os.getpid()

        self._containers: list[dict[str, object]] = []

        if container_id:
            self._containers.append(
                {
                    "container_id": container_id,
                    "capacity": min(worlds, self.worlds_per_container),
                    "start": 0,
                    "end": min(worlds, self.worlds_per_container),
                    "owns": False,
                }
            )
        else:
            remaining = worlds
            start = 0
            launch_plan: list[tuple[int, int]] = []
            while remaining > 0:
                capacity = min(self.worlds_per_container, remaining)
                launch_plan.append((capacity, start))
                start += capacity
                remaining -= capacity

            launched: list[dict[str, object]] = []
            for index, (capacity, start_idx) in enumerate(launch_plan):
                logger.info(
                    "provider.container.launching",
                    idx=index + 1,
                    total=len(launch_plan),
                    capacity=capacity,
                    start=start_idx,
                    end=start_idx + capacity,
                )
                container_id = self.prepare_shared_container(capacity)
                launched.append(
                    {
                        "container_id": container_id,
                        "capacity": capacity,
                        "start": start_idx,
                        "end": start_idx + capacity,
                        "owns": True,
                    }
                )
            self._containers.extend(launched)

        if not self._containers:
            raise RuntimeError("DockerExecProvider failed to initialise any containers")

    def __getstate__(self):
        state = self.__dict__.copy()
        # lock cannot be pickled; rebuild it in children.
        state.pop("_lock", None)
        # strip ownership so pickled copies don't try to stop containers
        # that the parent process owns.
        state["_containers"] = [dict(entry, owns=False) for entry in state["_containers"]]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()

    def prepare_shared_container(self, slots: int) -> str:
        """Start a new container with multiple game slots."""
        unique = uuid4().hex[:6]
        container_name = f"mud_shared_{int(time.time())}_{slots}_{unique}"

        logger.info("provider.container.starting", container_name=container_name, slots=slots)

        # run container with sleep infinity to prevent it from exiting
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--init",
                "--rm",
                "--ipc",
                "private",
                "--pids-limit",
                "4096",
                "--log-driver",
                "none",
                "--name",
                container_name,
                self.image,
                "/bin/sh",
                "-c",
                f"/app/bin/boot -n {slots} -f -k && sleep infinity",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        container_id = result.stdout.strip()
        logger.info("provider.container.started", container_id=container_id)

        return container_id

    def create_connection(self, env_index: int) -> MudConnection:
        """
        Create a connection for the given environment index.

        Connections always map env_index to world_index via `db_slot = env_index % self.worlds` to allow
        multiple agents to exist in the same game world (when envs > worlds).

        db_slot is the historical term to refer to the database index within the container.

        world_index is a mudgym term that refers to the index of worlds this provider is creating for training.

        This distinction is only really relevant for trying to run across multiple containers as the original
        MUD2 engine supports up to 128 worlds, probably because it was hard to imagine that one day people from
        the future would want to run more.
        """
        if self._closed:
            raise RuntimeError("Provider is closed.")
        with self._lock:
            # env_index mapped to world_index via modulo (for multiple envs/agents per world)
            world_index = env_index % self.worlds

            container = None
            for entry in self._containers:
                if entry["start"] <= world_index < entry["end"]:
                    container = entry
                    break

            if container is None:
                raise RuntimeError(f"No container assignment found for world index {world_index}")

            # db_slot passed is container local (0..capacity-1) because each container can run up to
            # worlds_per_container DB slots (game worlds).
            db_slot_in_container = world_index - container["start"]

            kwargs = dict(self._connection_kwargs)
            kwargs["account_id"] = f"W{env_index + 1:08d}"
            kwargs["db_slot"] = db_slot_in_container
            container_id = container["container_id"]

        return self._connection_class(container_id=container_id, **kwargs)

    def close(self) -> None:
        """Close the provider and clean up shared infrastructure (containers).

        Note: This does not close connections - environments own their connections
        and close them in env.close().
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            to_stop = [entry for entry in self._containers if entry.get("owns")]
            self._containers.clear()

        # Only the creating process should stop containers (covers fork + spawn).
        if os.getpid() != self._owner_pid:
            logger.debug("provider.container.close.skip_non_owner", pid=os.getpid(), owner_pid=self._owner_pid)
            return

        first_exc = None
        for entry in to_stop:
            container_id = entry["container_id"]
            try:
                subprocess.run(["docker", "stop", container_id], check=True, capture_output=True)
            except Exception as e:
                logger.error(
                    "provider.container.close.stop_failed",
                    container_id=container_id,
                    pid=os.getpid(),
                    owner_pid=self._owner_pid,
                    exc_info=True,
                )
                if first_exc is None:
                    first_exc = e

        if first_exc is not None:
            raise first_exc
