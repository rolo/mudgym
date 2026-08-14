import os
import subprocess
import time
from contextlib import suppress
from threading import Lock
from typing import Protocol
from uuid import uuid4

from mudgym.connections.config import DOCKER_IMAGE
from mudgym.connections.connection import MudConnection
from mudgym.connections.docker_exec import DockerExecConnection
from mudgym.logs import get_logger

logger = get_logger(__name__)


class ConnectionProvider(Protocol):
    """Provides batches of connections backed by some shared set of resources.

    The provider decides what those connections are connected to. That might be one world per
    connection, several players sharing a world, or something else entirely - the vector env doesn't need to
    know.

    Once ``create_connections`` returns, the caller owns the connections. If it fails before returning, the
    provider cleans up whatever that call managed to create. The provider itself stays around until the owning
    environment closes, as it may also own containers or other resources the connections depend on.
    """

    def create_connections(self, count: int) -> list[MudConnection]:
        """Create exactly ``count`` connections, cleaning up this call if it fails."""
        ...

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        """Reset managed resources, interpreting seeds according to the provider's topology."""
        ...

    def close(self) -> None:
        """Close the provider and clean up shared infrastructure."""
        ...


class DockerExecProvider(ConnectionProvider):
    """Provides one fixed batch of connections backed by Docker worlds.

    ``worlds`` describes the topology, while the count passed to ``create_connections`` is simply how many connections the caller needs. If ``worlds`` is omitted we use one world per connection. Docker needs the whole count up front to size its containers, so this particular provider only supplies one batch.
    """

    def __init__(
        self,
        worlds: int | None = None,
        image: str = DOCKER_IMAGE,
        container_id: str | None = None,
        worlds_per_container: int = 128,
        *,
        connection_class: type[MudConnection] = DockerExecConnection,
        connection_kwargs: dict | None = None,
    ):
        if worlds is not None and worlds < 1:
            raise ValueError("worlds must be at least 1.")
        if worlds_per_container < 1:
            raise ValueError("worlds_per_container must be at least 1.")
        self.worlds = worlds
        self.image = image
        self.container_id = container_id
        self.worlds_per_container = worlds_per_container
        self.connection_class = connection_class
        self.connection_kwargs = dict(connection_kwargs or {})

        reserved = {"account_id", "db_slot", "container_id"}
        bad = reserved & set(self.connection_kwargs)
        if bad:
            raise ValueError(f"connection_kwargs must not set provider-managed keys: {sorted(bad)}")

        self._lock = Lock()
        self._closed = False
        self._batch_created = False
        self._owner_pid = os.getpid()

        self.containers: list[str] = []
        # A supplied container belongs to its caller. Containers we start ourselves belong to us.
        self.owns_containers = container_id is None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Locks cannot be pickled, so an unpickled provider gets a fresh one below.
        state.pop("_lock", None)
        # Once a batch exists, a pickled copy only refers to the allocating process's containers.
        # It must not decide they are now its containers and stop them on close.
        if self._batch_created:
            state["owns_containers"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()
        # Before allocation there is nothing to inherit. This process owns anything it starts later.
        if not self._batch_created:
            self._owner_pid = os.getpid()

    def prepare_shared_container(self, slots: int) -> str:
        """Start a new container with multiple game slots."""
        unique = uuid4().hex[:6]
        container_name = f"mud_shared_{int(time.time())}_{slots}_{unique}"

        logger.info("provider.container.starting", container_name=container_name, slots=slots)

        # boot prepares the worlds and exits, so sleep keeps the container around for docker exec.
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

    def create_connections(self, count: int) -> list[MudConnection]:
        """Start enough Docker worlds and create this provider's one connection batch."""
        if count < 1:
            raise ValueError("count must be at least 1.")

        connections: list[MudConnection] = []
        allocation_started = False
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Provider is closed.")
                if self._batch_created:
                    raise RuntimeError("Provider has already created its connection batch.")
                self._batch_created = True
                allocation_started = True
                if self.owns_containers:
                    # Allocation is lazy, so the process doing this work owns the containers. It
                    # may not be the process that originally constructed or pickled the provider.
                    self._owner_pid = os.getpid()

                world_count = self.worlds if self.worlds is not None else count
                if self.container_id:
                    if world_count > self.worlds_per_container:
                        raise ValueError("Single container mode only supports up to 'worlds_per_container' worlds")
                    self.containers.append(self.container_id)
                else:
                    starts = range(0, world_count, self.worlds_per_container)
                    total = (world_count + self.worlds_per_container - 1) // self.worlds_per_container
                    for index, start in enumerate(starts):
                        slots = min(self.worlds_per_container, world_count - start)
                        logger.info(
                            "provider.container.launching",
                            idx=index + 1,
                            total=total,
                            slots=slots,
                            start=start,
                        )
                        self.containers.append(self.prepare_shared_container(slots))

                for env_index in range(count):
                    # Connections wrap around the configured worlds, which is how several players
                    # can share one. ``world_index`` is provider-wide; ``db_slot`` is the historical
                    # MUD2 name for the index inside one container. The distinction only matters
                    # once we need more worlds than one MUD2 process can hold (a problem the
                    # original authors can probably be forgiven for not anticipating).
                    world_index = env_index % world_count
                    container_index, db_slot = divmod(world_index, self.worlds_per_container)

                    kwargs = dict(self.connection_kwargs)
                    kwargs["account_id"] = f"W{env_index + 1:08d}"
                    kwargs["db_slot"] = db_slot
                    connections.append(
                        self.connection_class(
                            container_id=self.containers[container_index],
                            **kwargs,
                        )
                    )

            return connections
        except BaseException:
            # A failure after allocation starts belongs to us: close the connections we did make,
            # then the containers beneath them. Errors here must not replace the original failure.
            # A rejected second call never started an allocation, so it leaves the first batch alone.
            if not allocation_started:
                raise
            for connection in connections:
                with suppress(Exception):
                    connection.close()
            try:
                self.close()
            except Exception:
                logger.error("provider.batch_cleanup_failed", exc_info=True)
            raise

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        """Docker worlds currently retain their running state across environment resets."""

    def close(self) -> None:
        """Close the shared infrastructure, but not the connections owned by environments."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            to_stop = list(self.containers) if self.owns_containers else []
            self.containers.clear()

        # Forked and spawned copies can close their references, but only the allocating process
        # gets to stop the actual containers.
        if os.getpid() != self._owner_pid:
            logger.debug("provider.container.close.skip_non_owner", pid=os.getpid(), owner_pid=self._owner_pid)
            return

        first_exc = None
        for container_id in to_stop:
            try:
                subprocess.run(["docker", "stop", container_id], check=True, capture_output=True)
            except Exception as exc:
                logger.error(
                    "provider.container.close.stop_failed",
                    container_id=container_id,
                    pid=os.getpid(),
                    owner_pid=self._owner_pid,
                    exc_info=True,
                )
                if first_exc is None:
                    first_exc = exc

        if first_exc is not None:
            raise first_exc
