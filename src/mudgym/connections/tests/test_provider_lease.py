import json
import multiprocessing
import subprocess
import time

from mudgym.connections.provider import DockerExecProvider


def allocate_shared_container(result) -> None:
    provider = DockerExecProvider(
        worlds=2,
        worlds_per_container=1,
        lease_timeout_seconds=6,
    )
    provider.create_connections(2)
    result.send(provider.containers)
    result.close()
    while True:
        time.sleep(60)


def container_exists(container_id: str) -> bool:
    result = subprocess.run(
        ["docker", "container", "inspect", container_id],
        capture_output=True,
    )
    return result.returncode == 0


def test_shared_containers_expire_after_allocating_process_is_killed():
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=allocate_shared_container, args=(sender,))
    process.start()
    sender.close()
    containers = []
    try:
        assert receiver.poll(60), "allocator did not report its container within 60 seconds"
        containers = receiver.recv()
        labels = json.loads(
            subprocess.check_output(
                ["docker", "container", "inspect", "--format", "{{json .Config.Labels}}", containers[0]],
                text=True,
            )
        )
        assert labels["mudgym.role"] == "shared-world"
        assert labels["mudgym.lease.timeout-seconds"] == "6"
        assert labels["mudgym.lease.owner-pid"].isdigit()
        assert labels["mudgym.lease.id"]

        time.sleep(7)
        assert all(container_exists(container_id) for container_id in containers)

        process.kill()
        process.join(10)
        assert process.exitcode is not None

        deadline = time.monotonic() + 15
        while any(container_exists(container_id) for container_id in containers) and time.monotonic() < deadline:
            time.sleep(0.25)

        assert not any(container_exists(container_id) for container_id in containers)
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        for container_id in containers:
            subprocess.run(
                ["docker", "rm", "--force", container_id],
                check=False,
                capture_output=True,
            )
