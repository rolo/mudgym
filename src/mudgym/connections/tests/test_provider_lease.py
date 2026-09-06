import json
import multiprocessing
import subprocess
import time

import pytest

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


@pytest.mark.parametrize("container_count", [1, 2])
def test_close_accepts_an_expired_container_and_reaps_renewers(container_count):
    provider = DockerExecProvider(worlds=container_count, worlds_per_container=1, lease_timeout_seconds=6)
    try:
        provider.create_connections(container_count)
        containers = list(provider.containers)
        renewers = list(provider._lease.renewers)
        renewers[0].terminate()

        deadline = time.monotonic() + 20
        while container_exists(containers[0]) and time.monotonic() < deadline:
            time.sleep(0.25)
        assert not container_exists(containers[0]), "container did not expire"

        provider.close()

        assert all(renewer.returncode is not None for renewer in renewers)
        assert not any(container_exists(container_id) for container_id in containers)
        assert provider._lease is None
        assert provider.containers == []
        provider.close()
    finally:
        provider.close()


def test_close_still_reports_an_unavailable_docker_daemon(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path}/missing.sock")
    provider = DockerExecProvider()
    provider.containers.append("mudgym-cleanup-test")

    with pytest.raises(subprocess.CalledProcessError) as error:
        provider.close()

    assert b"No such container:" not in error.value.stderr


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
