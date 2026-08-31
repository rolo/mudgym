import subprocess
from contextlib import contextmanager
from uuid import uuid4

import pytest

from mudgym import make_env, make_parallel_env
from mudgym.connections.config import DOCKER_IMAGE
from mudgym.connections.docker_exec import DockerExecConnection
from mudgym.connections.docker_readiness import DockerWorldReadinessError, wait_for_docker_worlds_ready
from mudgym.connections.provider import DockerExecProvider


@contextmanager
def running_test_container(command: str):
    container_name = f"mudgym-readiness-probe-{uuid4().hex}"
    container_id = subprocess.check_output(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            DOCKER_IMAGE,
            "/bin/sh",
            "-c",
            command,
        ],
        text=True,
    ).strip()
    try:
        yield container_id
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container_id],
            check=False,
            capture_output=True,
        )


@pytest.fixture(scope="module")
def delayed_boot_image():
    image_name = f"mudgym-readiness-test:{uuid4().hex}"
    dockerfile = """
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN mv /app/bin/boot /app/bin/boot-real \
    && printf '%s\\n' '#!/bin/sh' 'sleep 1' 'exec /app/bin/boot-real "$@"' > /app/bin/boot \
    && chmod +x /app/bin/boot \
    && mv /app/bin/mudlogin /app/bin/mudlogin-real \
    && printf '%s\\n' \
        '#!/bin/sh' \
        'grep -qF "[Reset took" /home/mudgod/LOGS/reset.0/mmud || {' \
        '    echo "mudlogin started before the world was ready" >&2' \
        '    exit 86' \
        '}' \
        'exec /app/bin/mudlogin-real "$@"' \
        > /app/bin/mudlogin \
    && chmod +x /app/bin/mudlogin
"""
    subprocess.run(
        [
            "docker",
            "build",
            "--quiet",
            "--build-arg",
            f"BASE_IMAGE={DOCKER_IMAGE}",
            "--tag",
            image_name,
            "-",
        ],
        input=dockerfile,
        text=True,
        check=True,
        capture_output=True,
    )
    try:
        yield image_name
    finally:
        subprocess.run(
            ["docker", "image", "rm", image_name],
            check=True,
            capture_output=True,
        )


def test_provider_waits_for_world_readiness_before_login(delayed_boot_image):
    environment = make_parallel_env(
        agents=1,
        provider=DockerExecProvider(worlds=1, image=delayed_boot_image),
    )
    try:
        environment.reset(seed=1234)
    finally:
        environment.close()


def test_standalone_connection_waits_for_world_readiness_before_login(delayed_boot_image):
    connection = DockerExecConnection(
        container_name=f"mudgym-readiness-{uuid4().hex}",
        container_image=delayed_boot_image,
    )
    environment = make_env(connection=connection)
    try:
        environment.reset(seed=1234)
    finally:
        environment.close()


def test_readiness_waits_for_every_requested_slot():
    command = """
mkdir -p /home/mudgod/LOGS/reset.0 /home/mudgod/LOGS/reset.1
printf '%s\n' '[Reset took 1ms]' > /home/mudgod/LOGS/reset.0/mmud
sleep 1
printf '%s\n' '[Reset took 1ms]' > /home/mudgod/LOGS/reset.1/mmud
tail -f /dev/null
"""
    with running_test_container(command) as container_id:
        wait_for_docker_worlds_ready(container_id, 2)

        subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "grep",
                "-qF",
                "[Reset took",
                "/home/mudgod/LOGS/reset.1/mmud",
            ],
            check=True,
        )


def test_readiness_timeout_names_the_first_unready_slot():
    with (
        running_test_container("tail -f /dev/null") as container_id,
        pytest.raises(DockerWorldReadinessError, match="first unready slot: 0"),
    ):
        wait_for_docker_worlds_ready(container_id, 2, timeout_seconds=0.05)
