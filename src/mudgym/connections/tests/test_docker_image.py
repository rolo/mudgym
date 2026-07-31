import subprocess
from types import SimpleNamespace

import pytest

from mudgym.connections import docker_image
from mudgym.connections.docker_image import DockerSetupError, ensure_docker_image


@pytest.fixture(autouse=True)
def fresh_image_cache(monkeypatch):
    monkeypatch.setattr(docker_image, "_verified_images", set())


def fake_run_factory(calls, *, inspect_result, pull_result=None):
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "image":
            return inspect_result
        if command[1] == "pull":
            if pull_result is None:
                raise AssertionError("pull was not expected")
            return pull_result
        raise AssertionError(f"unexpected docker command: {command}")

    return fake_run


def completed(returncode, stderr=""):
    return SimpleNamespace(returncode=returncode, stderr=stderr)


def test_missing_docker_binary_names_the_fix(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: None)

    with pytest.raises(DockerSetupError, match="no `docker` executable"):
        ensure_docker_image("ghcr.io/rolo/mudgym")


def test_unreachable_daemon_is_distinguished_from_a_missing_image(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run_factory(
            calls,
            inspect_result=completed(1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"),
        ),
    )

    with pytest.raises(DockerSetupError, match="daemon is not reachable"):
        ensure_docker_image("ghcr.io/rolo/mudgym")


def test_cached_image_short_circuits_without_pulling(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(subprocess, "run", fake_run_factory(calls, inspect_result=completed(0)))

    ensure_docker_image("ghcr.io/rolo/mudgym")

    assert [command[1] for command in calls] == ["image"]


def test_verified_images_are_only_inspected_once_per_process(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(subprocess, "run", fake_run_factory(calls, inspect_result=completed(0)))

    ensure_docker_image("ghcr.io/rolo/mudgym")
    ensure_docker_image("ghcr.io/rolo/mudgym")

    assert len(calls) == 1


def test_absent_image_is_pulled(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run_factory(
            calls,
            inspect_result=completed(1, "Error: No such image: ghcr.io/rolo/mudgym"),
            pull_result=completed(0),
        ),
    )

    ensure_docker_image("ghcr.io/rolo/mudgym")

    assert [command[1] for command in calls] == ["image", "pull"]


def test_daemon_worded_missing_image_error_still_pulls(monkeypatch):
    """The real daemon reports a missing image as an 'Error response from daemon'."""
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run_factory(
            calls,
            inspect_result=completed(1, "Error response from daemon: No such image: ghcr.io/rolo/mudgym:v9"),
            pull_result=completed(0),
        ),
    )

    ensure_docker_image("ghcr.io/rolo/mudgym:v9")

    assert [command[1] for command in calls] == ["image", "pull"]


def test_denied_pull_explains_registry_access(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run_factory(
            calls,
            inspect_result=completed(1, "Error: No such image"),
            pull_result=completed(1, "Error response from daemon: denied"),
        ),
    )

    with pytest.raises(DockerSetupError, match="refused access"):
        ensure_docker_image("ghcr.io/rolo/mudgym")


def test_unknown_tag_pull_names_the_image(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run_factory(
            calls,
            inspect_result=completed(1, "Error: No such image"),
            pull_result=completed(1, "manifest unknown"),
        ),
    )

    with pytest.raises(DockerSetupError, match="does not exist on the registry"):
        ensure_docker_image("ghcr.io/rolo/mudgym:nope")


def test_failed_image_is_not_cached_as_verified(monkeypatch):
    monkeypatch.setattr(docker_image.shutil, "which", lambda name: "/usr/bin/docker")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run_factory(
            calls,
            inspect_result=completed(1, "Error: No such image"),
            pull_result=completed(1, "manifest unknown"),
        ),
    )

    with pytest.raises(DockerSetupError):
        ensure_docker_image("ghcr.io/rolo/mudgym:nope")
    with pytest.raises(DockerSetupError):
        ensure_docker_image("ghcr.io/rolo/mudgym:nope")

    assert len(calls) == 4
