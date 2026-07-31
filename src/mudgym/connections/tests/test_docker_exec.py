import subprocess

from mudgym.connections.docker_exec import DockerExecConnection


def test_default_container_name_can_be_scoped_by_environment(monkeypatch):
    container_name = "mud2-boot-run-86-attempt-1-python-3.14"
    discovered_names = []
    monkeypatch.setenv("MUDGYM_DOCKER_EXEC_CONTAINER_NAME", container_name)

    def find_running_container(command, *, text):
        discovered_names.append(command[-1])
        return "container-id\n"

    monkeypatch.setattr(subprocess, "check_output", find_running_container)

    connection = DockerExecConnection(start_if_missing=False)

    assert connection.container_name == container_name
    assert discovered_names == [f"name={container_name}"]
