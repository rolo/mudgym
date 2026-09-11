import json
import subprocess
import uuid

import pytest

from mudgym.connections.config import DOCKER_IMAGE
from mudgym.connections.docker_exec import DockerExecConnection
from mudgym.connections.docker_image import ensure_docker_image
from mudgym.connections.docker_run import DockerRunConnection
from mudgym.connections.recording import RecordingConnection
from mudgym.envs.fields import FEInventoryField
from mudgym.featurizers.responses import echo_pattern


@pytest.mark.parametrize("connection_class", [DockerRunConnection, DockerExecConnection])
@pytest.mark.parametrize("send_delay", [0, 0.005])
def test_live_reset_and_step_keep_complete_command_echoes(
    live_env_factory, tmp_path, record_property, connection_class, send_delay
):
    ensure_docker_image(DOCKER_IMAGE)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"], text=True
    ).strip()

    class DelayedRecordingConnection(RecordingConnection):
        def reset(self, *, seed=None):
            super().reset(seed=seed)
            # Only vary game sends. Login retains the connection's normal delay.
            self.connection.sm.child.delaybeforesend = send_delay

    connection = connection_class(container_name=f"mudgym-echo-{uuid.uuid4().hex}")
    capture_path = tmp_path / "echo.session.jsonl"
    recording = DelayedRecordingConnection(
        connection, capture_path, metadata={"image_id": image_id, "send_delay": send_delay}
    )
    env = live_env_factory(connection=recording, field_parsers=[FEInventoryField])
    record_property("image_id", image_id)
    record_property("send_delay", send_delay)

    for _ in range(3):
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
        assert "fei" not in observation["portables"]
        assert "fei" not in observation["inventory"]

        observation, _, terminated, truncated, _ = env.step("look")
        assert not terminated and not truncated
        assert env.observation_space.contains(observation)
        assert "fei" not in observation["portables"]
        assert "fei" not in observation["inventory"]

    terminal_settings = subprocess.check_output(
        ["docker", "exec", connection.container_name, "stty", "-a", "-F", "/dev/pts/0"], text=True
    )
    record_property("terminal_settings", terminal_settings)
    assert "-echo" in terminal_settings.split()

    # Count echoes in received bytes, independently of the send log. Separate lines may be
    # echoed on either side of the action response, but each must arrive intact exactly once.
    pending_lines = []
    for call in map(json.loads, capture_path.read_text().splitlines()[1:]):
        if call["call"] == "send_line":
            pending_lines.append(call["line"])
        elif call["call"] == "read_response":
            raw_bytes = call["raw_text"].encode("latin-1")
            assert call["marker_arrived"] and not call["incomplete"] and not call["terminated"]
            for line in pending_lines:
                assert len(list(echo_pattern(line).finditer(raw_bytes))) == 1, (line, raw_bytes)
            pending_lines.clear()
    assert not pending_lines
