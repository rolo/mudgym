import json

from mudgym.connections.recording import RecordingConnection
from mudgym.connections.wasm import create_connection
from mudgym.envs.fields import FEInventoryField
from mudgym.featurizers.responses import echo_pattern


def test_live_reset_and_step_keep_complete_command_echoes(live_env_factory, tmp_path, wasm_runtime):
    capture_path = tmp_path / "echo.session.jsonl"
    recording = RecordingConnection(create_connection(runtime=wasm_runtime), capture_path)
    env = live_env_factory(connection=recording, field_parsers=[FEInventoryField])

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
