import numpy as np

from mudgym.connections.recording import RecordingConnection, ReplayConnection
from mudgym.envs.factory import make_env
from tests.scripted import (
    OK_RESPONSE,
    PROMPT,
    ScriptedConnection,
    scripted_response,
)

ACTION = "xyzzyfrobnicate"
OBSERVATION_COMMAND_LINE = "sql,fes,fex,fei"
REJECTION_TEXT = b'I don\'t know the word "xyzzyfrobnicate".\r\n'


def rejected_response(action: str = ACTION, *, preceding_output: bytes = b""):
    raw_bytes = (
        action.encode("ascii")
        + b"\r\n"
        + preceding_output
        + REJECTION_TEXT
        + PROMPT
        + scripted_response([OBSERVATION_COMMAND_LINE])
    )
    return raw_bytes, False, False, {"rejected": True, "marker_arrived": True}


def assert_observations_equal(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(actual[key], value), key
        else:
            assert actual[key] == value, key


def test_rejected_action_keeps_structured_observation(scripted_env_factory):
    """Observation commands sent separately still report state after an action rejection."""
    connection = ScriptedConnection(responses={ACTION: rejected_response()})
    env = scripted_env_factory(
        observation="parsed",
        connection=connection,
    )
    env.reset()

    observation, reward, terminated, truncated, info = env.step(ACTION)

    assert connection.sent_lines[-1] == [ACTION, OBSERVATION_COMMAND_LINE]
    assert info["action_rejected"] is True
    assert "don't know the word" in observation["text"].lower()
    assert observation["room_name"] == "dally lane"
    assert observation["points"] == 200
    assert observation["available_exits"].sum() == 8
    assert reward == 0
    assert terminated is False
    assert truncated is False


def test_output_before_a_compound_player_command_rejection_survives_observation(scripted_env_factory):
    action = f"dance,{ACTION}"
    connection = ScriptedConnection(
        responses={action: rejected_response(action, preceding_output=OK_RESPONSE + PROMPT)}
    )
    env = scripted_env_factory(observation="parsed", connection=connection)
    env.reset()

    observation, _, terminated, truncated, info = env.step(action)

    assert "dances" in observation["text"]
    assert "don't know the word" in observation["text"].lower()
    assert observation["room_name"] == "dally lane"
    assert info["action_rejected"] is True
    assert terminated is False
    assert truncated is False


def test_rejected_environment_step_records_and_replays_identically(tmp_path):
    capture_path = tmp_path / "rejected-step.jsonl"
    live_connection = ScriptedConnection(responses={ACTION: rejected_response()})
    env = make_env(
        observation="parsed",
        render_mode="ansi",
        connection=RecordingConnection(live_connection, capture_path),
    )
    env.reset()
    expected = env.step(ACTION)
    expected_render = env.render()
    env.close()

    replay_connection = ReplayConnection(capture_path)
    replay_env = make_env(observation="parsed", render_mode="ansi", connection=replay_connection)
    replay_env.reset()
    actual = replay_env.step(ACTION)
    actual_render = replay_env.render()
    replay_env.close()

    assert_observations_equal(actual[0], expected[0])
    assert actual[1:4] == expected[1:4]
    assert actual[4]["action_rejected"] is True
    assert actual_render == expected_render
    assert replay_connection.remaining_events() == 0
