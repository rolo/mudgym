import numpy as np

from mudgym.connections.recording import RecordingConnection, ReplayConnection
from mudgym.envs.factory import make_env
from tests.scripted import (
    AUTO_COMMAND_RESPONSES,
    OK_RESPONSE,
    PROMPT,
    ScriptedConnection,
    scripted_response,
)

ACTION = "xyzzyfrobnicate"
AUTO_COMMAND_LINE = "sql,fes,fex,fei"
REJECTION_TEXT = b'I don\'t know the word "xyzzyfrobnicate".\r\n'


def rejected_response(action: str = ACTION, *, preceding_output: bytes = b""):
    wire_line = f"{action},{AUTO_COMMAND_LINE}"
    raw_bytes = wire_line.encode("ascii") + b"\r\n" + preceding_output + REJECTION_TEXT + PROMPT
    return raw_bytes, False, False, {"rejected": True, "marker_arrived": False}


def assert_observations_equal(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(actual[key], value), key
        else:
            assert actual[key] == value, key


def test_rejected_action_refreshes_structured_observation(scripted_env_factory):
    """A parser rejection must not discard the state reported by the follow-up auto commands."""
    connection = ScriptedConnection(responses={ACTION: rejected_response()})
    env = scripted_env_factory(
        observation="parsed",
        connection=connection,
    )
    env.reset()

    observation, reward, terminated, truncated, info = env.step(ACTION)

    assert connection.sent_lines[-2:] == [
        [f"{ACTION},{AUTO_COMMAND_LINE}"],
        [AUTO_COMMAND_LINE],
    ]
    assert info["action_rejected"] is True
    assert info["wire_lines"][-2:] == [
        f"{ACTION},{AUTO_COMMAND_LINE}",
        AUTO_COMMAND_LINE,
    ]
    assert "don't know the word" in observation["text"].lower()
    assert observation["room_name"] == "dally lane"
    assert observation["points"] == 200
    assert observation["available_exits"].sum() == 8
    assert len(info["auto_command_chunks"]) == 4
    assert reward == 0
    assert terminated is False
    assert truncated is False


def test_output_before_a_comma_chained_rejection_survives_the_refresh(scripted_env_factory):
    action = f"dance,{ACTION}"
    connection = ScriptedConnection(
        responses={"dance": rejected_response(action, preceding_output=OK_RESPONSE + PROMPT)}
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


def test_rejected_auto_refresh_truncates_the_episode(scripted_env_factory):
    rejected_refresh = (
        AUTO_COMMAND_LINE.encode("ascii") + b"\r\n" + REJECTION_TEXT + PROMPT,
        False,
        False,
        {"rejected": True, "marker_arrived": False},
    )
    env = scripted_env_factory(
        observation="parsed",
        responses={ACTION: rejected_response(), "sql": rejected_refresh},
    )
    env.reset()

    _, _, terminated, truncated, info = env.step(ACTION)

    assert info["action_rejected"] is True
    assert terminated is False
    assert truncated is True


def test_rejected_split_speech_action_reports_rejection_without_repeating_completed_autos(scripted_env_factory):
    action = "say hello"
    raw_bytes = action.encode("ascii") + b"\r\n" + REJECTION_TEXT + PROMPT + scripted_response(AUTO_COMMAND_LINE)
    response = raw_bytes, False, False, {"rejected": True, "marker_arrived": True}
    connection = ScriptedConnection(responses={action: response})
    env = scripted_env_factory(observation="parsed", connection=connection)
    env.reset()

    observation, _, terminated, truncated, info = env.step(action)

    assert connection.sent_lines[-1] == [action, AUTO_COMMAND_LINE]
    assert info["wire_lines"] == [action, AUTO_COMMAND_LINE]
    assert info["action_rejected"] is True
    assert observation["room_name"] == "dally lane"
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
    assert actual[4]["render_bytes"] == expected[4]["render_bytes"]
    assert actual_render == expected_render
    assert replay_connection.remaining_events() == 0


def test_scripted_auto_only_line_starts_with_the_first_auto_response():
    raw_bytes = scripted_response(AUTO_COMMAND_LINE)

    first_response = raw_bytes.split(PROMPT, 1)[0]
    assert AUTO_COMMAND_RESPONSES["sql"] in first_response
    assert OK_RESPONSE not in first_response
