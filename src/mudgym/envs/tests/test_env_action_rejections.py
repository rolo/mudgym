from contextlib import closing

from mudgym.connections.recording import RecordingConnection, ReplayConnection
from mudgym.envs.factory import make_env
from mudgym.envs.tests.assertions import assert_observations_equal
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
    return raw_bytes, False, False, {"rejected": True}


def test_rejected_action_keeps_structured_observation(scripted_env_factory):
    """Observation commands sent separately still report state after an action rejection."""
    connection = ScriptedConnection(responses={ACTION: rejected_response()})
    env = scripted_env_factory(
        observation="parsed",
        connection=connection,
    )
    env.reset()

    obs, reward, terminated, truncated, info = env.step(ACTION)

    assert connection.sent_lines[-1] == [ACTION, OBSERVATION_COMMAND_LINE]
    assert info["action_rejected"] is True
    assert "don't know the word" in obs["text"].lower()
    assert obs["room_name"] == "dally lane"
    assert obs["points"] == 200
    assert obs["available_exits"].sum() == 8
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

    obs, _, terminated, truncated, info = env.step(action)

    assert "dances" in obs["text"]
    assert "don't know the word" in obs["text"].lower()
    assert obs["room_name"] == "dally lane"
    assert info["action_rejected"] is True
    assert terminated is False
    assert truncated is False


def test_rejected_env_step_records_and_replays_identically(tmp_path):
    capture_path = tmp_path / "rejected-step.jsonl"
    live_connection = ScriptedConnection(responses={ACTION: rejected_response()})
    with closing(
        make_env(
            observation="parsed",
            render_mode="ansi",
            connection=RecordingConnection(live_connection, capture_path),
        )
    ) as env:
        env.reset()
        expected = env.step(ACTION)
        expected_render = env.render()

    replay_connection = ReplayConnection(capture_path)
    with closing(make_env(observation="parsed", render_mode="ansi", connection=replay_connection)) as env:
        env.reset()
        actual = env.step(ACTION)
        actual_render = env.render()

    assert_observations_equal(actual[0], expected[0])
    assert actual[1:4] == expected[1:4]
    assert actual[4]["action_rejected"] is True
    assert actual_render == expected_render
    replay_connection.assert_exhausted()
