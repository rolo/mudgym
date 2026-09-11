import numpy as np
import pytest

from mudgym.envs.fields import FEScoreField
from mudgym.envs.fields.rawbytes import DEFAULT_MAX_BYTES

CAPTURED_RESET_SWEEP = (
    b"\r\n+- Database 2 has finished initialising -+\r\n"
    b"\r\n+- Database 5 has finished initialising -+\r\n"
    b"\r\n+- Database 1 has finished initialising -+\r\n"
    b"fes\r\n"
    b"\x1b[0;37;40m\x1b[1;32;40m57\x1b[0;37;40m \x1b[1;32;40m57\x1b[0;37;40m 65 65 58 58 0 57 0200 N N N N 105 R\r\n"
    b"\x1b[1;37;40m\x1b[0;34;40m\x1b[1;34;40m*\x1b[0;34;40m\x1b[1;37;40m"
)


def test_database_reset_broadcasts_stay_out_of_text(scripted_env_factory):
    env = scripted_env_factory(observation="text")
    env.reset()
    obs, render_bytes, _ = env.unwrapped.bytes_to_observation(
        CAPTURED_RESET_SWEEP, sent_lines=["fes"], response_complete=True
    )
    assert "initialising" not in obs["text"]
    assert b"initialising" not in render_bytes
    assert obs["points"] == 200


def test_env_bytes_reset_returns_raw_bytes(scripted_env_factory):
    env = scripted_env_factory(observation="bytes")
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert set(obs) == {"text", "raw_bytes", "points"}
    assert obs["points"] == 200
    assert isinstance(obs["raw_bytes"], np.ndarray)

    bytes_length = len(info["raw_bytes"])
    assert bytes_length > 0
    assert obs["raw_bytes"].any()
    np.testing.assert_array_equal(
        obs["raw_bytes"][:bytes_length],
        np.frombuffer(info["raw_bytes"], dtype=obs["raw_bytes"].dtype),
    )
    # episode setup ends with the tearoom exit, so neither its command nor narration belongs to
    # the first observation
    assert b"move north" not in info["raw_bytes"]
    assert isinstance(info["render_bytes"], bytes)
    assert info["render_bytes"] == env.unwrapped.last_render_bytes
    assert b"move north" not in info["render_bytes"]
    # check that we stripped out the tearoom exit narration
    assert b"Elizabethan tearoom" not in info["raw_bytes"]
    assert env.observation_space["raw_bytes"].shape == obs["raw_bytes"].shape


@pytest.mark.parametrize("connection", ["docker_run", "docker_exec"])
def test_live_bytes_reset_step_and_reset(live_env_factory, connection):
    env = live_env_factory(observation="bytes", connection=connection)

    for _ in range(2):
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)

        observation, _, terminated, truncated, info = env.step("look")

        assert (terminated, truncated) == (False, False)
        assert set(observation) == {"text", "raw_bytes", "points"}
        assert env.observation_space.contains(observation)
        assert observation["raw_bytes"].shape == (DEFAULT_MAX_BYTES,)
        raw_bytes = info["raw_bytes"]
        assert observation["raw_bytes"][: len(raw_bytes)].tobytes() == raw_bytes
        assert not observation["raw_bytes"][len(raw_bytes) :].any()
        assert FEScoreField.end_of_turn_marker.search(raw_bytes)
