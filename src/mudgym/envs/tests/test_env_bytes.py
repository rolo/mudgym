import numpy as np


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
    # the observation is the verbatim wire, echo included; forgery resistance lives in the
    # parsers' anchored patterns, not in editing the bytes
    assert b"move north" in info["raw_bytes"]
    assert isinstance(info["render_bytes"], bytes)
    assert info["render_bytes"] == env.unwrapped.last_render_bytes
    assert b"move north" not in info["render_bytes"]
    # check that we stripped out the tearoom exit narration
    assert b"Elizabethan tearoom" not in info["raw_bytes"]
    assert env.observation_space["raw_bytes"].shape == obs["raw_bytes"].shape
