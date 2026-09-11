"""Real session retirement and replay regressions."""

import pytest

from mudgym import make_env
from mudgym.connections.recording import RecordingConnection, ReplayConnection
from mudgym.connections.wasm import WasmtimeProvider


def test_failed_reset_requires_a_clean_reset_before_normal_use(wasm_runtime):
    with make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}, observation="text") as environment:
        environment.reset(seed=123)
        environment.act("look")

        with pytest.raises(RuntimeError, match="unread responses"):
            environment.reset(seed=456)
        with pytest.raises(RuntimeError, match="reset"):
            environment.act("north")
        with pytest.raises(RuntimeError, match="reset"):
            environment.observe()

        environment.reset(seed=456)
        _, _, terminated, truncated, _ = environment.step("look")
        assert not terminated and not truncated


@pytest.mark.parametrize("phase", ["action", "observation"])
@pytest.mark.parametrize("preset", ["parsed", "text", "bytes"])
def test_pending_combat_death_survives_the_next_command_and_reset(wasm_runtime, phase, preset, tmp_path):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1, seed=123)
    victim, survivor = provider.create_connections(2)
    capture = tmp_path / "departure.jsonl"
    environment = make_env(connection=RecordingConnection(victim, capture), observation=preset)
    try:
        environment.reset()
        survivor.reset()
        survivor.send_line("north")
        survivor.read_response(None)
        victim.send_line("west")
        assert b"Beaten track near cliff" in victim.read_response(None)[0]
        victim.send_line("kill Alba")
        assert b"You attack Alba" in victim.read_response(None)[0]
        if phase == "observation":
            environment.unwrapped.act("look")

        world = victim._session.world
        old_session = victim._session
        for _ in range(40):
            provider.advance_worlds(1)
            if b"You have killed Ada" in survivor._session.receive():
                break
        else:
            pytest.fail("Seeded combat did not kill Ada within 40 ticks")
        assert world.is_alive() and not old_session.departed

        result = environment.step("look") if phase == "action" else environment.unwrapped.observe()
        observation, reward, terminated, truncated, info = result
        final_result = result
        final_message = b"You have been killed by Alba"
        assert terminated and not truncated
        assert reward == -200 and observation["points"] == 0
        assert info["raw_bytes"].count(final_message) == 1
        assert b"sql,fes,fex,fei\r\n" not in info["raw_bytes"]
        if phase == "action":
            assert b"look\r\n" not in info["raw_bytes"]
        assert info["transport"]["sent_lines"] == (["look"] if phase == "observation" else [])
        assert "look" not in observation["text"].splitlines()
        assert b"look" not in info["render_bytes"].splitlines()
        assert old_session.departed and world.is_alive()
        # Unrelated uses of the retired handle must still fail loudly.
        with pytest.raises(RuntimeError, match=r"session-gone code=-7"):
            old_session.send("look", 5000)

        observation, reset_info = environment.reset()
        assert observation["points"] == 200
        assert final_message not in reset_info["raw_bytes"]
        assert victim._session.world is world
        assert victim._session.generation > old_session.generation
        _, _, terminated, truncated, info = environment.step("look")
        assert not terminated and not truncated
        assert final_message not in info["raw_bytes"]
    finally:
        environment.close()
        provider.close()

    replay = ReplayConnection(capture)
    replay_environment = make_env(connection=replay, observation=preset)
    try:
        replay_environment.reset()
        if phase == "observation":
            replay_environment.unwrapped.act("look")
            replay_result = replay_environment.unwrapped.observe()
        else:
            replay_result = replay_environment.step("look")
        replay_observation, replay_reward, replay_terminated, replay_truncated, replay_info = replay_result
        assert replay_observation["text"] == final_result[0]["text"]
        assert (replay_reward, replay_terminated, replay_truncated) == final_result[1:4]
        for key in ("raw_bytes", "render_bytes"):
            assert replay_info[key] == final_result[4][key]
        assert replay_info["transport"]["sent_lines"] == final_result[4]["transport"]["sent_lines"]
        replay_environment.reset()
        replay_environment.step("look")
        replay.assert_exhausted()
    finally:
        replay_environment.close()
