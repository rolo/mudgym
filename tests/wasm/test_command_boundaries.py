"""WASM completion comes from engine calls without marker-only probes."""

from contextlib import closing

import pytest

from mudgym import make_env
from mudgym.connections.wasm import WasmProvider
from mudgym.envs.fields import FEInventoryField, SuperQuickLookField


def test_non_ascii_commands_are_rejected_without_losing_pending_output(wasm_runtime):
    with closing(
        WasmProvider(
            runtime=wasm_runtime,
            worlds=1,
            seed=123,
            personas=(("Aaron", "male"), ("Abbie", "male")),
        )
    ) as provider:
        player, peer = provider.create_connections(2)
        player.reset()
        peer.reset()
        peer.send_line('tell "pending message" to Aaron')
        peer.read_response()

        with pytest.raises(ValueError, match="outside ASCII"):
            player.send_line("say café")

        player.send_line("look")
        raw, terminated, incomplete, info = player.read_response()
        assert b"pending message" in raw
        assert b"Elizabethan tearoom" in raw
        assert not terminated and not incomplete
        assert info["sent_lines"] == ["look"]
        assert player.read_response()[0] == b""


def test_wasm_accepts_an_observation_field_without_an_end_marker(wasm_runtime):
    env = make_env(connection="wasm", connection_kwargs={"runtime": wasm_runtime}, field_parsers=[SuperQuickLookField])
    try:
        obs, _ = env.reset(seed=123)
        room_name = obs["room_name"]
        obs, _, terminated, truncated, info = env.step("look")
        assert not terminated and not truncated
        assert obs["room_name"] == room_name
        assert info["transport"]["sent_lines"] == ["look", "sql"]
    finally:
        env.close()


@pytest.mark.parametrize(
    "observation_options",
    [
        {"observation": "text"},
        {"field_parsers": (FEInventoryField,)},
    ],
    ids=["text", "inventory-only"],
)
def test_points_track_combat_gains_flee_losses_and_terminal_score_without_fes(wasm_runtime, observation_options):
    provider = WasmProvider(
        runtime=wasm_runtime,
        worlds=1,
        seed=123,
        personas=(("Aaron", "male"), ("Abbie", "male")),
    )
    player, opponent = provider.create_connections(2)
    env = make_env(connection=player, **observation_options)

    def step(command):
        result = env.step(command)
        assert env.observation_space.contains(result[0])
        expected_lines = [command]
        if not result[2] and env.session.observation_line:
            expected_lines.append(env.session.observation_line)
        assert result[4]["transport"]["sent_lines"] == expected_lines
        assert b"fes\r\n" not in result[4]["raw_bytes"]
        return result

    try:
        obs, _ = env.reset()
        start_points = int(obs["points"])
        assert start_points == 200
        opponent.reset()
        opponent.send_line("north")
        opponent.read_response()
        for command in ("west", "kill Abbie"):
            obs, reward, terminated, truncated, _ = step(command)
            assert obs["points"] == 200 and reward == 0
            assert not terminated and not truncated

        obs, reward, terminated, truncated, info = step("flee east")
        assert b"Persona saved on -51 = \x1b[0;31;40m149" in info["raw_bytes"]
        assert obs["points"] == 149 and reward == -51
        assert not terminated and not truncated

        for command in ("west", "kill Abbie"):
            obs, reward, terminated, truncated, _ = step(command)
            assert obs["points"] == 149 and reward == 0
            assert not terminated and not truncated
        for _ in range(40):
            obs, reward, terminated, truncated, info = step("look")
            assert not terminated and not truncated
            if b"You have killed Abbie" in info["raw_bytes"]:
                break
            assert obs["points"] == 149 and reward == 0
        else:
            pytest.fail("Seeded combat did not produce Abbie's defeat within 40 steps")
        assert b"Persona saved on +120 = \x1b[0;32;40m269" in info["raw_bytes"]
        assert obs["points"] == 269 and reward == 120

        obs, reward, terminated, truncated, info = step("jump")
        assert b"Persona saved on -12 = \x1b[0;31;40m257" in info["raw_bytes"]
        assert b"Overall, you scored 257 points this game." in info["raw_bytes"]
        assert obs["points"] == 257 and reward == -12
        assert terminated and not truncated

        provider.reset(seed=123)
        obs, _ = env.reset()
        assert obs["points"] == start_points
        assert env.observation_space.contains(obs)
    finally:
        env.close()
        provider.close()
