from contextlib import closing

import pytest
from pettingzoo.test import parallel_api_test

from mudgym import make_parallel_env
from mudgym.connections.wasm import WasmtimeProvider
from mudgym.envs.specs import ACTION_MAX_LENGTH
from tests.scripted import ScriptedProvider


@pytest.mark.parametrize(
    ("render_mode", "expected_child_render_mode"),
    [(None, None), ("ansi", "ansi"), ("human", "ansi")],
)
def test_parallel_construction_forwards_render_mode_and_spaces(render_mode, expected_child_render_mode):
    with closing(make_parallel_env(2, provider=ScriptedProvider(), render_mode=render_mode)) as env:
        assert env.render_mode == render_mode
        for agent, child in env.envs.items():
            assert child.render_mode == expected_child_render_mode
            assert env.observation_space(agent) is child.observation_space
            assert env.action_space(agent) is child.action_space
        if render_mode is None:
            assert env.render() is None


def test_ansi_render_labels_child_output():
    with closing(make_parallel_env(2, provider=ScriptedProvider(), render_mode="ansi")) as env:
        env.reset(seed=10)
        rendered = env.render()
        assert rendered.startswith("[player_0]\n")
        assert "[player_1]\n" in rendered
        assert rendered.count("Dally Lane") == 2


def test_human_render_prints_labeled_child_output(capsys):
    with closing(make_parallel_env(2, provider=ScriptedProvider(), render_mode="human")) as env:
        env.reset(seed=10)
        capsys.readouterr()
        assert env.render() is None
        captured = capsys.readouterr()
        assert captured.out.startswith("[player_0]\n")
        assert "[player_1]\n" in captured.out
        assert captured.out.count("Dally Lane") == 2


@pytest.mark.parametrize("preset", ["text", "parsed"])
def test_same_step_messages_reach_every_shared_world_observation(wasm_runtime, preset):
    with closing(
        make_parallel_env(
            2,
            provider=WasmtimeProvider(
                runtime=wasm_runtime,
                worlds=1,
                personas=(("Aaron", "male"), ("Abbie", "male")),
            ),
            observation=preset,
        )
    ) as env:
        env.reset(seed=10)

        obs, _, terminates, truncates, _ = env.step(
            {"player_0": 'tell "alpha greeting" to abbie', "player_1": 'tell "beta greeting" to aaron'}
        )

        assert 'Abbie the protector tells you "beta greeting"' in obs["player_0"]["text"]
        assert 'Aaron the protector tells you "alpha greeting"' in obs["player_1"]["text"]
        assert not any(terminates.values())
        assert not any(truncates.values())


def test_parallel_step_requires_an_action_for_every_live_agent():
    with closing(make_parallel_env(2, provider=ScriptedProvider())) as env:
        env.reset(seed=10)
        with pytest.raises(KeyError, match="player_1"):
            env.step({"player_0": "look"})

        obs, _, terminates, truncates, _ = env.step({"player_0": "look", "player_1": "look"})

        assert set(obs) == {"player_0", "player_1"}
        assert not any(terminates.values())
        assert not any(truncates.values())


@pytest.mark.parametrize(
    "action",
    ["", "look\nlook", "x" * (ACTION_MAX_LENGTH + 1), None],
    ids=["empty", "line-break", "too-long", "non-string"],
)
def test_invalid_parallel_action_does_not_partially_step(action):
    ticks = []
    with closing(make_parallel_env(2, provider=ScriptedProvider(), world_ticker=lambda: ticks.append("tick"))) as env:
        env.reset()
        connections = [child.session.connection for child in env.envs.values()]
        sent_before = [list(connection.send_calls) for connection in connections]

        with pytest.raises(ValueError, match="Invalid action"):
            env.step({"player_0": "look", "player_1": action})

        assert ticks == []
        assert [connection.send_calls for connection in connections] == sent_before
        assert all(child.step_count == 0 for child in env.envs.values())
        assert all(child.session.pending_command is None for child in env.envs.values())
        assert all(not child.session.connection.pending_lines for child in env.envs.values())

        obs, _rewards, _terminates, _truncates, _infos = env.step(dict.fromkeys(env.agents, "look"))

        assert set(obs) == set(env.agents)
        assert ticks == ["tick"]
        assert all(child.step_count == 1 for child in env.envs.values())
        assert all(child.session.pending_command is None for child in env.envs.values())


def test_parallel_step_uses_only_live_agent_keys():
    with closing(make_parallel_env(2, provider=ScriptedProvider())) as env:
        env.reset()

        obs, rewards, terminates, truncates, infos = env.step(
            {"player_0": "look", "player_1": "dance", "player_2": "bow"}
        )

        for result in (obs, rewards, terminates, truncates, infos):
            assert set(result) == {"player_0", "player_1"}
        assert infos["player_0"]["transport"]["sent_lines"] == ["look", "sql,fes,fex,fei"]
        assert infos["player_1"]["transport"]["sent_lines"] == ["dance", "sql,fes,fex,fei"]
        assert not any(terminates.values()) and not any(truncates.values())


def test_step_removes_terminated_and_truncated_agents(wasm_runtime):
    with closing(make_parallel_env(3, provider=WasmtimeProvider(runtime=wasm_runtime, worlds=1))) as env:
        initial_obs, _ = env.reset(seed=10)

        # /t asks for terminal width, leaving an unanswered input request that truncates the step.
        _, _, terminates, truncates, infos = env.step({"player_0": "quit", "player_1": "/t", "player_2": "look"})

        assert terminates == {"player_0": True, "player_1": False, "player_2": False}
        assert truncates == {"player_0": False, "player_1": True, "player_2": False}
        assert b"New terminal width" in infos["player_1"]["raw_bytes"]
        assert env.agents == ["player_2"]
        obs, _, terminates, truncates, _ = env.step({"player_2": "look"})
        assert set(obs) == {"player_2"}
        assert initial_obs["player_2"]["room_name"] in obs["player_2"]["text"].lower()
        assert terminates == {"player_2": False}
        assert truncates == {"player_2": False}


def test_parallel_reset_propagates_distinct_seeds_to_children_and_action_spaces():
    with closing(make_parallel_env(3, provider=ScriptedProvider())) as env:
        env.reset(seed=17)
        assert [child.np_random_seed for child in env.envs.values()] == [17, 18, 19]
        actions = [child.action_space.sample() for child in env.envs.values()]

        env.reset(seed=17)
        assert [child.action_space.sample() for child in env.envs.values()] == actions


def test_parallel_reset_restarts_the_shared_world(wasm_runtime):
    provider = WasmtimeProvider(runtime=wasm_runtime, worlds=1)
    with closing(make_parallel_env(2, provider=provider)) as env:
        initial_obs, _ = env.reset(seed=17)
        env.step({"player_0": "look", "player_1": "look"})
        assert provider.advance_worlds(0) == {0: 1}

        obs, infos = env.reset(seed=17)

        assert provider.advance_worlds(0) == {0: 0}
        assert {agent: player_obs["text"] for agent, player_obs in obs.items()} == {
            agent: player_obs["text"] for agent, player_obs in initial_obs.items()
        }
        assert all(info["step"] == 0 for info in infos.values())


def test_parallel_reset_observations_include_every_player(wasm_runtime):
    provider = WasmtimeProvider(
        runtime=wasm_runtime,
        worlds=1,
        personas=(("Aaron", "male"), ("Abbie", "male")),
    )
    with closing(make_parallel_env(2, provider=provider)) as env:
        obs, infos = env.reset(seed=10)

        assert obs["player_0"]["room_name"] == obs["player_1"]["room_name"]
        assert obs["player_0"]["players"] == ("Abbie the protector",)
        assert obs["player_1"]["players"] == ("Aaron the protector",)
        for agent, player_obs in obs.items():
            assert player_obs["text"].count("Badly-paved road") == 1
            assert infos[agent]["render_bytes"].count(b"Badly-paved road") == 1
        assert all(info["step"] == 0 for info in infos.values())


def test_failed_provider_reset_after_all_agents_finish_blocks_the_step_clock(monkeypatch):
    provider = ScriptedProvider()
    ticker_calls = []
    failure = OSError("provider reset failed")

    def fail_reset(*, seed=None):
        raise failure

    with closing(make_parallel_env(2, provider=provider, world_ticker=lambda: ticker_calls.append("tick"))) as env:
        for connection in provider.connections:
            connection.responses["quit"] = (b"quit\r\nCheerio!\r\n", True, False, {})
        env.reset()
        env.step(dict.fromkeys(env.agents, "quit"))
        assert env.agents == []
        ticker_calls.clear()
        original_reset = provider.reset
        monkeypatch.setattr(provider, "reset", fail_reset)
        with pytest.raises(OSError) as raised:
            env.reset()
        assert raised.value is failure
        assert all(child.points is None for child in env.envs.values())
        with pytest.raises(RuntimeError, match="successful reset"):
            env.step({})
        assert ticker_calls == []

        monkeypatch.setattr(provider, "reset", original_reset)
        env.reset()
        _, _, terminates, truncates, _ = env.step(dict.fromkeys(env.agents, "look"))
        assert ticker_calls == ["tick"]
        assert not any(terminates.values()) and not any(truncates.values())


def test_parallel_api_contract(wasm_runtime):
    with closing(make_parallel_env(2, provider=WasmtimeProvider(runtime=wasm_runtime, worlds=1))) as env:
        parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize(
    ("phase", "command"),
    [
        ("preparation", "qs"),
        ("entry", "move north"),
        ("observation", "sql,fes,fex,fei"),
    ],
)
def test_coordinated_reset_stops_on_failure_and_can_retry(phase, command, capsys):
    provider = ScriptedProvider()
    failure = OSError(f"failed {phase}")
    with closing(make_parallel_env(2, provider=provider, render_mode="human")) as env:
        provider.connections[1].send_errors[command] = failure
        with pytest.raises(OSError) as raised:
            env.reset()
        assert raised.value is failure
        assert any(phase in note for note in failure.__notes__)
        assert capsys.readouterr().out == ""
        assert all(connection.invalidated and not connection.pending_lines for connection in provider.connections)
        sent = [line for connection in provider.connections for line in connection.send_calls]
        if phase == "preparation":
            assert "move north" not in sent
        if phase in {"preparation", "entry"}:
            assert "sql,fes,fex,fei" not in sent
        actions = {"player_0": "look", "player_1": "look"}
        with pytest.raises(RuntimeError, match="reset"):
            env.step(actions)
        assert capsys.readouterr().out == ""

        provider.connections[1].send_errors.clear()
        env.reset()
        _, _, terminates, truncates, _ = env.step(actions)
        assert not any(terminates.values())
        assert not any(truncates.values())
