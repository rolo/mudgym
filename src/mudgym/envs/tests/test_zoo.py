from typing import Any

import gymnasium as gym
import pytest
from pettingzoo.test import parallel_api_test

from mudgym import make_parallel_env
from mudgym.envs.zoo import MudParallelEnv
from tests.scripted import NoOpProvider, ScriptedProvider, scripted_response


def make_test_parallel_env(envs, *, provider=None, render_mode=None):
    if provider is None:
        provider = NoOpProvider()
    return MudParallelEnv(envs, provider=provider, render_mode=render_mode)


def test_parallel_env_requires_a_provider():
    with pytest.raises(TypeError, match="provider"):
        MudParallelEnv({"player_0": TrackingEnv("player_0")})


@pytest.mark.parametrize(
    ("render_mode", "expected_child_render_mode"),
    [(None, None), ("ansi", "ansi"), ("human", "ansi")],
)
def test_parallel_factory_derives_child_render_mode(render_mode, expected_child_render_mode):
    env = make_parallel_env(agents=1, provider=ScriptedProvider(), render_mode=render_mode)
    try:
        assert env.render_mode == render_mode
        assert env.envs["player_0"].render_mode == expected_child_render_mode
    finally:
        env.close()


def test_parallel_infos_carry_render_bytes_for_each_agent():
    env = make_parallel_env(agents=2, provider=ScriptedProvider())
    try:
        _, infos = env.reset()

        assert set(infos) == {"player_0", "player_1"}
        assert all(isinstance(info["render_bytes"], bytes) for info in infos.values())
    finally:
        env.close()


class TrackingEnv:
    observation_space = gym.spaces.Dict({"value": gym.spaces.Discrete(10)})
    action_space = gym.spaces.Discrete(3)

    def __init__(
        self,
        name: str,
        events: list[tuple[str, str, Any]] | None = None,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ):
        self.name = name
        self.events = events if events is not None else []
        self.terminated = terminated
        self.truncated = truncated
        self.reset_calls: list[tuple[int | None, dict | None]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.reset_calls.append((seed, options))
        return {"value": 0}, {"seed": seed}

    def act(self, action: Any) -> None:
        self.events.append(("act", self.name, action))

    def observe(self):
        self.events.append(("observe", self.name, None))
        return {"value": 0}, 0.0, self.terminated, self.truncated, {}

    def render(self) -> str | None:
        return None

    def close(self) -> None:
        pass


def test_parallel_spaces_reference_each_child_space(scripted_env_factory):
    child_0 = scripted_env_factory()
    child_1 = scripted_env_factory()
    env = make_test_parallel_env({"player_0": child_0, "player_1": child_1})

    try:
        assert env.observation_space("player_0") is child_0.observation_space
        assert env.observation_space("player_1") is child_1.observation_space
        assert env.action_space("player_0") is child_0.action_space
        assert env.action_space("player_1") is child_1.action_space
    finally:
        env.close()


def test_render_returns_none_when_rendering_is_disabled(scripted_env_factory):
    child = scripted_env_factory(render_mode="ansi")
    child.reset()
    env = make_test_parallel_env({"player_0": child}, render_mode=None)

    try:
        assert env.render() is None
    finally:
        env.close()


def test_ansi_render_labels_child_output(scripted_env_factory):
    child = scripted_env_factory(render_mode="ansi")
    child.reset()
    env = make_test_parallel_env({"player_0": child}, render_mode="ansi")

    try:
        rendered = env.render()
        assert rendered.startswith("[player_0]\n")
        assert "Dally Lane" in rendered
    finally:
        env.close()


def test_human_render_prints_labeled_child_output(scripted_env_factory, capsys):
    child = scripted_env_factory(render_mode="ansi")
    child.reset()
    capsys.readouterr()
    env = make_test_parallel_env({"player_0": child}, render_mode="human")

    try:
        assert env.render() is None
        captured = capsys.readouterr()
        assert captured.out.startswith("[player_0]\n")
        assert "Dally Lane" in captured.out
    finally:
        env.close()


def test_parallel_step_acts_for_every_agent_before_any_observation():
    events: list[tuple[str, str, Any]] = []
    env = make_test_parallel_env(
        {
            "player_0": TrackingEnv("player_0", events),
            "player_1": TrackingEnv("player_1", events),
            "player_2": TrackingEnv("player_2", events),
        }
    )

    env.step({"player_0": 0, "player_1": 1, "player_2": 2})

    assert events == [
        ("act", "player_0", 0),
        ("act", "player_1", 1),
        ("act", "player_2", 2),
        ("observe", "player_0", None),
        ("observe", "player_1", None),
        ("observe", "player_2", None),
    ]


def test_parallel_step_uses_live_agent_keys():
    events: list[tuple[str, str, Any]] = []
    env = make_test_parallel_env(
        {
            "player_0": TrackingEnv("player_0", events),
            "player_1": TrackingEnv("player_1", events),
        }
    )

    with pytest.raises(KeyError, match="player_1"):
        env.step({"player_0": 0})
    events.clear()
    env.step({"player_0": 0, "player_1": 1, "player_2": 2})

    assert events == [
        ("act", "player_0", 0),
        ("act", "player_1", 1),
        ("observe", "player_0", None),
        ("observe", "player_1", None),
    ]


def test_same_step_actions_are_visible_to_every_shared_world_observation():
    actions: dict[str, str] = {}

    class SharedWorldEnv(TrackingEnv):
        def act(self, action: str) -> None:
            actions[self.name] = action

        def observe(self):
            return {"actions": dict(actions)}, 0.0, False, False, {}

    env = make_test_parallel_env(
        {
            "player_0": SharedWorldEnv("player_0"),
            "player_1": SharedWorldEnv("player_1"),
        }
    )

    observations, *_ = env.step({"player_0": "bow", "player_1": "howl"})

    expected_actions = {"player_0": "bow", "player_1": "howl"}
    assert observations["player_0"]["actions"] == expected_actions
    assert observations["player_1"]["actions"] == expected_actions


def test_step_removes_terminated_and_truncated_agents():
    env = make_test_parallel_env(
        {
            "player_0": TrackingEnv("player_0", terminated=True),
            "player_1": TrackingEnv("player_1", truncated=True),
            "player_2": TrackingEnv("player_2"),
        }
    )

    _, _, terminations, truncations, _ = env.step({"player_0": 0, "player_1": 1, "player_2": 2})

    assert terminations == {"player_0": True, "player_1": False, "player_2": False}
    assert truncations == {"player_0": False, "player_1": True, "player_2": False}
    assert env.agents == ["player_2"]


def test_parallel_reset_uses_distinct_deterministic_seeds_and_shared_options():
    children = {
        "player_0": TrackingEnv("player_0"),
        "player_1": TrackingEnv("player_1"),
        "player_2": TrackingEnv("player_2"),
    }
    env = make_test_parallel_env(children)
    options = {"nested": {"value": 1}}

    env.reset(seed=17, options=options)

    assert [children[agent].reset_calls for agent in env.possible_agents] == [
        [(17, options)],
        [(18, options)],
        [(19, options)],
    ]
    assert all(child.reset_calls[0][1] is options for child in children.values())


def test_single_agent_reset_always_finishes_with_an_authoritative_observation():
    events: list[tuple[str, str, Any]] = []
    env = make_test_parallel_env({"player_0": TrackingEnv("player_0", events)})

    env.reset()

    assert events == [("observe", "player_0", None)]


def test_parallel_reset_resets_provider_before_children():
    events = []

    class ResetTrackingEnv(TrackingEnv):
        def reset(self, *, seed: int | None = None, options: dict | None = None):
            events.append(("child", self.name, seed))
            return super().reset(seed=seed, options=options)

    class ResetTrackingProvider:
        def reset(self, *, seed=None):
            events.append(("provider", seed))

        def close(self):
            pass

    env = make_test_parallel_env(
        {
            "player_0": ResetTrackingEnv("player_0"),
            "player_1": ResetTrackingEnv("player_1"),
        },
        provider=ResetTrackingProvider(),
    )

    env.reset(seed=17)

    assert events[:3] == [
        ("provider", 17),
        ("child", "player_0", 17),
        ("child", "player_1", 18),
    ]


def test_parallel_reset_refreshes_early_observations_after_every_player_enters():
    present_players: set[str] = set()

    class SharedResetEnv(TrackingEnv):
        def reset(self, *, seed: int | None = None, options: dict | None = None):
            present_players.add(self.name)
            return {"player_count": len(present_players)}, {}

        def act(self, action: str) -> None:
            raise AssertionError(f"reset issued a player action: {action!r}")

        def observe(self):
            return {"player_count": len(present_players)}, 0.0, False, False, {}

    env = make_test_parallel_env(
        {
            "player_0": SharedResetEnv("player_0"),
            "player_1": SharedResetEnv("player_1"),
        }
    )

    observations, _ = env.reset()

    assert observations == {
        "player_0": {"player_count": 2},
        "player_1": {"player_count": 2},
    }


def test_terminated_player_does_not_desynchronise_survivor(scripted_env_factory):
    response = scripted_response(["quit", "sql,fes,fex,fei"]), True, False, {}
    env = make_test_parallel_env(
        {
            "player_0": scripted_env_factory(responses={"quit": response}),
            "player_1": scripted_env_factory(),
        }
    )

    _, _, terminations, truncations, infos = env.step({"player_0": "quit", "player_1": "look"})

    assert terminations == {"player_0": True, "player_1": False}
    assert truncations == {"player_0": False, "player_1": False}
    assert env.agents == ["player_1"]


def test_parallel_api_contract(scripted_env_factory):
    env = make_test_parallel_env(
        {
            "player_0": scripted_env_factory(),
            "player_1": scripted_env_factory(),
        }
    )

    parallel_api_test(env, num_cycles=10)


def test_live_shared_world_reset_and_step_observations_are_coherent():
    env = make_parallel_env(agents=2, render_mode="ansi")
    try:
        observations, infos = env.reset()
        assert all(observation["room_name"] for observation in observations.values())
        assert all(info["step"] == 0 for info in infos.values())

        observations, *_ = env.step({"player_0": "mgtransport ff0 me,yodel", "player_1": "mgtransport ff0 me,howl"})

        assert all(len(observation["players"]) == 1 for observation in observations.values())
        assert "howl" in observations["player_0"]["text"].lower()
    finally:
        env.close()
