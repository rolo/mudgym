import pytest

from mudgym.envs.zoo import MudParallelEnv
from tests.scripted import ScriptedConnection, scripted_response


def test_parallel_spaces_reference_each_child_space(scripted_env_factory):
    child_0 = scripted_env_factory()
    child_1 = scripted_env_factory()

    env = MudParallelEnv(
        {
            "player_0": child_0,
            "player_1": child_1,
        }
    )

    try:
        assert env.observation_space("player_0") is child_0.observation_space
        assert env.observation_space("player_1") is child_1.observation_space
        assert env.action_space("player_0") is child_0.action_space
        assert env.action_space("player_1") is child_1.action_space
    finally:
        env.close()


def test_parallel_env_rejects_heterogeneous_child_spaces(scripted_env_factory):
    child_0 = scripted_env_factory(observation="text")
    child_1 = scripted_env_factory(observation="parsed")

    with pytest.raises(ValueError, match="observation spaces differ"):
        MudParallelEnv(
            {
                "player_0": child_0,
                "player_1": child_1,
            }
        )


def test_render_returns_none_when_rendering_is_disabled(scripted_env_factory):
    child = scripted_env_factory(render_mode="ansi")
    child.reset()
    env = MudParallelEnv({"player_0": child}, render_mode=None)

    try:
        assert env.render() is None
    finally:
        env.close()


def test_ansi_render_labels_child_output(scripted_env_factory):
    child = scripted_env_factory(render_mode="ansi")
    child.reset()
    env = MudParallelEnv({"player_0": child}, render_mode="ansi")

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
    env = MudParallelEnv({"player_0": child}, render_mode="human")

    try:
        assert env.render() is None

        captured = capsys.readouterr()
        assert captured.out.startswith("[player_0]\n")
        assert "Dally Lane" in captured.out
        assert child.render_mode == "ansi"
    finally:
        env.close()


def test_validate_actions_rejects_missing_and_extra_agents(scripted_env_factory):
    env = MudParallelEnv(
        {
            "player_0": scripted_env_factory(),
            "player_1": scripted_env_factory(),
        }
    )

    try:
        with pytest.raises(ValueError, match="Missing=\\['player_1'\\]"):
            env.validate_actions({"player_0": "look"})

        with pytest.raises(ValueError, match="extra=\\['player_2'\\]"):
            env.validate_actions({"player_0": "look", "player_1": "look", "player_2": "look"})
    finally:
        env.close()


def test_rotate_step_order_advances_after_each_step(scripted_env_factory):
    env = MudParallelEnv(
        {f"player_{i}": scripted_env_factory() for i in range(3)},
        step_order="rotate",
    )
    actions = {f"player_{i}": "look" for i in range(3)}

    try:
        first_observations, *_ = env.step(actions)
        second_observations, *_ = env.step(actions)

        assert list(first_observations) == ["player_0", "player_1", "player_2"]
        assert list(second_observations) == ["player_1", "player_2", "player_0"]
    finally:
        env.close()


def test_step_removes_terminated_agents(scripted_env_factory):
    responses = {"quit": (scripted_response("quit,sql,fes,fex,fei"), True, False, {})}
    env = MudParallelEnv({"player_0": scripted_env_factory(responses=responses)})

    try:
        _, _, terminations, truncations, _ = env.step({"player_0": "quit"})

        assert terminations == {"player_0": True}
        assert truncations == {"player_0": False}
        assert env.agents == []
    finally:
        env.close()


def test_fixed_step_order_sends_actions_to_children_in_agent_order(scripted_env_factory):
    connection_0 = ScriptedConnection()
    connection_1 = ScriptedConnection()
    env = MudParallelEnv(
        {
            "player_0": scripted_env_factory(connection=connection_0),
            "player_1": scripted_env_factory(connection=connection_1),
        },
        step_order="fixed",
    )

    try:
        env.step({"player_0": "look", "player_1": "dance"})

        assert connection_0.commands == ["look,sql,fes,fex,fei"]
        assert connection_1.commands == ["dance,sql,fes,fex,fei"]
    finally:
        env.close()
