from mudgym.envs.factory import make_env, make_parallel_env, make_vector_env
from tests.scripted import ScriptedConnection, ScriptedProvider


class WorldTickerRecorder:
    """Record each advancement with the unread line counts the transports held at call time."""

    def __init__(self, connections_supplier):
        self.connections_supplier = connections_supplier
        self.calls: list[list[int]] = []

    def __call__(self) -> None:
        self.calls.append([len(connection.pending_lines) for connection in self.connections_supplier()])


def test_scalar_step_ticks_between_the_action_and_its_observation():
    connection = ScriptedConnection()
    world_ticker = WorldTickerRecorder(lambda: [connection])
    env = make_env(connection=connection, world_ticker=world_ticker)
    try:
        env.reset()
        assert world_ticker.calls == []
        env.step("look")
        # One advancement, taken while the action line was sent but unread.
        assert world_ticker.calls == [[1]]
        env.step("dance")
        assert world_ticker.calls == [[1], [1]]
    finally:
        env.close()


def test_vector_ticks_once_per_joint_step_after_every_action():
    provider = ScriptedProvider()
    world_ticker = WorldTickerRecorder(lambda: provider.connections)
    env = make_vector_env(2, provider=provider, world_ticker=world_ticker)
    try:
        env.reset()
        assert world_ticker.calls == []
        env.step(["look", "dance"])
        # Both children had acted (one unread action line each) before the single advancement.
        assert world_ticker.calls == [[1, 1]]
    finally:
        env.close()


def test_vector_children_never_advance_shared_world_time():
    provider = ScriptedProvider()
    world_ticker = WorldTickerRecorder(lambda: provider.connections)
    env = make_vector_env(2, provider=provider, world_ticker=world_ticker)
    try:
        env.reset()
        env.envs[0].step("look")
        assert world_ticker.calls == []
        env.step(["look", "dance"])
        assert world_ticker.calls == [[1, 1]]
    finally:
        env.close()


def test_parallel_ticks_the_shared_world_once_per_joint_step():
    provider = ScriptedProvider()
    world_ticker = WorldTickerRecorder(lambda: provider.connections)
    env = make_parallel_env(2, provider=provider, world_ticker=world_ticker)
    try:
        env.reset()
        assert world_ticker.calls == []
        env.step({"player_0": "look", "player_1": "dance"})
        assert world_ticker.calls == [[1, 1]]
    finally:
        env.close()


def test_parallel_children_never_advance_shared_world_time():
    provider = ScriptedProvider()
    world_ticker = WorldTickerRecorder(lambda: provider.connections)
    env = make_parallel_env(2, provider=provider, world_ticker=world_ticker)
    try:
        env.reset()
        env.envs["player_0"].step("look")
        assert world_ticker.calls == []
        env.step({"player_0": "look", "player_1": "dance"})
        assert world_ticker.calls == [[1, 1]]
    finally:
        env.close()


def test_directions_wrapper_preserves_the_world_ticker():
    connection = ScriptedConnection()
    world_ticker = WorldTickerRecorder(lambda: [connection])
    env = make_env(connection=connection, actions="directions", world_ticker=world_ticker)
    try:
        env.reset()
        env.step(0)
        assert world_ticker.calls == [[1]]
    finally:
        env.close()
