from collections.abc import Sequence
from typing import Any

import pytest
from gymnasium.vector import AutoresetMode

from mudgym.connections.connection import MudConnection
from mudgym.envs.factory import make_vector_env
from tests.scripted import ScriptedConnection


class TrackingConnection(ScriptedConnection):
    def __init__(self, index: int, events: list[tuple]):
        super().__init__()
        self.index = index
        self.events = events

    def reset(self) -> None:
        self.events.append(("connection.reset", self.index))
        super().reset()

    def send_line(self, line: str) -> None:
        self.events.append(("send", self.index, line))
        super().send_line(line)

    def read_response(self, lines: Sequence[str], end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        self.events.append(("receive", self.index, list(lines)))
        return super().read_response(lines, end_of_turn_marker)


class TrackingProvider:
    instances: list["TrackingProvider"] = []

    def __init__(self, *, worlds: int | None = None):
        self.worlds = worlds
        self.requested_count: int | None = None
        self.events: list[tuple] = []
        self.connections: list[TrackingConnection] = []
        self.closed = False
        self.instances.append(self)

    def create_connections(self, count: int) -> list[MudConnection]:
        self.requested_count = count
        for index in range(count):
            self.connections.append(TrackingConnection(index, self.events))
        return list(self.connections)

    def reset(self, *, seed=None) -> None:
        self.events.append(("provider.reset", seed))

    def close(self) -> None:
        self.closed = True


def make_tracking_vector(**kwargs):
    TrackingProvider.instances.clear()
    vector_env = make_vector_env(2, provider=TrackingProvider(worlds=1), **kwargs)
    return vector_env, TrackingProvider.instances[0]


def test_vector_reset_resets_provider_before_children():
    vector_env, provider = make_tracking_vector()
    try:
        vector_env.reset(seed=17)

        assert provider.events[0] == ("provider.reset", 17)
        assert vector_env.np_random_seed == 17
        assert [event for event in provider.events if event[0] == "connection.reset"] == [
            ("connection.reset", 0),
            ("connection.reset", 1),
        ]
        assert provider.requested_count == 2
        observation_line = "sql,fes,fex,fei"
        assert provider.events[-4:] == [
            ("send", 0, observation_line),
            ("receive", 0, [observation_line]),
            ("send", 1, observation_line),
            ("receive", 1, [observation_line]),
        ]
    finally:
        vector_env.close()


def test_vector_batches_child_info_with_gymnasium_masks():
    vector_env, _ = make_tracking_vector()
    try:
        _, infos = vector_env.reset()

        assert infos["step"].tolist() == [0, 0]
        assert infos["_step"].tolist() == [True, True]
    finally:
        vector_env.close()


def test_vector_copies_child_metadata_and_render_mode():
    vector_env, _ = make_tracking_vector(render_mode="ansi")
    try:
        assert vector_env.metadata["render_modes"] == ["human", "ansi"]
        assert vector_env.metadata["autoreset_mode"] is AutoresetMode.DISABLED
        assert vector_env.render_mode == "ansi"
    finally:
        vector_env.close()


def test_vector_step_sends_every_action_before_receiving_any_observation():
    vector_env, provider = make_tracking_vector()
    try:
        vector_env.reset()
        provider.events.clear()

        vector_env.step(["look", "dance"])

        assert provider.events == [
            ("send", 0, "look"),
            ("send", 1, "dance"),
            ("send", 0, "sql,fes,fex,fei"),
            ("receive", 0, ["look", "sql,fes,fex,fei"]),
            ("send", 1, "sql,fes,fex,fei"),
            ("receive", 1, ["dance", "sql,fes,fex,fei"]),
        ]
    finally:
        vector_env.close()


def test_vector_supports_vector_action_wrappers():
    vector_env, _ = make_tracking_vector(actions="directions")
    try:
        vector_env.reset()
        vector_env.step([0, 1])

        assert vector_env.single_action_space.n == 14
    finally:
        vector_env.close()


def test_vector_has_no_autoreset_option():
    with pytest.raises(TypeError, match="autoreset_mode"):
        make_tracking_vector(autoreset_mode="same_step")
