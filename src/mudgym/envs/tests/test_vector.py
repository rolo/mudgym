from typing import Any

import numpy as np
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

    def reset(self, *, seed: int | None = None) -> None:
        self.events.append(("connection.reset", self.index))
        super().reset(seed=seed)

    def send_line(self, line: str) -> None:
        self.events.append(("send", self.index, line))
        super().send_line(line)

    def read_response(self, end_of_turn_marker) -> tuple[bytes, bool, bool, dict[str, Any]]:
        self.events.append(("receive", self.index, list(self.pending_lines)))
        return super().read_response(end_of_turn_marker)


class TrackingProvider:
    def __init__(self):
        self.requested_count: int | None = None
        self.events: list[tuple] = []
        self.connections: list[TrackingConnection] = []
        self.closed = False

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
    provider = TrackingProvider()
    vector_env = make_vector_env(2, provider=provider, **kwargs)
    return vector_env, provider


def test_vector_reset_prepares_every_child_before_entry_and_final_observations():
    vector_env, provider = make_tracking_vector()
    try:
        vector_env.reset(seed=17)

        assert vector_env.np_random_seed == 17
        assert provider.requested_count == 2
        observation_line = "sql,fes,fex,fei"
        assert provider.events == [
            ("provider.reset", 17),
            ("connection.reset", 0),
            ("send", 0, "qs"),
            ("receive", 0, ["qs"]),
            ("connection.reset", 1),
            ("send", 1, "qs"),
            ("receive", 1, ["qs"]),
            ("send", 0, "move north"),
            ("receive", 0, ["move north"]),
            ("send", 1, "move north"),
            ("receive", 1, ["move north"]),
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
        assert all(isinstance(render_bytes, bytes) for render_bytes in infos["render_bytes"])
        assert infos["_render_bytes"].tolist() == [True, True]
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


@pytest.mark.parametrize("actions", [["look"], ["look", "dance", "bow"]])
def test_vector_rejects_wrong_action_count_before_sending(actions):
    vector_env, provider = make_tracking_vector()
    try:
        vector_env.reset()
        provider.events.clear()

        with pytest.raises(ValueError, match="Expected 2 actions"):
            vector_env.step(actions)

        assert provider.events == []
    finally:
        vector_env.close()


def test_vector_supports_vector_action_wrappers():
    vector_env, _ = make_tracking_vector(
        actions="directions",
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    try:
        vector_env.reset()
        vector_env.step([0, 1])

        assert vector_env.single_action_space.n == 14
        assert vector_env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP
    finally:
        vector_env.close()


def test_vector_masked_reset_resets_only_selected_children():
    vector_env, provider = make_tracking_vector()
    try:
        vector_env.reset(seed=17)
        observations_before_reset, *_ = vector_env.step(["look", "dance"])
        provider.events.clear()
        reset_mask = np.asarray([False, True], dtype=np.bool_)
        options = {"reset_mask": reset_mask}

        observations, infos = vector_env.reset(seed=[101, 202], options=options)

        assert options["reset_mask"] is reset_mask
        assert not any(event[0] == "provider.reset" for event in provider.events)
        assert [event for event in provider.events if event[0] == "connection.reset"] == [
            ("connection.reset", 1),
        ]
        assert all(event[1] == 1 for event in provider.events if event[0] in {"send", "receive"})
        assert observations["text"][0] == observations_before_reset["text"][0]
        assert vector_env.envs[0].np_random_seed == 17
        assert vector_env.envs[1].np_random_seed == 202
        assert infos["_step"].tolist() == [False, True]
    finally:
        vector_env.close()


def test_vector_first_reset_accepts_a_full_reset_mask():
    vector_env, provider = make_tracking_vector()
    try:
        observations, infos = vector_env.reset(seed=5, options={"reset_mask": np.asarray([True, True])})

        assert provider.events[0] == ("provider.reset", 5)
        assert [event for event in provider.events if event[0] == "connection.reset"] == [
            ("connection.reset", 0),
            ("connection.reset", 1),
        ]
        assert len(observations["text"]) == 2
        assert infos["_step"].tolist() == [True, True]
    finally:
        vector_env.close()


def test_vector_first_reset_rejects_a_partial_reset_mask():
    vector_env, _ = make_tracking_vector()
    try:
        with pytest.raises(RuntimeError, match="partial reset_mask"):
            vector_env.reset(options={"reset_mask": np.asarray([True, False])})
    finally:
        vector_env.close()


def test_vector_masked_reset_of_the_dead_child_lets_the_next_step_proceed():
    vector_env, provider = make_tracking_vector()
    provider.connections[0].responses["fuck"] = (
        b"fuck,sql,fes,fex,fei\r\n"
        b"In order to keep the game uncorrupted, you have been killed.\r\n"
        b"(Persona saved on -11 = \x1b[0;31;40m189\x1b[1;37;40m).\r\n",
        True,
        False,
        {"scripted": True, "marker_arrived": False},
    )
    try:
        vector_env.reset()
        _, _, terminations, _, _ = vector_env.step(["fuck", "look"])
        assert terminations.tolist() == [True, False]
        provider.events.clear()

        vector_env.reset(options={"reset_mask": np.asarray([True, False])})
        _, rewards, terminations, truncations, _ = vector_env.step(["look", "dance"])

        assert [event for event in provider.events if event[0] == "connection.reset"] == [
            ("connection.reset", 0),
        ]
        assert ("send", 0, "look") in provider.events
        assert ("send", 1, "dance") in provider.events
        assert rewards.tolist() == [0.0, 0.0]
        assert terminations.tolist() == [False, False]
        assert truncations.tolist() == [False, False]
    finally:
        vector_env.close()


def test_vector_next_step_autoreset_preserves_terminal_result_then_resets_only_done_child():
    ticker_calls = []

    def world_ticker():
        ticker_calls.append([len(connection.pending_lines) for connection in provider.connections])

    vector_env, provider = make_tracking_vector(
        autoreset_mode=AutoresetMode.NEXT_STEP,
        world_ticker=world_ticker,
    )
    terminal_bytes = b"die\r\nYou have died.\r\n"
    provider.connections[0].responses["die"] = (
        terminal_bytes,
        True,
        False,
        {"scripted": True, "marker_arrived": False},
    )
    try:
        vector_env.reset(seed=31)
        provider.events.clear()

        _, rewards, terminations, truncations, infos = vector_env.step(["die", "look"])

        assert rewards.tolist() == [0.0, 0.0]
        assert terminations.tolist() == [True, False]
        assert truncations.tolist() == [False, False]
        assert infos["raw_bytes"][0] == terminal_bytes
        assert ticker_calls == [[1, 1]]

        provider.events.clear()
        ticker_calls.clear()
        _, rewards, terminations, truncations, _ = vector_env.step(["ignored", "dance"])

        assert rewards.tolist() == [0.0, 0.0]
        assert terminations.tolist() == [False, False]
        assert truncations.tolist() == [False, False]
        assert ("send", 0, "ignored") not in provider.events
        assert ("send", 1, "dance") in provider.events
        assert ticker_calls == [[0, 1]]
        assert [event for event in provider.events if event[0] == "connection.reset"] == [
            ("connection.reset", 0),
        ]
        live_observation = next(
            position for position, event in enumerate(provider.events) if event[0] == "receive" and event[1] == 1
        )
        done_reset = provider.events.index(("connection.reset", 0))
        assert live_observation < done_reset
    finally:
        vector_env.close()


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
def test_failed_autoreset_keeps_the_live_childs_completed_transition(terminated, truncated):
    vector_env, provider = make_tracking_vector(autoreset_mode=AutoresetMode.NEXT_STEP)
    first, second = provider.connections
    first.responses["die"] = (b"die\r\nYou have died.\r\n", True, False, {"marker_arrived": False})
    second.responses["finish"] = (
        b"finish\r\nThe second episode has finished.\r\n",
        terminated,
        truncated,
        {"marker_arrived": False},
    )
    failure = OSError("autoreset preparation failed")
    try:
        vector_env.reset()
        vector_env.step(["die", "look"])
        first.send_errors["qs"] = failure
        with pytest.raises(OSError) as raised:
            vector_env.step(["ignored", "finish"])
        assert raised.value is failure
        assert vector_env._needs_reset.tolist() == [True, True]
        completed = vector_env.observations[1]
        assert "The second episode has finished." in completed["text"]

        first.send_errors.clear()
        vector_env.reset(options={"reset_mask": np.array([True, False])})
        assert vector_env._needs_reset.tolist() == [False, True]
        assert vector_env.observations[1] is completed
        provider.events.clear()
        _, rewards, terminations, truncations, infos = vector_env.step(["dance", "must not run"])
        assert ("send", 1, "must not run") not in provider.events
        assert ("connection.reset", 1) in provider.events
        assert infos["step"][1] == 0 and rewards[1] == 0
        assert not terminations.any() and not truncations.any()
    finally:
        vector_env.close()


def test_vector_next_step_autoreset_does_not_tick_when_every_child_is_resetting():
    ticker_calls = []
    vector_env, provider = make_tracking_vector(
        autoreset_mode=AutoresetMode.NEXT_STEP,
        world_ticker=lambda: ticker_calls.append("tick"),
    )
    for connection in provider.connections:
        connection.responses["die"] = (
            b"die\r\nYou have died.\r\n",
            True,
            False,
            {"scripted": True, "marker_arrived": False},
        )
    try:
        vector_env.reset()
        vector_env.step(["die", "die"])
        ticker_calls.clear()

        _, rewards, terminations, truncations, _ = vector_env.step(["ignored", "ignored"])

        assert ticker_calls == []
        assert rewards.tolist() == [0.0, 0.0]
        assert terminations.tolist() == [False, False]
        assert truncations.tolist() == [False, False]
    finally:
        vector_env.close()


def test_vector_default_still_requires_an_explicit_reset_after_a_terminal_result():
    vector_env, provider = make_tracking_vector()
    provider.connections[0].responses["die"] = (
        b"die\r\nYou have died.\r\n",
        True,
        False,
        {"scripted": True, "marker_arrived": False},
    )
    try:
        vector_env.reset()
        vector_env.step(["die", "look"])

        with pytest.raises(RuntimeError, match=r"children \[0\] are done"):
            vector_env.step(["ignored", "dance"])
    finally:
        vector_env.close()


@pytest.mark.parametrize(
    "autoreset_mode",
    [AutoresetMode.SAME_STEP, "SameStep", "not-a-mode"],
)
def test_vector_rejects_unsupported_autoreset_modes(autoreset_mode):
    with pytest.raises(ValueError, match="autoreset_mode|AutoresetMode"):
        make_tracking_vector(autoreset_mode=autoreset_mode)


@pytest.mark.parametrize(
    ("reset_mask", "message"),
    [
        ([True, False], "numpy array"),
        (np.asarray([True]), "shape"),
        (np.asarray([1, 0]), "dtype"),
        (np.asarray([False, False]), "at least one child"),
    ],
)
def test_vector_rejects_invalid_reset_masks(reset_mask, message):
    vector_env, _ = make_tracking_vector()
    try:
        vector_env.reset()
        with pytest.raises((TypeError, ValueError), match=message):
            vector_env.reset(options={"reset_mask": reset_mask})
    finally:
        vector_env.close()
