from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

import gymnasium as gym
from gymnasium.vector import VectorEnv
from pettingzoo import ParallelEnv

from mudgym.connections import registry
from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import ConnectionProvider
from mudgym.envs.actions.discrete import (
    DiscreteDirectionsWrapper,
    ParallelDiscreteDirectionsWrapper,
    VectorDiscreteDirectionsWrapper,
)
from mudgym.envs.env import MudEnv
from mudgym.envs.fields import (
    FEInventoryField,
    FEScoreField,
    FEXitsField,
    FieldSpec,
    MGCheatsField,
    RawBytesField,
    SuperQuickLookField,
)
from mudgym.envs.vector import MudVectorEnv
from mudgym.envs.zoo import MudParallelEnv

OBSERVATION_PRESETS: dict[str, tuple[FieldSpec, ...]] = {
    # Every preset must end with a field that can tell the transport its read window is complete.
    "bytes": (RawBytesField, FEScoreField(include_keys=("points",))),
    "text": (FEScoreField(include_keys=("points",)),),
    "parsed": (
        SuperQuickLookField(include_keys=("room_name", "room_name_index", "here", "features", "mobiles", "players")),
        FEScoreField,
        FEXitsField,
        FEInventoryField,
    ),
    "cheats": (
        FEScoreField,
        FEXitsField,
        MGCheatsField,
        FEInventoryField,
    ),
}


def close_quietly(*closeables: Any) -> None:
    """Close what we can without replacing the construction error already being raised."""
    for closeable in closeables:
        if closeable is not None:
            with suppress(Exception):
                closeable.close()


def _resolve_field_parsers(
    observation: str,
    field_parsers: Sequence[FieldSpec] | None,
) -> tuple[FieldSpec, ...]:
    if field_parsers is not None:
        return tuple(field_parsers)
    try:
        return OBSERVATION_PRESETS[observation]
    except KeyError:
        raise ValueError(f"observation must be one of {sorted(OBSERVATION_PRESETS)} (got {observation!r})") from None


def make_env(
    observation: str = "parsed",
    field_parsers: Sequence[FieldSpec] | None = None,
    actions: str = "text",
    render_mode: str | None = None,
    connection: str | type[MudConnection] | Callable[..., MudConnection] | MudConnection | None = None,
    connection_kwargs: Mapping[str, Any] | None = None,
    tearoom_commands: str | None = None,
) -> gym.Env:
    """Build one Gymnasium environment."""
    if actions not in {"text", "directions"}:
        raise ValueError(f"actions must be one of: 'text', 'directions' (got {actions!r})")
    if isinstance(connection, MudConnection) and connection_kwargs:
        raise ValueError("connection_kwargs is not valid when passing an explicit connection instance.")
    fields = _resolve_field_parsers(observation, field_parsers)

    connection_factory = registry.default_connection if connection is None else connection
    if isinstance(connection_factory, str):
        connection_factory = registry.connections[connection_factory]
    resolved_connection = (
        connection_factory
        if isinstance(connection_factory, MudConnection)
        else connection_factory(**dict(connection_kwargs or {}))
    )

    try:
        # MudEnv owns the connection as soon as construction succeeds. Until then it is still ours to close if field validation or session setup fails.
        env: gym.Env = MudEnv(
            connection=resolved_connection,
            field_parsers=fields,
            render_mode=render_mode,
            tearoom_commands=tearoom_commands,
        )
    except BaseException:
        close_quietly(resolved_connection)
        raise
    if actions == "text":
        return env

    try:
        return DiscreteDirectionsWrapper(env)
    except BaseException:
        close_quietly(env)
        raise


def make_vector_env(
    envs: int,
    *,
    observation: str = "parsed",
    field_parsers: Sequence[FieldSpec] | None = None,
    actions: str = "text",
    render_mode: str | None = None,
    tearoom_commands: str | None = None,
    provider: ConnectionProvider | None = None,
) -> VectorEnv:
    """Create a Gymnasium vector env. It need not know how worlds are arranged.

    The provider decides where the connections lead. Reset is deliberately all-or-nothing for now, and both reset and step finish the shared action/setup work before collecting observations. A supplied provider becomes the resulting environment's responsibility and is closed with it.
    """
    if envs < 1:
        raise ValueError("envs must be at least 1.")
    if actions not in {"text", "directions"}:
        raise ValueError(f"actions must be one of: 'text', 'directions' (got {actions!r})")
    fields = _resolve_field_parsers(observation, field_parsers)

    if provider is None:
        provider = registry.default_provider_factory()
    connections: list[MudConnection] = []
    children: list[MudEnv] = []

    try:
        # The provider returns the whole batch in one go so it can size any shared resources. We still check the length here: a custom provider should fail loudly rather than create a vector whose actual shape disagrees with its public shape.
        connections = provider.create_connections(envs)
        if len(connections) != envs:
            raise RuntimeError(f"Provider returned {len(connections)} connections, expected {envs}.")

        for connection in connections:
            children.append(
                MudEnv(
                    connection=connection,
                    field_parsers=fields,
                    render_mode=render_mode,
                    tearoom_commands=tearoom_commands,
                )
            )

        base_env = MudVectorEnv(children, provider=provider)
        if actions == "directions":
            return VectorDiscreteDirectionsWrapper(base_env)
        return base_env
    except BaseException:
        # Every successful child owns its matching connection. Anything after that prefix never made it into a MudEnv and still needs closing directly; the provider owns the resources underneath both groups. Cleanup errors are secondary to the construction failure.
        close_quietly(*children, *connections[len(children) :], provider)
        raise


def make_parallel_env(
    agents: int = 2,
    *,
    observation: str = "parsed",
    field_parsers: Sequence[FieldSpec] | None = None,
    actions: str = "text",
    render_mode: str | None = None,
    tearoom_commands: str | None = None,
    provider: ConnectionProvider | None = None,
) -> ParallelEnv:
    """Create a PettingZoo environment whose players share one MUD world.

    The registry supplies a one-world default. If a caller passes a provider we trust that it honours the same promise.
    The resulting environment owns that provider, and action wrappers sit around the joint environment rather than around each player.
    """
    if agents < 1:
        raise ValueError("agents must be at least 1.")
    if actions not in {"text", "directions"}:
        raise ValueError(f"actions must be one of: 'text', 'directions' (got {actions!r})")
    fields = _resolve_field_parsers(observation, field_parsers)
    child_render_mode = "ansi" if render_mode is not None else None

    if provider is None:
        provider = registry.default_parallel_provider_factory()
    connections: list[MudConnection] = []
    children: dict[str, MudEnv] = {}

    try:
        # As with the vector factory, batch size is worth checking even though the protocol promises it.
        connections = provider.create_connections(agents)
        if len(connections) != agents:
            raise RuntimeError(f"Provider returned {len(connections)} connections, expected {agents}.")

        for index, connection in enumerate(connections):
            children[f"player_{index}"] = MudEnv(
                connection=connection,
                field_parsers=fields,
                render_mode=child_render_mode,
                tearoom_commands=tearoom_commands,
            )

        base_env = MudParallelEnv(children, provider=provider, render_mode=render_mode)
        if actions == "directions":
            return ParallelDiscreteDirectionsWrapper(base_env)
        return base_env
    except BaseException:
        # Dict insertion order follows the connection batch. Successful children own the prefix;
        # the remaining connections were allocated but never made it into a MudEnv.
        close_quietly(*children.values(), *connections[len(children) :], provider)
        raise
