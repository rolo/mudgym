from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

import gymnasium as gym
from pettingzoo import ParallelEnv

from mudgym.connections import registry
from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import ConnectionProvider
from mudgym.envs.actions.discrete import (
    DiscreteDirectionsWrapper,
    ParallelDiscreteDirectionsWrapper,
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
from mudgym.envs.zoo import MudParallelEnv

OBSERVATION_PRESETS: dict[str, tuple[FieldSpec, ...]] = {
    "bytes": (RawBytesField,),
    "text": (),
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
    *,
    requires_end_of_turn_marker: bool = True,
) -> tuple[FieldSpec, ...]:
    if field_parsers is not None:
        return tuple(field_parsers)
    fields = OBSERVATION_PRESETS[observation]
    if requires_end_of_turn_marker and observation in {"text", "bytes"}:
        # Marker-based transports still need a command that closes the read window.
        return (*fields, FEScoreField(include_keys=()))
    return fields


def make_env(
    observation: str = "parsed",
    field_parsers: Sequence[FieldSpec] | None = None,
    actions: str = "text",
    render_mode: str | None = None,
    connection: str | type[MudConnection] | Callable[..., MudConnection] | MudConnection | None = None,
    connection_kwargs: Mapping[str, Any] | None = None,
    tearoom_commands: str | None = None,
    world_ticker: Callable[[], None] | None = None,
) -> gym.Env:
    """Build one Gymnasium environment.

    ``world_ticker`` runs once after the action for a step and before its observation. When omitted, the connection's ``tick_for_step`` hook supplies the clock if present.
    """
    if actions not in {"text", "directions"}:
        raise ValueError(f"actions must be one of: 'text', 'directions' (got {actions!r})")
    if isinstance(connection, MudConnection) and connection_kwargs:
        raise ValueError("connection_kwargs is not valid when passing an explicit connection instance.")
    if field_parsers is None and observation not in OBSERVATION_PRESETS:
        raise ValueError(f"observation must be one of {sorted(OBSERVATION_PRESETS)} (got {observation!r})")

    connection_factory = registry.default_connection if connection is None else connection
    if isinstance(connection_factory, str):
        connection_factory = registry.connections[connection_factory]
    resolved_connection = (
        connection_factory
        if isinstance(connection_factory, MudConnection)
        else connection_factory(**dict(connection_kwargs or {}))
    )
    if world_ticker is None:
        world_ticker = getattr(resolved_connection, "tick_for_step", None)

    try:
        # MudEnv owns the connection as soon as construction succeeds. Until then it is still ours to close if field
        # validation or session setup fails.
        env: gym.Env = MudEnv(
            connection=resolved_connection,
            field_parsers=_resolve_field_parsers(
                observation, field_parsers, requires_end_of_turn_marker=resolved_connection.requires_end_of_turn_marker
            ),
            render_mode=render_mode,
            tearoom_commands=tearoom_commands,
            world_ticker=world_ticker,
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


def create_players(
    count: int,
    provider: ConnectionProvider,
    observation: str,
    field_parsers: Sequence[FieldSpec] | None,
    render_mode: str | None,
    tearoom_commands: str | None,
) -> list[MudEnv]:
    """Adopt one provider batch and close everything if construction fails."""
    connections: list[MudConnection] = []
    children: list[MudEnv] = []
    try:
        connections = provider.create_connections(count)
        if len(connections) != count:
            raise RuntimeError(f"Provider returned {len(connections)} connections, expected {count}.")
        for connection in connections:
            children.append(
                MudEnv(
                    connection=connection,
                    field_parsers=_resolve_field_parsers(
                        observation, field_parsers, requires_end_of_turn_marker=connection.requires_end_of_turn_marker
                    ),
                    render_mode=render_mode,
                    tearoom_commands=tearoom_commands,
                )
            )
        return children
    except BaseException:
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
    world_ticker: Callable[[], None] | None = None,
) -> ParallelEnv:
    """Create a PettingZoo environment whose players share one MUD world.

    The registry supplies a one-world default. If a caller passes a provider we trust that it honours the same promise.
    The resulting environment owns that provider, and action wrappers sit around the joint environment rather than
    around each player.

    ``world_ticker`` runs once after all actions for a step and before any observations. It belongs to the coordinator
    alone. When omitted, the provider's ``tick_for_step`` hook supplies the clock if present.
    """
    if agents < 1:
        raise ValueError("agents must be at least 1.")
    if actions not in {"text", "directions"}:
        raise ValueError(f"actions must be one of: 'text', 'directions' (got {actions!r})")
    if field_parsers is None and observation not in OBSERVATION_PRESETS:
        raise ValueError(f"observation must be one of {sorted(OBSERVATION_PRESETS)} (got {observation!r})")
    child_render_mode = "ansi" if render_mode is not None else None

    if provider is None:
        provider = registry.default_parallel_provider_factory()
    if world_ticker is None:
        world_ticker = getattr(provider, "tick_for_step", None)
    children = {
        f"player_{index}": child
        for index, child in enumerate(
            create_players(agents, provider, observation, field_parsers, child_render_mode, tearoom_commands)
        )
    }
    try:
        base_env = MudParallelEnv(
            children,
            provider=provider,
            render_mode=render_mode,
            world_ticker=world_ticker,
        )
        if actions == "directions":
            return ParallelDiscreteDirectionsWrapper(base_env)
        return base_env
    except BaseException:
        close_quietly(*children.values(), provider)
        raise
