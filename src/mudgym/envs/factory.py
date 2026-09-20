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


def make_env(
    observation: str = "parsed",
    field_parsers: Sequence[FieldSpec] | None = None,
    actions: str = "text",
    render_mode: str | None = None,
    connection: str | type[MudConnection] | Callable[..., MudConnection] | MudConnection | None = None,
    connection_kwargs: Mapping[str, Any] | None = None,
    tearoom_commands: str | None = None,
    world_ticker: Callable[[], None] | None = None,
    *,
    persona: str | None = None,
    sex: str | None = None,
    persona_pool: Sequence[tuple[str, str | None]] | None = None,
) -> gym.Env:
    """Build a Gymnasium environment.

    Reset and step expose ``info["persona"]`` and ``info["persona_sex"]``. Non-None persona options override ``connection_kwargs``. Configure existing connections on their provider. See ``make_parallel_env`` for per-player tuple shorthand.

    ``world_ticker`` runs between the action and observation, defaulting to the connection's ``tick_for_step`` hook if present.
    """
    action_wrapper = {"text": None, "directions": DiscreteDirectionsWrapper}[actions]
    fields = OBSERVATION_PRESETS[observation] if field_parsers is None else tuple(field_parsers)
    options = {"persona": persona, "sex": sex, "persona_pool": persona_pool}
    connection_kwargs = {
        **(connection_kwargs or {}),
        **{key: value for key, value in options.items() if value is not None},
    }
    if isinstance(connection, MudConnection) and connection_kwargs:
        raise ValueError("Configure connection options on the provider when passing an explicit connection instance.")
    connection_factory = registry.default_connection if connection is None else connection
    if isinstance(connection_factory, str):
        connection_factory = registry.connections[connection_factory]
    resolved_connection = (
        connection_factory if isinstance(connection_factory, MudConnection) else connection_factory(**connection_kwargs)
    )
    if world_ticker is None:
        world_ticker = getattr(resolved_connection, "tick_for_step", None)

    try:
        # MudEnv owns the connection as soon as construction succeeds. Until then it is still ours to close if field
        # validation or session setup fails.
        env: gym.Env = MudEnv(
            connection=resolved_connection,
            field_parsers=fields,
            render_mode=render_mode,
            tearoom_commands=tearoom_commands,
            world_ticker=world_ticker,
        )
    except BaseException:
        close_quietly(resolved_connection)
        raise
    try:
        return env if action_wrapper is None else action_wrapper(env)
    except BaseException:
        close_quietly(env)
        raise


def create_players(
    count: int,
    provider: ConnectionProvider,
    field_parsers: Sequence[FieldSpec],
    render_mode: str | None,
    tearoom_commands: str | None,
    *,
    require_shared_world: bool = False,
) -> list[MudEnv]:
    """Adopt one provider batch and close everything if construction fails."""
    connections: list[MudConnection] = []
    children: list[MudEnv] = []
    try:
        connections = provider.create_connections(count)
        if len(connections) != count:
            raise RuntimeError(f"Provider returned {len(connections)} connections, expected {count}.")
        world_for_connection = getattr(provider, "world_for_connection", None)
        if require_shared_world and world_for_connection is not None:
            world_indexes = {world_for_connection(index) for index in range(len(connections))}
            if len(world_indexes) != 1:
                raise ValueError("Parallel environment connections must share one world.")
        for connection in connections:
            children.append(
                MudEnv(
                    connection=connection,
                    field_parsers=field_parsers,
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
    personas: Sequence[tuple[str] | tuple[str | None, str | None]] | None = None,
    persona_pool: Sequence[tuple[str, str | None]] | None = None,
) -> ParallelEnv:
    """Create a PettingZoo environment whose players share one MUD world.

    ``world_ticker`` runs after all actions and before any observations, defaulting to the provider's ``tick_for_step`` hook if present.
    """
    action_wrapper = {"text": None, "directions": ParallelDiscreteDirectionsWrapper}[actions]
    fields = OBSERVATION_PRESETS[observation] if field_parsers is None else tuple(field_parsers)
    child_render_mode = "ansi" if render_mode is not None else None

    if provider is not None and (personas is not None or persona_pool is not None):
        raise ValueError("Configure personas and persona_pool on the explicit provider.")
    if provider is None:
        options = {"personas": personas, "persona_pool": persona_pool}
        provider = registry.default_parallel_provider_factory(
            **{key: value for key, value in options.items() if value is not None}
        )
    if world_ticker is None:
        world_ticker = getattr(provider, "tick_for_step", None)
    children = {
        f"player_{index}": child
        for index, child in enumerate(
            create_players(
                agents,
                provider,
                fields,
                child_render_mode,
                tearoom_commands,
                require_shared_world=True,
            )
        )
    }
    try:
        base_env = MudParallelEnv(
            children,
            provider=provider,
            render_mode=render_mode,
            world_ticker=world_ticker,
        )
        return base_env if action_wrapper is None else action_wrapper(base_env)
    except BaseException:
        close_quietly(*children.values(), provider)
        raise
