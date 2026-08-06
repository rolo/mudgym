from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from functools import partial
from typing import Any

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import FilterObservation

from mudgym.connections.connection import MudConnection
from mudgym.connections.provider import ConnectionProvider, DockerExecProvider
from mudgym.connections.registry import connections, default_connection
from mudgym.envs.actions.discrete import DiscreteDirectionsWrapper
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
from mudgym.envs.zoo import MudParallelEnv, RenderMode, StepOrder

OBSERVATION_FIELDS: dict[str, tuple[FieldSpec, ...]] = {
    # every preset needs to end with an end of turn marker capable field
    "bytes": (RawBytesField, FEScoreField(include_keys=())),
    "text": (FEScoreField(include_keys=("points",)),),
    "parsed": (
        # sql also parses portables/inventory from the look text, but FEInventoryField's dedicated
        # fei response owns those keys
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


def _close_quietly(*closeables: Any) -> None:
    """Best-effort teardown: a failing close() must not mask the error that got us here."""
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
        return OBSERVATION_FIELDS[observation]
    except KeyError:
        raise ValueError(f"observation must be one of {sorted(OBSERVATION_FIELDS)} (got {observation!r})") from None


def make_env(
    observation: str = "parsed",
    exclude_keys: str | Sequence[str] | None = None,
    field_parsers: Sequence[FieldSpec] | None = None,
    actions: str = "text",
    render_mode: RenderMode | None = None,
    connection: str | type[MudConnection] | Callable[..., MudConnection] | MudConnection = default_connection,
    connection_kwargs: Mapping[str, Any] | None = None,
    wrappers: Sequence[Callable[[gym.Env], gym.Env]] | None = None,
    auto_commands: Sequence[str] | None = None,
    tearoom_commands: str | None = None,
) -> gym.Env:
    """Build a single MUD2 environment, applying observation parsing, action wrapping, and rendering.

    Args:
        observation: Named observation preset -- ``"bytes"``, ``"text"``, ``"parsed"`` (default), or ``"cheats"``.
        exclude_keys: Observation key(s) to drop from the resulting space; unknown keys raise.
        field_parsers: Explicit observation fields; replaces the ``observation`` preset when given.
        actions: ``"text"`` (free-form, default) or ``"directions"`` (discrete compass wrapper).
            A text action is a single logical game input line: comma-chaining is allowed, line
            breaks are not (the env appends its own auto-command line each step).
        render_mode: ``None``, ``"human"``, or ``"ansi"``.
        connection: Connection backend -- a registry slug, a class, an instance, or any zero-arg callable.
        connection_kwargs: Kwargs bound to the connection class/callable; not valid with an instance.
        wrappers: Extra Gymnasium wrappers applied last, in order.
        auto_commands: Commands appended after the action each step to populate fields; defaults to the
            configured fields' commands.
        tearoom_commands: Command line issued once per ``reset()`` while still in the tearoom
    """
    env_kwargs: dict[str, Any] = {}
    # field_parsers replaces the observation preset when given (full control); otherwise the named preset.
    env_kwargs["field_parsers"] = _resolve_field_parsers(observation, field_parsers)
    env_kwargs["auto_commands"] = auto_commands
    env_kwargs["tearoom_commands"] = tearoom_commands
    env_kwargs["render_mode"] = render_mode
    exclude_list = [exclude_keys] if isinstance(exclude_keys, str) else list(exclude_keys or [])

    # resolve string slugs to a connection class
    if isinstance(connection, str):
        connection = connections[connection]

    # already instantiated
    if isinstance(connection, MudConnection):
        if connection_kwargs:
            raise ValueError("connection_kwargs is not valid when passing an explicit connection instance.")
        env_kwargs["connection"] = connection
    else:
        # a connection class (or any zero-arg-capable callable); bind kwargs into a factory
        env_kwargs["connection"] = partial(connection, **connection_kwargs) if connection_kwargs else connection

    # Validate configuration before acquiring resources: MudEnv owns a connection, so anything that can
    # be rejected up front must be rejected before the env (and its connection) is constructed.
    if actions not in {"text", "directions"}:
        raise ValueError(f"actions must be one of: 'text', 'directions' (got {actions!r})")

    # Once MudEnv is constructed it owns the connection; any failure while wrapping it must close the
    # env (Gymnasium treats close() as the cleanup point for external resources) so we don't leak it.
    env: gym.Env = MudEnv(**env_kwargs)
    try:
        # drop unwanted keys, failing fast on typos so callers learn about bad keys here
        if exclude_list:
            existing = set(env.observation_space.spaces.keys())
            unknown = set(exclude_list) - existing
            if unknown:
                raise ValueError(f"Unknown exclude_keys: {sorted(unknown)}. Valid: {sorted(existing)}")

            excluded = set(exclude_list)
            final_keys = [key for key in env.observation_space.spaces if key not in excluded]
            if not final_keys:
                raise ValueError("exclude_keys cannot remove every observation key.")
            if len(final_keys) != len(existing):
                env = FilterObservation(env, filter_keys=final_keys)

        # action space wrappers ("text" needs none)
        if actions == "directions":
            env = DiscreteDirectionsWrapper(env)

        # any other wrappers
        for wrapper in wrappers or ():
            env = wrapper(env)

        return env
    except BaseException:
        env.close()
        raise


def make_vector_env(
    envs: int,
    *,
    worlds: int | None = None,
    make_env_kwargs: Mapping[str, Any] | None = None,
    provider_factory: Callable[..., ConnectionProvider] = DockerExecProvider,
    provider_kwargs: Mapping[str, Any] | None = None,
    wrappers: Sequence[Callable[[gym.Env], gym.Env]] | None = None,
    autoreset_mode: AutoresetMode = AutoresetMode.DISABLED,
) -> MudVectorEnv:
    """
    Create a vectorized environment backed by a ConnectionProvider.

    Args:
        envs: Number of environments.
        worlds: Number of game world instances. Defaults to `envs` (one world per env).
        make_env_kwargs: Keyword args to pass to `make_env()`.
        provider_factory: Callable that creates a ConnectionProvider. Defaults to DockerExecProvider.
        provider_kwargs: Keyword args to pass to `provider_factory`.
        autoreset_mode: Defaults to `AutoresetMode.DISABLED`; `NEXT_STEP` and `SAME_STEP` are also supported.
    """

    # default to one game world per env if not specified
    if worlds is None:
        worlds = envs

    make_env_kwargs = dict(make_env_kwargs or {})

    provider_kwargs = dict(provider_kwargs or {})
    provider_kwargs.setdefault("worlds", worlds)
    connection_provider = provider_factory(**provider_kwargs)

    # The provider owns shared infrastructure (containers) the moment it is constructed, and
    # create_connection() / SyncVectorEnv / MudVectorEnv can all fail. Build the whole sequence inside a
    # single cleanup scope so a failure anywhere tears down both the vector env (and its sub-env
    # connections) and the provider, rather than leaking them.
    venv: SyncVectorEnv | None = None
    try:
        env_fns = [
            partial(
                make_env,
                connection=connection_provider.create_connection(i),
                wrappers=wrappers,
                **make_env_kwargs,
            )
            for i in range(envs)
        ]
        venv = SyncVectorEnv(env_fns, copy=False, autoreset_mode=autoreset_mode)
        return MudVectorEnv(venv, provider=connection_provider)
    except BaseException:
        _close_quietly(venv, connection_provider)
        raise


def make_parallel_env(
    agents: int = 2,
    *,
    make_env_kwargs: Mapping[str, Any] | None = None,
    provider_factory: Callable[..., ConnectionProvider] = DockerExecProvider,
    provider_kwargs: Mapping[str, Any] | None = None,
    render_mode: RenderMode | None = None,
    step_order: StepOrder = "rotate",
) -> MudParallelEnv:
    """Create a PettingZoo ParallelEnv with multiple agents sharing one game world.

    Args:
        agents: Number of agents.
        make_env_kwargs: Keyword args passed to `make_env()` for each agent.
        provider_factory: Callable that creates a ConnectionProvider.
        provider_kwargs: Keyword args for provider_factory.
        render_mode: Top-level PettingZoo render mode. When set, child envs default to
            ``ansi`` so the wrapper labels each frame rather than children printing on their own.
        step_order: How same tick actions resolve in the shared world. "rotate" stops one
            agent always going first, "fixed" keeps a stable order, and "shuffle"
            randomises the order each step.
    """
    make_env_kwargs = dict(make_env_kwargs or {})
    if render_mode is not None:
        child_render_mode = make_env_kwargs.setdefault("render_mode", "ansi")
        if child_render_mode != "ansi":
            raise ValueError(
                "make_parallel_env(render_mode=...) requires child envs with render_mode='ansi'; "
                "omit make_env_kwargs['render_mode'] to use the default."
            )

    provider_kwargs = dict(provider_kwargs or {})
    provider_kwargs.setdefault("worlds", 1)
    connection_provider = provider_factory(**provider_kwargs)
    envs: dict[str, gym.Env] = {}

    try:
        for i in range(agents):
            envs[f"player_{i}"] = make_env(
                connection=connection_provider.create_connection(i),
                **make_env_kwargs,
            )
        return MudParallelEnv(
            envs,
            provider=connection_provider,
            render_mode=render_mode,
            step_order=step_order,
        )
    except BaseException:
        _close_quietly(*envs.values(), connection_provider)
        raise
