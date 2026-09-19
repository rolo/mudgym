"""Ordered player lifecycle operations for parallel environments."""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from mudgym.connections.provider import ConnectionProvider
from mudgym.envs.env import MudEnv


def reset_worlds(players: Mapping[Any, MudEnv], provider: ConnectionProvider, seed) -> None:
    """Reset provider worlds and invalidate every affected player if replacement fails."""
    try:
        provider.reset(seed=seed)
    except BaseException as error:
        error.add_note("player reset failed while resetting the provider")
        for child in players.values():
            child._invalidate_reset(error)
        raise


def reset_players[Player](
    players: Mapping[Player, MudEnv],
    seeds: Mapping[Player, int | None],
    options: dict[str, Any] | None,
) -> dict[Player, tuple[dict[str, Any], dict[str, Any]]]:
    """Prepare every selected player, enter them all, then collect initial observations.

    This relogs selected players only. Full resets replace provider worlds before calling this function.
    """
    phase = "preparation"
    player = None
    try:
        for player, child in players.items():
            child._prepare_reset(seed=seeds[player], options=options)
        phase = "entry"
        entries = {}
        for player, child in players.items():
            entries[player] = child._enter_world()
        phase = "observation"
        results = {}
        for player, child in players.items():
            results[player] = child._finish_reset(*entries[player])
        return results
    except BaseException as error:
        error.add_note(f"player reset failed during {phase} for player {player!r}")
        for child in players.values():
            child._invalidate_reset(error)
        raise


def step_players[Player](
    players: Mapping[Player, MudEnv],
    actions: Mapping[Player, str],
    world_ticker: Callable[[], None] | None,
) -> Iterator[tuple[Player, tuple[dict[str, Any], float, bool, bool, dict[str, Any]]]]:
    """Act in player order and tick on the call, then return a lazy observation iterator.

    A caller must consume these results before relogging finished players. Yielding each transition lets the caller retain completed observations if a later player fails. Providers without a manual clock supply no ticker.
    """
    selected = [(player, players[player], action) for player, action in actions.items()]
    for player, child, action in selected:
        child.act(action)
    if selected and world_ticker is not None:
        world_ticker()
    return ((player, child.observe()) for player, child, action in selected)


def close_players(players: Mapping[Any, MudEnv], provider: ConnectionProvider) -> None:
    """Attempt every connection close before closing the provider, and report all failures."""
    errors: list[Exception] = []
    for child in players.values():
        try:
            child.close()
        except Exception as error:  # noqa: BLE001 - collect failures for the group raised after cleanup
            errors.append(error)
    try:
        provider.close()
    except Exception as error:  # noqa: BLE001 - include provider failures in the same cleanup group
        errors.append(error)
    if errors:
        raise ExceptionGroup("Player cleanup failed", errors)
