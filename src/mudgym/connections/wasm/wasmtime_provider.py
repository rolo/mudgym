"""Connections, world ownership and transitions over the Wasmtime WASI runtime."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from mudgym.connections.connection import MudConnection
from mudgym.connections.errors import ConnectionClosedError
from mudgym.connections.persona import PERSONA_NAMES
from mudgym.connections.prompts import INVALID_COMMAND_PROMPTS
from mudgym.connections.termination import has_game_over_prompt
from mudgym.featurizers.strings import encode_command_bytes

from .engine_contract import (
    CommandTicketStatus,
    WorldStatus,
    validate_seed,
    validate_session_count,
)
from .futures import ordered_future_results
from .wasmtime_runtime import (
    WasmtimeRuntime,
    WasmtimeSession,
    WasmtimeTerminalTick,
    WasmtimeWorld,
)


def _shutdown_worlds(worlds: Iterable[WasmtimeWorld]) -> list[BaseException]:
    """Attempt every shutdown and retain failures for the caller's error report."""
    errors = []
    for world in worlds:
        try:
            world.shutdown()
        except BaseException as error:
            errors.append(error)
    return errors


class WorldAdvancementFailed(RuntimeError):
    """Some worlds ended during advancement, while healthy siblings still advanced.

    ``successes`` maps each advanced world to its tick count. ``failures`` retains each terminal world's
    exception, including any partial advancement reported by the engine.
    """

    def __init__(
        self,
        message: str,
        *,
        successes: dict[int, int],
        failures: dict[int, WasmtimeTerminalTick],
    ):
        super().__init__(message)
        self.successes = successes
        self.failures = failures


@dataclass(frozen=True, slots=True)
class WasmtimeResponse:
    raw_bytes: bytes
    game_bytes: bytes
    echoed_line: str | None
    terminated: bool
    incomplete: bool


class WasmtimeMudConnection(MudConnection):
    """One stable MudGym connection rebound to each fresh WASI session."""

    requires_end_of_turn_marker = False

    def __init__(self, *, provider: WasmtimeProvider, connection_index: int, timeout_ms: int) -> None:
        super().__init__(db_slot=provider.world_for_connection(connection_index))
        self.provider = provider
        self.connection_index = connection_index
        self.timeout_ms = timeout_ms
        self.owns_provider = False
        self._session: WasmtimeSession | None = None
        self._started = False
        self._closed = False
        self._invalidated = False
        self._pending_responses: list[WasmtimeResponse] = []

    def _bind(self, session: WasmtimeSession) -> None:
        if self._pending_responses:
            raise RuntimeError(f"cannot replace connection {self.connection_index} with unread responses")
        self._session = session
        self._started = False
        self._invalidated = False

    def reset(self, *, seed: int | None = None) -> None:
        if self._closed:
            raise RuntimeError("Wasmtime connection is closed")
        if self.owns_provider:
            self.provider.reset(seed=seed)
        else:
            self.provider._prepare_connection(self, seed=seed)
        session = self._require_session()
        session.receive()
        ticket = session.send("sip t", self.timeout_ms)
        if ticket.status is not CommandTicketStatus.COMPLETED:
            raise RuntimeError(f"tearoom preparation returned {ticket.status.name}")
        session.receive()
        self._pending_responses.clear()
        self._started = True
        self._invalidated = False

    def _require_session(self) -> WasmtimeSession:
        if self._session is None:
            raise RuntimeError("Wasmtime connection has no live session; call reset() first")
        return self._session

    def _execute_line(self, command: str) -> WasmtimeResponse:
        command_bytes = encode_command_bytes(command)
        session = self._require_session()
        with session.world.lock:
            pending_output = session.receive()
            if session.departed:
                # The final receive retires the native handle. No command was sent.
                return WasmtimeResponse(
                    raw_bytes=pending_output,
                    game_bytes=pending_output,
                    echoed_line=None,
                    terminated=True,
                    incomplete=False,
                )
            ticket = session.send(command, self.timeout_ms)
            queued_output = b""
            if ticket.status is not CommandTicketStatus.HALTED:
                queued_output = session.receive()
        game_bytes = pending_output + ticket.output + queued_output
        raw_bytes = pending_output + command_bytes + b"\r\n" + ticket.output + queued_output
        # A drain may report departure with final bytes after a COMPLETED ticket.
        # Scan all engine output, including bytes queued before this command.
        terminated = (
            ticket.status
            in {
                CommandTicketStatus.HALTED,
                CommandTicketStatus.PLAYER_DEPARTED,
            }
            or session.departed
            or has_game_over_prompt(game_bytes)
        )
        incomplete = ticket.status is CommandTicketStatus.INPUT_REQUIRED
        if ticket.world_indeterminate:
            # An indeterminate world also returns FAILED, but must preserve the transition for truncation.
            incomplete = True
        elif ticket.status is CommandTicketStatus.FAILED:
            raise RuntimeError(f"WASI command {command!r} returned FAILED")
        if not terminated and not incomplete and not session.world.is_alive():
            incomplete = True
        return WasmtimeResponse(
            raw_bytes=raw_bytes,
            game_bytes=game_bytes,
            echoed_line=command,
            terminated=terminated,
            incomplete=incomplete,
        )

    def send_line(self, line: str) -> None:
        if not self._started:
            raise RuntimeError("Connection has not been reset, call reset() first.")
        if self._invalidated:
            raise ConnectionClosedError("Wasmtime connection was invalidated; call reset() before sending")
        if self._pending_responses:
            previous = self._pending_responses[-1]
            if previous.terminated or previous.incomplete:
                raise ConnectionClosedError("Wasmtime connection closed after the previous command line")
        self._pending_responses.append(self._execute_line(line))

    def read_response(self, end_of_turn_marker: re.Pattern | None) -> tuple[bytes, bool, bool, dict[str, Any]]:
        if self._closed or not self._started:
            raise ConnectionClosedError("Wasmtime connection is not open. Call reset() before receiving.")
        responses = self._pending_responses
        self._pending_responses = []
        raw_bytes = b"".join(response.raw_bytes for response in responses)
        # Retain the engine bytes separately so synthetic echoes never become game control text.
        game_bytes = b"".join(response.game_bytes for response in responses)
        terminated = any(response.terminated for response in responses)
        incomplete = any(response.incomplete for response in responses)
        # Later peer commands and the coordinator's tick can enqueue output after our send.
        # Drain at the observation boundary, including passive reads during a joint reset.
        session = self._require_session()
        if not session.departed:
            queued_output = session.receive()
            raw_bytes += queued_output
            game_bytes += queued_output
        status = session.world.status()
        terminated = terminated or session.departed or status is WorldStatus.HALTED or has_game_over_prompt(game_bytes)
        incomplete = incomplete or status is WorldStatus.INDETERMINATE
        return (
            raw_bytes,
            terminated,
            incomplete,
            {
                "backend": "wasmtime",
                "connection_index": self.connection_index,
                "world_index": self.provider.world_for_connection(self.connection_index),
                "bytes_length": len(raw_bytes),
                "marker_arrived": not (terminated or incomplete),
                "rejected": any(pattern.search(game_bytes) for pattern in INVALID_COMMAND_PROMPTS),
                "sent_lines": [response.echoed_line for response in responses if response.echoed_line is not None],
            },
        )

    def advance_world_ticks(self, ticks: int) -> int:
        with self.provider._lock:
            if self._closed:
                raise RuntimeError("Wasmtime connection is closed")
            session = self._require_session()
            try:
                return session.world.tick(ticks, self.timeout_ms)
            except WasmtimeTerminalTick as terminal:
                world_index = self.provider.world_for_connection(self.connection_index)
                raise WorldAdvancementFailed(
                    f"world advancement failed for world {world_index}",
                    successes={},
                    failures={world_index: terminal},
                ) from terminal

    def tick_for_step(self) -> None:
        # Observation delivers the final transition when a world ends during this tick.
        with suppress(WorldAdvancementFailed):
            self.advance_world_ticks(1)

    def invalidate(self) -> None:
        self._pending_responses = []
        self._invalidated = True

    def _mark_closed(self) -> None:
        """Release Store-owning references after the provider is shut down."""
        self._pending_responses = []
        self._session = None
        self._closed = True
        self._started = False
        self._invalidated = True

    def close(self) -> None:
        if self._closed:
            return
        if self.owns_provider:
            self.provider.close()
            return
        self._pending_responses = []
        session = self._session
        if session is not None and session.world.is_alive() and not session.departed:
            session.world.remove_session(session, self.timeout_ms)
        self._mark_closed()


class WasmtimeProvider:
    """MudGym provider spreading stable connections over fresh isolated WASI worlds."""

    def __init__(
        self,
        *,
        runtime: WasmtimeRuntime | None = None,
        worlds: int | None = None,
        seed: int = 0,
        civil_time_anchor: datetime = datetime(2026, 1, 1, tzinfo=UTC),
        timeout_ms: int = 5_000,
    ) -> None:
        # Resolve worlds=None when the factory supplies its connection count.
        # An explicit world count shares connections by modulo.
        if worlds is not None and (isinstance(worlds, bool) or not isinstance(worlds, int) or worlds < 1):
            raise ValueError("worlds must be a positive integer or None")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise ValueError("timeout_ms must be a positive integer")
        validate_seed(seed, label="seed")
        if worlds is not None:
            validate_seed(seed + worlds - 1, label="seed plus the final world index")
        self.runtime = WasmtimeRuntime() if runtime is None else runtime
        self.worlds = worlds
        self.seed = seed
        self.civil_time_anchor = civil_time_anchor
        self.timeout_ms = timeout_ms
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._connections: list[WasmtimeMudConnection] = []
        self._world_slots: list[list[int]] = []
        self._ordered_worlds: list[WasmtimeWorld] = []
        self._world_seeds: tuple[int, ...] = ()
        self._closed = False

    def world_for_connection(self, connection_index: int) -> int:
        if isinstance(connection_index, bool) or not isinstance(connection_index, int) or connection_index < 0:
            raise ValueError("connection_index must be a non-negative integer")
        if self._connections and connection_index >= len(self._connections):
            raise IndexError(f"connection index {connection_index} is outside {len(self._connections)} connections")
        return connection_index % self.worlds

    def create_connections(self, count: int) -> list[WasmtimeMudConnection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("provider is closed")
            if self._connections:
                raise RuntimeError("provider has already created connections")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("count must be a positive integer")
            # Commit the topology only after validation so a refused batch can be retried.
            resolved_worlds = count if self.worlds is None else self.worlds
            validate_seed(self.seed + resolved_worlds - 1, label="seed plus the final world index")
            if resolved_worlds > count:
                raise ValueError(f"worlds cannot exceed connections: {resolved_worlds} > {count}")
            largest_world_session_count = -(-count // resolved_worlds)
            validate_session_count(largest_world_session_count)
            self.worlds = resolved_worlds
            self._world_seeds = tuple(self.seed + world_index for world_index in range(resolved_worlds))
            self._world_slots = [list(range(index, count, resolved_worlds)) for index in range(resolved_worlds)]
            self._connections = [
                WasmtimeMudConnection(provider=self, connection_index=index, timeout_ms=self.timeout_ms)
                for index in range(count)
            ]
            self._executor = ThreadPoolExecutor(
                max_workers=min(resolved_worlds, os.process_cpu_count() or 1),
                thread_name_prefix="wasmtime-world",
            )
            return list(self._connections)

    def _seeds_for_reset(self, seed: int | list[int | None] | None) -> tuple[int, ...]:
        if seed is None:
            return self._world_seeds
        if isinstance(seed, int) and not isinstance(seed, bool):
            return tuple(
                validate_seed(seed + world_index, label=f"seed for world {world_index}")
                for world_index in range(self.worlds)
            )
        if not isinstance(seed, list):
            raise TypeError("provider reset seed must be an integer, a list of integers/None, or None")
        if len(seed) != len(self._connections):
            raise ValueError(
                f"provider reset seed list must contain one entry per connection: "
                f"expected {len(self._connections)}, got {len(seed)}"
            )
        # Validate every connection's seed before building worlds, including players sharing a world.
        for connection_index, candidate in enumerate(seed):
            if candidate is not None:
                validate_seed(candidate, label=f"seed[{connection_index}]")
        world_seeds = []
        for world_index in range(self.worlds):
            candidate = seed[world_index]
            world_seeds.append(self._world_seeds[world_index] if candidate is None else candidate)
        return tuple(world_seeds)

    def _build_world_with_sessions(self, count: int, seed: int) -> tuple[WasmtimeWorld, list[WasmtimeSession]]:
        world = self.runtime.create_world(max_players=count, seed=seed, civil_time_anchor=self.civil_time_anchor)
        try:
            sessions = [world.add_session(PERSONA_NAMES[local_index], self.timeout_ms) for local_index in range(count)]
            return world, sessions
        except BaseException as error:
            if cleanup_errors := _shutdown_worlds([world]):
                raise BaseExceptionGroup("WASI session admission failed", [error, *cleanup_errors]) from error
            raise

    def _require_world_connections_quiescent(self, world_index: int) -> None:
        unread_connections = [
            connection_index
            for connection_index in self._world_slots[world_index]
            if self._connections[connection_index]._pending_responses
        ]
        if unread_connections:
            raise RuntimeError(
                f"cannot reset WASI world {world_index} while connections {unread_connections} have unread responses"
            )

    def _replace_connection_session(
        self,
        connection: WasmtimeMudConnection,
        world_index: int,
    ) -> None:
        """Replace one departed player without changing its active world."""
        self._require_world_connections_quiescent(world_index)
        world = self._ordered_worlds[world_index]
        previous_session = connection._require_session()
        if not previous_session.departed:
            world.remove_session(previous_session, connection.timeout_ms)
        new_session = world.add_session(previous_session.persona_name, self.timeout_ms)
        connection._bind(new_session)

    def _replace_world(self, world_index: int, *, seed: int | None = None) -> None:
        """Replace one Store and Instance without rebinding any other world."""
        self._require_world_connections_quiescent(world_index)
        slots = self._world_slots[world_index]
        seed = self._world_seeds[world_index] if seed is None else validate_seed(seed, label="world seed")
        new_world, sessions = self._build_world_with_sessions(len(slots), seed)
        old_world = self._ordered_worlds[world_index]
        for connection_index, session in zip(slots, sessions, strict=True):
            self._connections[connection_index]._bind(session)
        self._ordered_worlds[world_index] = new_world
        self._world_seeds = tuple(
            seed if index == world_index else previous for index, previous in enumerate(self._world_seeds)
        )
        old_world.shutdown()

    def reset(self, *, seed: int | list[int | None] | None = None) -> None:
        """Replace every Store and Instance, then rebind the stable connections."""
        with self._lock:
            if self._closed:
                raise RuntimeError("provider is closed")
            if self._executor is None:
                raise RuntimeError("cannot reset before creating connections")
            if any(connection._pending_responses for connection in self._connections):
                raise RuntimeError("cannot reset Wasmtime worlds while a connection has unread responses")
            world_seeds = self._seeds_for_reset(seed)
            futures = []
            try:
                built = ordered_future_results(
                    (
                        self._executor.submit(self._build_world_with_sessions, len(slots), world_seeds[world_index])
                        for world_index, slots in enumerate(self._world_slots)
                    ),
                    "WASI world reset failed",
                    submitted=futures,
                )
            except BaseException as error:
                # The collector has settled every future, including after a main-thread interruption.
                new_worlds = (
                    future.result()[0] for future in futures if not future.cancelled() and future.exception() is None
                )
                if cleanup_errors := _shutdown_worlds(new_worlds):
                    raise BaseExceptionGroup("WASI world reset failed", [error, *cleanup_errors]) from error
                raise

            new_worlds = [world for world, _ in built]
            for world_index, (_, sessions) in enumerate(built):
                for connection_index, session in zip(self._world_slots[world_index], sessions, strict=True):
                    self._connections[connection_index]._bind(session)

            old_worlds = self._ordered_worlds
            self._ordered_worlds = new_worlds
            self._world_seeds = world_seeds
            if shutdown_errors := _shutdown_worlds(old_worlds):
                raise BaseExceptionGroup("previous WASI world shutdown failed after reset", shutdown_errors)

    def _prepare_connection(self, connection: WasmtimeMudConnection, *, seed: int | None = None) -> None:
        with self._lock:
            if connection not in self._connections:
                raise RuntimeError("connection does not belong to this provider")
            if not self._ordered_worlds:
                self.reset(seed=seed)
                return
            if not connection._started:
                return

            world_index = self.world_for_connection(connection.connection_index)
            if seed is not None:
                if len(self._world_slots[world_index]) > 1:
                    raise ValueError("Cannot reseed one player in a shared world. Reset the whole provider instead.")
                self._replace_world(world_index, seed=seed)
                return
            world = self._ordered_worlds[world_index]
            session = connection._require_session()
            if world.is_alive() and (
                len(self._world_slots[world_index]) > 1 or connection._invalidated or session.departed
            ):
                self._replace_connection_session(connection, world_index)
                return
            self._replace_world(world_index)

    def advance_worlds(self, ticks: int) -> dict[int, int]:
        """Advance every world once, reporting a world that ended structurally.

        Healthy worlds still advance when a sibling ends. Terminal worlds are reported together as
        ``WorldAdvancementFailed`` so the step hook can leave their final transitions for observation.
        """

        def tick(world: WasmtimeWorld) -> int | WasmtimeTerminalTick:
            try:
                return world.tick(ticks, self.timeout_ms)
            except WasmtimeTerminalTick as terminal:
                return terminal

        with self._lock:
            if self._closed:
                raise RuntimeError("provider is closed")
            if not self._ordered_worlds or self._executor is None:
                raise RuntimeError("cannot advance worlds before reset")
            futures = (self._executor.submit(tick, world) for world in self._ordered_worlds)
            results = ordered_future_results(futures, "WASI world advancement failed")
            successes: dict[int, int] = {}
            failures: dict[int, WasmtimeTerminalTick] = {}
            for world_index, result in enumerate(results):
                if isinstance(result, WasmtimeTerminalTick):
                    failures[world_index] = result
                else:
                    successes[world_index] = result
            if failures:
                raise WorldAdvancementFailed(
                    f"world advancement failed for {len(failures)} of {len(results)} worlds "
                    f"({len(successes)} advanced)",
                    successes=successes,
                    failures=failures,
                )
            return successes

    def tick_for_step(self) -> None:
        # Terminal worlds retain their observations while healthy siblings still advance.
        with suppress(WorldAdvancementFailed):
            self.advance_worlds(1)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors = _shutdown_worlds(self._ordered_worlds)
            self._ordered_worlds = []
            for connection in self._connections:
                connection._mark_closed()
            self._connections = []
            self._world_slots = []
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            if errors:
                raise BaseExceptionGroup("WASI provider close failed", errors)


def create_connection(**provider_options: Any) -> WasmtimeMudConnection:
    """Create a standalone connection that owns and closes its entire provider."""
    provider = WasmtimeProvider(**provider_options)
    try:
        connection = provider.create_connections(1)[0]
        connection.owns_provider = True
        return connection
    except BaseException:
        provider.close()
        raise
