"""Wasmtime execution of the separately packaged MUD2 WASI engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from wasmtime import Config, Engine, Linker, Module, Store, WasiConfig

from mudgym.featurizers.strings import encode_command_bytes

from .engine_contract import CommandTicketStatus, SessionResultCode, WorldStatus, validate_seed, validate_session_count

PACKAGED_WASI_MODULE = "wasi/mud2-wasi.wasm"
EMBEDDED_ASSETS_VERSION = 1


def _resolve_wasm_path(given: Path | None) -> Path:
    """Use an explicit artifact path or the installed binary package, never a checkout."""
    if given is not None:
        resolved = Path(given).expanduser().resolve()
    else:
        package = resources.files("mudgym_wasm_engine")
        resolved = Path(str(package.joinpath(PACKAGED_WASI_MODULE)))
    if not resolved.is_file():
        raise FileNotFoundError(
            f"the WASI engine module not found at {resolved}. Install a complete mudgym-wasm-engine wheel."
        )
    return resolved


class WasmtimeTerminalTick(RuntimeError):
    """A world ended during advancement, carrying the engine's own account."""

    def __init__(self, *, status: int, requested: int, completed: int, current: int):
        super().__init__(
            f"shared world reached a terminal status {status} after {completed}/{requested} cycles at tick {current}"
        )
        self.tick_failure = {
            "status": status,
            "requested_ticks": requested,
            "completed_ticks": completed,
            "current_tick": current,
        }


def _engine_error(kind: str, code: int | None, message: str) -> RuntimeError:
    code_text = "" if code is None else f" code={code}"
    return RuntimeError(f"MGERROR[{kind}{code_text}] {message}")


@dataclass(frozen=True, slots=True)
class WasmtimeTicket:
    status: CommandTicketStatus
    output: bytes
    # FAILED covers both command errors and indeterminate worlds. Retain the native return's distinction.
    world_indeterminate: bool = False


class WasmtimeSession:
    """One generation-safe player handle within a :class:`WasmtimeWorld`."""

    def __init__(self, world: WasmtimeWorld, player_id: int, generation: int, persona_name: str) -> None:
        self.world = world
        self.player_id = player_id
        self.generation = generation
        self.persona_name = persona_name
        self.departed = False

    def send(self, command: str, timeout_ms: int) -> WasmtimeTicket:
        with self.world.lock:
            result = self.world._call_with_text(
                "mud2_shared_step_session",
                command,
                arguments_before_text=(self.player_id, float(self.generation)),
                arguments_after_text=(timeout_ms,),
            )
            self.world._raise_if_handle_gone(result, self, "send")
            status_value = int(self.world._call("mud2_shared_ticket_status"))
            try:
                status = CommandTicketStatus(status_value)
            except ValueError as error:
                raise _engine_error(
                    "step-failed", status_value, f"unknown ticket status for player {self.player_id}"
                ) from error
            if result == SessionResultCode.OUTPUT_OVERFLOW:
                raise _engine_error("output-overflow", result, f"output overflowed for player {self.player_id}")
            if result == SessionResultCode.PLAYER_GONE:
                self.departed = True
                output = b""
            elif result == SessionResultCode.GAME_HALTED:
                status = CommandTicketStatus.HALTED
                output = self.receive()
            elif result == SessionResultCode.INPUT_REQUIRED:
                output = self.receive()
            elif result == SessionResultCode.GAME_INDETERMINATE:
                # Preserve the transition for truncation. This return carries no trustworthy output.
                output = b""
            elif result < 0:
                raise _engine_error("step-failed", result, f"session step failed for player {self.player_id}")
            else:
                output = self.world._read_output(result)
            return WasmtimeTicket(
                status=status,
                output=output,
                world_indeterminate=result == SessionResultCode.GAME_INDETERMINATE,
            )

    def receive(self) -> bytes:
        with self.world.lock:
            result = int(self.world._call("mud2_shared_receive_session", self.player_id, float(self.generation)))
            departed = bool(self.world._call("mud2_shared_receive_session_departed"))
            self.world._raise_if_handle_gone(result, self, "receive")
            if result == SessionResultCode.PLAYER_GONE:
                self.departed = True
                return b""
            if result == SessionResultCode.OUTPUT_OVERFLOW:
                raise _engine_error("output-overflow", result, f"receive overflowed for player {self.player_id}")
            if result == SessionResultCode.GAME_INDETERMINATE:
                # Same rule as send: the caller learns from the world's status,
                # not from an exception unwinding a Gym transition.
                return b""
            if result < 0:
                raise _engine_error("receive-failed", result, f"receive failed for player {self.player_id}")
            self.departed = departed
            return self.world._read_output(result)


class WasmtimeWorld:
    """One isolated Store, Instance, linear memory and shared MUD world."""

    def __init__(
        self,
        runtime: WasmtimeRuntime,
        *,
        max_players: int,
        seed: int,
        civil_time_anchor: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    ) -> None:
        validate_session_count(max_players)
        validate_seed(seed, label="world seed")
        offset = civil_time_anchor.utcoffset()
        if offset is None:
            raise ValueError("civil_time_anchor must include a UTC offset")
        if civil_time_anchor.microsecond or offset.microseconds:
            raise ValueError("civil_time_anchor and its UTC offset must have whole-second precision")
        unix_seconds = civil_time_anchor.timestamp()
        utc_offset_seconds = int(offset.total_seconds())

        self.lock = threading.RLock()
        self.store = Store(runtime.engine)
        wasi = WasiConfig()
        wasi.inherit_stderr()
        self.store.set_wasi(wasi)
        linker = Linker(runtime.engine)
        linker.define_wasi()
        instance = linker.instantiate(self.store, runtime.module)
        self.exports = instance.exports(self.store)
        self.memory = self.exports["memory"]
        self._call("_initialize")
        if self._call("mud2_embedded_assets_version") != EMBEDDED_ASSETS_VERSION:
            raise RuntimeError("Unsupported embedded asset version. Install a compatible mudgym-wasm-engine wheel.")
        if self._call("mud2_seed_world", float(seed)) != 1:
            raise RuntimeError(f"WASI engine refused world seed {seed}")
        if self._call("mud2_shared_init_with_options", max_players, 0, unix_seconds, utc_offset_seconds, 1) != 1:
            raise RuntimeError("WASI shared world refused to start")
        self.closed = False

    def _call(self, name: str, *arguments: Any) -> Any:
        return self.exports[name](self.store, *arguments)

    def _call_with_text(
        self,
        export_name: str,
        text: str,
        *,
        arguments_before_text: tuple[Any, ...] = (),
        arguments_after_text: tuple[Any, ...] = (),
    ) -> int:
        encoded = encode_command_bytes(text) + b"\0"
        pointer = int(self._call("malloc", len(encoded)))
        if pointer == 0:
            raise RuntimeError(f"WASI allocation failed for {export_name}")
        try:
            self.memory.write(self.store, encoded, pointer)
            return int(self._call(export_name, *arguments_before_text, pointer, *arguments_after_text))
        finally:
            self._call("free", pointer)

    def _read_output(self, length: int) -> bytes:
        if length == 0:
            return b""
        pointer = int(self._call("mud2_shared_output"))
        output = bytes(self.memory.read(self.store, pointer, pointer + length))
        if len(output) != length:
            raise RuntimeError(f"WASI engine reported {length} output bytes but exposed {len(output)}")
        return output

    def _raise_if_handle_gone(self, result: int, session: WasmtimeSession, operation: str) -> None:
        if result == SessionResultCode.WORLD_GONE:
            session.departed = True
            raise _engine_error("world-gone", result, f"session world was replaced during {operation}")
        if result == SessionResultCode.SESSION_GONE:
            session.departed = True
            raise _engine_error("session-gone", result, f"session no longer exists during {operation}")

    def add_session(self, persona_name: str, timeout_ms: int = 5_000) -> WasmtimeSession:
        with self.lock:
            player_id = self._call_with_text(
                "mud2_shared_add_session",
                persona_name,
                # The ABI requires these fields. Zero selects the engine's default mortal persona.
                arguments_after_text=(0, 0, 0, 0, 0, 0, 0, 0, timeout_ms),
            )
            if player_id == -2:
                raise _engine_error("persona-collision", player_id, f"persona name is reserved: {persona_name}")
            if player_id < 0:
                raise _engine_error("add-player-failed", player_id, f"could not add persona {persona_name}")
            generation = int(self._call("mud2_shared_last_session_generation"))
            if generation < 1:
                raise _engine_error("add-player-failed", None, f"persona {persona_name} has no generation")
            return WasmtimeSession(self, player_id, generation, persona_name)

    def remove_session(self, session: WasmtimeSession, timeout_ms: int) -> None:
        with self.lock:
            result = int(
                self._call("mud2_shared_remove_session", session.player_id, float(session.generation), timeout_ms)
            )
            self._raise_if_handle_gone(result, session, "remove")
            if result != 0:
                raise _engine_error("remove-player-failed", result, f"could not remove player {session.player_id}")
            session.departed = True

    def tick(self, count: int, timeout_ms: int = 5_000) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("tick count must be a non-negative integer")
        with self.lock:
            if count == 0:
                return self.current_tick()
            status = int(self._call("mud2_shared_tick", count, timeout_ms))
            completed = int(self._call("mud2_shared_tick_completed"))
            current = int(self._call("mud2_shared_tick_current"))
            if status == 8:
                raise _engine_error("world-gone", status, "world was replaced before it could advance")
            if status != 0 and self.status() in {WorldStatus.HALTED, WorldStatus.INDETERMINATE}:
                # Report terminal status structurally so observation can preserve the final transition.
                raise WasmtimeTerminalTick(status=status, requested=count, completed=completed, current=current)
            if status != 0:
                raise _engine_error(
                    "tick-failed", status, f"shared tick stopped after {completed}/{count} cycles at tick {current}"
                )
            return current

    def current_tick(self) -> int:
        with self.lock:
            current = int(self._call("mud2_shared_current_tick"))
            if current < 0:
                raise _engine_error("current-tick-failed", current, "could not read the world tick count")
            return current

    def is_alive(self) -> bool:
        return self.status() is WorldStatus.ACTIVE

    def status(self) -> WorldStatus:
        with self.lock:
            return WorldStatus.SHUT_DOWN if self.closed else WorldStatus(int(self._call("mud2_shared_world_status")))

    def shutdown(self) -> None:
        with self.lock:
            if self.closed:
                return
            result = int(self._call("mud2_shared_shutdown"))
            self.closed = True
            if result not in {0, SessionResultCode.WORLD_GONE}:
                raise _engine_error("shutdown-failed", result, "could not shut down WASI shared world")


class WasmtimeRuntime:
    """Compile one WASI module, then instantiate any number of isolated worlds."""

    def __init__(
        self,
        *,
        wasm_path: Path | None = None,
    ) -> None:
        """Load the self-contained engine module from its wheel or an explicit module path."""
        self.wasm_path = _resolve_wasm_path(wasm_path)
        config = Config()
        config.wasm_exceptions = True
        self.engine = Engine(config)
        self.module = Module.from_file(self.engine, str(self.wasm_path))
        if "mud2_embedded_assets_version" not in {export.name for export in self.module.exports}:
            raise RuntimeError("The WASI module has no embedded assets. Install a compatible mudgym-wasm-engine wheel.")

    def create_world(
        self,
        *,
        max_players: int,
        seed: int,
        civil_time_anchor: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    ) -> WasmtimeWorld:
        return WasmtimeWorld(self, max_players=max_players, seed=seed, civil_time_anchor=civil_time_anchor)
