import re
from collections.abc import Callable
from typing import Any

import pexpect

from mudgym.connections.prompts import PromptSpec, State
from mudgym.connections.state_machine import ConnectionState
from mudgym.logs import get_logger

logger = get_logger(__name__)


class MudConnection:
    """
    Base class for managing connections to MUD2 game instances.

    The state machine handles the fiddly lifecycle. This class wraps it so we can swap in different
    transports.

    Lifecycle:
    - ``reset()`` gets us to ``TEA_SIPPED``, ready for an episode to begin. The env's reset then
      exits the tearoom to start that episode.
    - ``send_line()`` and ``read_response()`` split sending from receiving so several players can
      see each other's actions.
    - ``close()`` terminates the child process. Use ``reset()`` instead when the connection will be
      reused for another episode.
    """

    # initial prompt we expect to see - subclasses can override
    initial_prompt: PromptSpec | None = None

    def __init__(
        self,
        *,
        account_id: str = "",
        password: str = "",
        persona_slot: int | None = None,
        db_slot: int | None = None,
        name_generator: Callable[[], str] | None = None,
    ):
        # I'm not convinced we actually need many of these anymore, but they're here for now.
        self.account_id = account_id
        self.password = password
        self.persona_slot = persona_slot
        self.db_slot = db_slot
        self.name_generator = name_generator

        # our state machine instance, that does most of the heavy lifting
        self.sm: ConnectionState | None = None
        self._pending_lines: list[str] = []

    def spawn(self) -> pexpect.spawn:
        """
        The method that does the actual connecting by spawning and returning our child process.
        """
        return pexpect.spawn(
            self.command[0],
            self.command[1:] if len(self.command) > 1 else [],
            encoding=None,
            use_poll=True,  # poll() instead of select() to avoid FD_SETSIZE limit
        )

    def reset(self) -> None:
        """
        Resets the connection to be ready to start a new episode (TEA_SIPPED state).

        This tells us the `MudConnection` is ready but the `MudEnv` has its own `reset()` steps afterwards that does
        episode related things that don't make sense here, like issuing commands to set up the initial environment state
        (eg, score), running observation commands, and taking the northwards step outside of the tearoom.

        I can imagine a situation with multiple `MudConnection`s waiting on each other after `reset()` to be ready so it
        seemed negligent to leave agents hanging around outside of the sanctity of the Tearoom where they might get
        attacked by mobiles or something.
        """

        self._pending_lines.clear()

        # do we need to respawn the process or can we reuse via some menu choices?
        needs_respawn = self.sm is None or not self.sm.isalive()
        logger.debug(
            "connection.reset.start",
            sm_state=self.sm.state.name if self.sm is not None else None,
            needs_respawn=needs_respawn,
        )

        if self.sm is not None and self.sm.isalive():
            # reset-quit: leave The Land but stay in the mudlogin menu if our connection type
            # supports that (ie, not a quicklogin, which exits to DEAD)
            self.sm.quit()

            if self.sm.state == State.DEAD:
                needs_respawn = True

        if needs_respawn:
            logger.debug("connection.reset.spawn")
            child = self.spawn()
            self.sm = ConnectionState(
                child=child,
                account_id=self.account_id,
                password=self.password,
                persona_slot=self.persona_slot,
                db_slot=self.db_slot,
                name_generator=self.name_generator,
                initial_prompt=self.initial_prompt,
            )

        # OPTION -> persona selection/creation -> TEAROOM -> sip tea -> TEA_SIPPED
        logger.debug("connection.reset.continue_until_tea", sm_state=self.sm.state.name)
        self.sm.continue_until(State.TEA_SIPPED)
        logger.debug(
            "connection.reset.complete",
            sm_state=self.sm.state.name,
            last_prompt=self.sm.last_prompt.name if self.sm.last_prompt else None,
        )

    def send_line(self, line: str) -> None:
        """Send a line without waiting for its response."""
        if self.sm is None:
            raise RuntimeError("Connection has not been reset, call reset() first.")
        self.sm.send(line)
        # only lines that made it onto the wire belong to the response we drain later
        self._pending_lines.append(line)

    def read_response(
        self,
        end_of_turn_marker: re.Pattern,
    ) -> tuple[bytes, bool, bool, dict[str, Any]]:
        """Read the response up to the marker for lines already sent through ``send_line``.

        The connection remembers which lines were actually sent, which lets the state machine find
        their echoes without asking the caller to reconstruct the wire history afterwards.
        """
        if self.sm is None:
            raise RuntimeError("Connection has not been reset, call reset() first.")
        if not self._pending_lines:
            raise RuntimeError("No command lines are awaiting a response.")

        lines = self._pending_lines
        self._pending_lines = []
        raw_bytes, terminated, incomplete, debug_info = self.sm.read(lines, end_of_turn_marker)
        debug_info["wire_lines"] = list(lines)
        return raw_bytes, terminated, incomplete, debug_info

    def invalidate(self) -> None:
        """Throw away a desynchronised transport so the next reset has to spawn a fresh one."""
        self._pending_lines.clear()
        if self.sm is None:
            return
        state_machine = self.sm
        try:
            if state_machine.isalive():
                state_machine.quit()
        except RuntimeError as exc:
            logger.debug("connection.invalidate.quit_failed", exc_info=exc)
        finally:
            state_machine.close(force=True)
            self.sm = None

    def close(self):
        self._pending_lines.clear()
        if self.sm is None:
            return
        try:
            if self.sm.isalive():
                self.sm.quit()
        finally:
            self.sm.close()
            self.sm = None
