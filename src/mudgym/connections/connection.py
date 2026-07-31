import re
from collections.abc import Callable, Sequence
from typing import Any

import pexpect

from mudgym.connections.prompts import FINAL_COMMAND_MARKER, PromptSpec, State
from mudgym.connections.state_machine import ConnectionState
from mudgym.logs import get_logger

logger = get_logger(__name__)


class MudConnection:
    """
    Base class for managing connections to MUD2 instances.

    The connection lifecycle is managed by the state machine, this class wraps the state machine and allows for
    different types of transport whilst exposing a (hopefully) easier to deal with API.

    Lifecycle:
    - reset() -> Get us to the TEA_SIPPED state, ready for an episode to begin.
    - send_command(command: str) -> tuple[bytes, bool, bool, dict] -> Send a command to the game, receiving the
      raw response bytes, terminated and incomplete flags and some debug info.
    - close() -> Close the connection, terminating the child process. Typically you would use reset() instead if you
      are going to want to reuse the connection to start a new episode (for connections that support it).
    """

    # initial prompt we expect to see - subclasses can override
    initial_prompt: PromptSpec | None = None

    # the end of turn marker closing each step's read window: the pattern identifying the response of the batch's
    # final command. This class attribute is the protocol's one declared default (fei's ======== divider), which
    # keeps a bare connection usable on its own; the session overrides it per instance with whatever marker the
    # env's batch actually ends with.
    end_of_turn_marker: re.Pattern = FINAL_COMMAND_MARKER

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

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if the connection is available to use in the current environment. Subclasses can override this method to
        perform a check specific to the connection type. This is to help with experimenting with different connections
        types.
        """
        logger.debug("connection.is_available.default", connection_class=cls.__name__)
        return True

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
        (eg, score) and running start of episode autocommands and taking the northwards step outside of the tearoom.

        I can imagine a situation with multiple `MudConnection`s waiting on each other after `reset()` to be ready so it
        seemed negligent to leave agents hanging around outside of the sanctity of the Tearoom where they might get
        attacked by mobiles or something.
        """

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
                end_of_turn_marker=self.end_of_turn_marker,
            )

        # OPTION -> persona selection/creation -> TEAROOM -> sip tea -> TEA_SIPPED
        logger.debug("connection.reset.continue_until_tea", sm_state=self.sm.state.name)
        self.sm.continue_until(State.TEA_SIPPED)
        logger.debug(
            "connection.reset.complete",
            sm_state=self.sm.state.name,
            last_prompt=self.sm.last_prompt.name if self.sm.last_prompt else None,
        )

    def send_command(self, command: str | Sequence[str]) -> tuple[bytes, bool, bool, dict[str, Any]]:
        """
        Send a command batch (one wire line, or several when the caller split it) and return the
        raw response bytes, terminated/incomplete flags, and debug info.
        """
        if self.sm is None:
            raise RuntimeError("Connection has not been reset, call reset() first.")
        lines = [command] if isinstance(command, str) else list(command)
        if not lines:
            raise ValueError("send_command requires at least one command line; got an empty batch")
        return self.sm.send_command(lines)

    def close(self):
        if self.sm is None:
            return
        try:
            if self.sm.isalive():
                self.sm.quit()
        finally:
            self.sm.close()
            self.sm = None
