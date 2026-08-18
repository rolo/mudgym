import re
from collections import deque
from collections.abc import Callable, Collection, Sequence
from time import monotonic
from typing import Any

import pexpect

from mudgym.connections.errors import ConnectionClosedError
from mudgym.connections.prompts import (
    BROADCAST_PROMPTS,
    EXPECT_LIST,
    GAME_OVER_PROMPTS,
    INVALID_COMMAND_PROMPTS,
    PROMPTS,
    Prompt,
    PromptSpec,
    State,
    marker_up_to_next_prompt,
)
from mudgym.connections.termination import (
    END_OF_STEP_PATTERNS,
    NO_LONGER_IN_GAME_PROMPTS,
    TRANSPORT_BREAK_PROMPTS,
)
from mudgym.connections.transitions import MENU_ECHO_WINDOW_BYTES, TRANSITIONS, menu_answer_was_echoed
from mudgym.featurizers.strings import LINE_BREAK_RE, decode_wire_bytes, encode_command_bytes
from mudgym.logs import get_logger

# Fast lookup from the pattern pexpect matched back to our Prompt enum.
PROMPTS_BY_VALUE = {p.value: p for p in PROMPTS}

logger = get_logger(__name__)


class ConnectionState:
    """
    ConnectionState is the State Machine which handles where we are in the MUD2 software and handles access to the
    pexpect child.

    External modules shouldn't access the child directly, but rather should use the public methods here. This allows us
    to experiment with different ways of connecting to the game.

    If a higher-level module requires additional capabilities, add a new public method rather than calling expect()
    directly

    States vs Prompts

    State - where we are in the workflow - what is expected next.
    Prompt - something we matched from the wire that may cause a transition to a new state.
    """

    def __init__(
        self,
        child: pexpect.spawn | None = None,
        account_id: str | None = None,
        password: str | None = None,
        persona_slot: int | None = None,
        db_slot: int | None = None,
        name_generator: Callable[[], str] | None = None,
        initial_prompt: PromptSpec | None = None,
    ):
        self.child = child
        self.account_id = account_id or ""
        self.password = password or ""
        self.max_transition_steps = 15

        # a backstop, not flow control. broadcasts and the init retry loop are both free of the
        # step budget, and a slow database can spend thousands of matches legitimately, so only
        # elapsed time separates that from a session that will never move
        self.max_continue_seconds = 300.0

        self.default_persona_slot = persona_slot
        self.default_db_slot = db_slot
        self.default_name_generator = name_generator

        self.history = deque(maxlen=10)
        self.chunk_history = deque(maxlen=10)
        self.last_prompt: Prompt | None = None

        # the menu answer we have sent but not yet seen echoed back; guards against answering an
        # Option prompt redraw a second time (see transitions.send_db_slot)
        self.pending_menu_answer: bytes | None = None

        # what the game has sent since that answer - the echo can land in any match's chunk
        self.output_since_menu_answer = b""
        # latched, so bounding the window above cannot lose an echo we already saw
        self.menu_answer_echoed = False

        # we begin in the initial state and await the initial prompt
        self.state = State.INITIAL
        self.expect(initial_prompt or EXPECT_LIST, timeout=5.0)

    def isalive(self) -> bool:
        """Check if the child process is still alive.

        Returns:
            True if the child process is alive, False otherwise
        """
        return self.child is not None and self.child.isalive()

    def close(self, *, force: bool = True) -> None:
        """Close the underlying child process and mark this state machine dead.

        A session left at the mudlogin menu keeps the account logged in even if the child dies (eg, killing a docker
        exec client leaves the process running inside the container), and the next login would hit SUPERSEDE. So if
        we're at the menu, log out through the CLOSING transitions first.
        """
        logger.debug("sm.close.start", state=self.state.name, child_alive=self.isalive(), force=force)
        if self.state == State.OPTION and self.isalive():
            self.state = State.CLOSING
            try:
                self.continue_until(State.DEAD)
            except RuntimeError:
                # the EOF after "logged out." surfaces as a RuntimeError from expect()
                logger.debug("sm.close.logout_finished", state=self.state.name, child_alive=self.isalive())
        try:
            if self.child is not None:
                self.child.close(force=force)
        except (pexpect.ExceptionPexpect, OSError) as exc:
            logger.debug("sm.close.failed", state=self.state.name, exc_info=exc)
        finally:
            self.state = State.DEAD
            logger.debug("sm.close.complete", state=self.state.name)

    def send(self, data: str | bytes, add_cr: bool = True) -> None:
        """
        Send data to the child process, optionally adding a carriage return.
        """
        if not self.isalive():
            raise ConnectionClosedError("Cannot send data to closed child process")

        if isinstance(data, str):
            data = encode_command_bytes(data)

        if add_cr:
            data += b"\r"

        try:
            self.child.send(data)
        except OSError as exc:
            raise ConnectionClosedError("Child process closed while sending data") from exc
        logger.debug("sm.send", data=data.replace(b"\r", b"\\r"), state=self.state.name, add_cr=add_cr)

    def send_cr(self) -> None:
        """
        Send just a carriage return.
        """
        self.send(b"", add_cr=True)

    def get_buffer(self) -> bytes:
        """Get the last output buffer from the child process.

        Returns:
            The bytes content of the last output, or empty bytes if no output
        """
        if not self.isalive():
            return b""
        try:
            return self.child.before or b""
        except (ValueError, AttributeError):
            # Child may be closed, return empty
            return b""

    def expect(
        self,
        patterns: Prompt | Sequence[Any] = EXPECT_LIST,
        timeout: float = 3.0,
        raise_on_eof_timeout: bool = True,
    ) -> tuple[int, Prompt | None]:
        """Expect one of the given patterns and apply its state transition.

        Prefer using the state machine transitions and continue_until methods when possible.

        Args:
            patterns: Prompt enum, or a sequence of Prompt enums or raw patterns
            timeout: Maximum time to wait
            raise_on_eof_timeout: If True (default), raise RuntimeError on EOF/TIMEOUT,
                set to False when caller wants to handle these as normal results.

        Returns:
            tuple[int, Prompt | None]: index of matched pattern, matched Prompt enum or None
        """
        if isinstance(patterns, Prompt):
            patterns = [patterns]

        pattern_values = [pattern.value if isinstance(pattern, Prompt) else pattern for pattern in patterns]

        state_before = self.state
        logger.debug(
            "sm.expect.start",
            state=state_before.name,
            pattern_count=len(pattern_values),
            timeout=timeout,
            raise_on_eof_timeout=raise_on_eof_timeout,
        )
        matched_index = self.child.expect(pattern_values, timeout=timeout)
        matched_pattern = pattern_values[matched_index]

        # remember what each real match consumed so timeout/EOF errors can show what led up to them
        if matched_pattern not in (pexpect.TIMEOUT, pexpect.EOF):
            consumed = (self.child.before or b"") + (self.child.after if isinstance(self.child.after, bytes) else b"")
            if consumed:
                self.chunk_history.append(consumed)
                if self.pending_menu_answer is not None and not self.menu_answer_echoed:
                    recent = self.output_since_menu_answer + consumed
                    self.menu_answer_echoed = menu_answer_was_echoed(recent, self.pending_menu_answer)
                    self.output_since_menu_answer = recent[-MENU_ECHO_WINDOW_BYTES:]

        matched_prompt = PROMPTS_BY_VALUE.get(matched_pattern)

        self.last_prompt = matched_prompt
        logger.debug(
            "sm.expect.matched",
            state=state_before.name,
            matched_index=matched_index,
            matched_prompt=matched_prompt.name if matched_prompt else None,
        )

        # Check if we matched TIMEOUT or EOF exception classes directly
        if matched_pattern in (pexpect.TIMEOUT, pexpect.EOF) and raise_on_eof_timeout:
            error_type = "Timeout" if matched_pattern == pexpect.TIMEOUT else "EOF"
            logger.debug("sm.expect.raise", error_type=error_type.lower(), before=self.child.before or b"")

            # Build detailed debug info like continue_until does
            last_buffer = decode_wire_bytes(self.child.before or b"")[-500:]
            history = " → ".join(s.name for s in list(self.history)[-5:]) if self.history else "None"
            chunk_hist = [decode_wire_bytes(c)[-100:] for c in list(self.chunk_history)[-3:]]

            debug_parts = [
                f"\n{error_type} while expecting prompt",
                f"Current state: {self.state.name}",
                f"Last prompt: {self.last_prompt.name if self.last_prompt else 'None'}",
                f"State history: {history}",
                f"Last buffer: {last_buffer!r}",
            ]
            if chunk_hist:
                debug_parts.append(f"Recent chunks: {chunk_hist}")

            raise RuntimeError("\n".join(debug_parts))
        # otherwise the caller handles EOF/TIMEOUT as normal results; matched_prompt is already
        # Prompt.EOF/Prompt.TIMEOUT (their enum values are the pexpect exception classes, so PROMPTS_BY_VALUE mapped
        # them above).

        if matched_prompt:
            self.maybe_apply_transition(matched_prompt)

        return (matched_index, matched_prompt)

    def maybe_apply_transition(self, prompt: Prompt) -> None:
        """
        See if it's necessary to apply a state transition for the given Prompt.

        Args:
            prompt: The prompt that was matched.
        """
        # see if we have a transition for this Prompt in this State
        transition = TRANSITIONS[self.state].get(prompt)

        # no transition, nothing to see here - an unhandled match must not mutate anything
        if not transition:
            logger.debug("sm.transition.missing", state=self.state.name, prompt=prompt.name)
            return

        # a prompt we handle means the menu moved on, so the pending answer is done with.
        # broadcasts are the exception - RESETTING acts on them but they say nothing about our answer
        if prompt is not Prompt.OPTION and prompt not in BROADCAST_PROMPTS:
            self.pending_menu_answer = None
            self.output_since_menu_answer = b""
            self.menu_answer_echoed = False

        # we have a transition to apply.

        # hang on to our previous state so we can use it in logging.
        prev_state = self.state

        logger.debug(
            "sm.transition",
            prev_state=prev_state.name,
            new_state=transition.next_state.name,
            prompt=prompt.name,
            action=transition.action.__name__ if transition.action else None,
        )

        # set out new state
        self.state = transition.next_state

        # apply the action callable if there is one
        if transition.action:
            transition.action(self)

        # add to state history if the state has changed
        if self.state != prev_state:
            self.history.append(self.state)

    def continue_until(
        self,
        until: State | Collection[State],
        timeout: float = 3.0,
    ) -> Prompt:
        """
        Continue until the `until` state is reached or timeout occurs.

        Returns the last matched Prompt.
        """

        # until can be any collection
        if not isinstance(until, Collection):
            until = [until]

        # standardise to a list
        until = list(until)

        steps = 0
        last_prompt = None
        prev_state = None
        started = monotonic()
        while (
            self.state not in until
            and steps < self.max_transition_steps
            and monotonic() - started < self.max_continue_seconds
        ):
            prev_state = self.state

            # Use longer timeout after SESSION_DYING since the session takes time to die
            expect_timeout = 5.0 if last_prompt == Prompt.SESSION_DYING else timeout
            _, last_prompt = self.expect(timeout=expect_timeout)

            # only charge for what could be progress - init retries and broadcasts would each eat it
            in_init_retry = State.RESETTING in (self.state, prev_state)
            if not in_init_retry and last_prompt not in BROADCAST_PROMPTS:
                steps += 1

        if self.state not in until:
            # Decode and show debug information and recent state history for troubleshooting
            last_buffer = decode_wire_bytes(self.child.before or b"")[-1000:]
            history = " → ".join([f"{s.name}" for s in list(self.history)[-5:]]) if self.history else "None"
            logger.debug(
                "sm.continue_until.timeout",
                state=self.state.name,
                until=until,
                steps=steps,
                max_transition_steps=self.max_transition_steps,
                elapsed=monotonic() - started,
                max_continue_seconds=self.max_continue_seconds,
                before=self.child.before or b"",
                after=self.child.after,
            )

            debug_parts = []
            debug_parts.append(f"\nTimeout waiting for {until}")
            debug_parts.append(f"Current state: {self.state.name}")
            debug_parts.append(f"Last prompt: {last_prompt.name if last_prompt else 'None'}")
            debug_parts.append(f"Recent state history: {history}")
            debug_parts.append(f"steps_taken: {steps}/{self.max_transition_steps}")
            debug_parts.append(f"elapsed: {monotonic() - started:.1f}s/{self.max_continue_seconds}s")
            debug_parts.append("Recent buffer output:")
            debug_parts.append(f"  {last_buffer!r}")
            error_msg = "\n".join(debug_parts)

            raise RuntimeError(error_msg)

        return last_prompt

    def read(
        self,
        lines: Sequence[str],
        end_of_turn_marker: re.Pattern,
    ) -> tuple[bytes, bool, bool, dict[str, Any]]:
        """Read a response for command lines that have already been sent.

        The lines were sent separately so a multi agent environment could let every player act first.
        """
        buffer = bytearray()
        terminated = False
        incomplete = False
        rejected = False
        marker_arrived = False
        prompt_enum: Prompt | None = None
        seen_any_command_echo = False
        seen_final_command_echo = False

        # Every sent line's echo is a trust boundary. Put echo patterns first so a command such as
        # "Option:" or "Cheerio!" is treated as our input rather than a real game prompt. The final
        # line is the observation command, so its echo tells us the next marker is ours.
        echo_patterns = [
            re.compile(re.escape(encode_command_bytes(line)) + rb"(?:" + LINE_BREAK_RE + rb")") for line in lines
        ]
        end_of_turn_marker_pattern = marker_up_to_next_prompt(end_of_turn_marker)
        end_of_step_patterns = echo_patterns + END_OF_STEP_PATTERNS + [end_of_turn_marker_pattern]

        while True:
            idx, prompt_enum = self.expect(end_of_step_patterns, raise_on_eof_timeout=False)
            matched_pattern = end_of_step_patterns[idx]

            # Output from between steps can arrive before or after our echo. Either way it happened in this read window, so keep it in the bytes returned to the environment.
            buffer += self.child.before or b""

            # A menu or login prompt means we have left the game and our marker is not coming. These are the only matched prompts excluded from the returned game bytes.
            if prompt_enum in NO_LONGER_IN_GAME_PROMPTS:
                logger.warning(
                    "sm.read.left_game",
                    state=self.state.name,
                    matched_prompt=prompt_enum.name if prompt_enum else None,
                )
                incomplete = True
                break

            # Everything else is part of the response. TIMEOUT and EOF are exception classes rather than bytes, so naturally add nothing here.
            if isinstance(self.child.after, bytes):
                buffer += self.child.after

            if idx < len(echo_patterns):
                seen_any_command_echo = True
                if idx == len(echo_patterns) - 1:
                    seen_final_command_echo = True
                continue

            # A marker only belongs to us after the final echo. One seen earlier is late output from an old, desynchronised window, so keep it as ordinary pre-echo content and carry on.
            if matched_pattern is end_of_turn_marker_pattern and seen_final_command_echo:
                marker_arrived = True
                break

            # Echoes can run ahead of responses, so keep reading for the marker.
            command_rejected = matched_pattern in INVALID_COMMAND_PROMPTS and seen_any_command_echo
            rejected = rejected or command_rejected
            if prompt_enum in GAME_OVER_PROMPTS or prompt_enum in TRANSPORT_BREAK_PROMPTS:
                terminated = prompt_enum in GAME_OVER_PROMPTS
                incomplete = prompt_enum in TRANSPORT_BREAK_PROMPTS
                if incomplete and seen_final_command_echo:
                    logger.warning(
                        "sm.read.marker_missing",
                        state=self.state.name,
                        hint="no end of turn marker before the read window closed",
                    )
                break

        debug_info: dict[str, Any] = {
            "matched_prompt": prompt_enum.name if prompt_enum else None,
            "rejected": rejected,
            "marker_arrived": marker_arrived,
        }
        logger.debug(
            "sm.read.complete",
            state=self.state.name,
            matched_prompt=debug_info["matched_prompt"],
            terminated=terminated,
            incomplete=incomplete,
        )

        return bytes(buffer), terminated, incomplete, debug_info

    def quit(self) -> None:
        """Quit the game cleanly.

        For quicklogin binaries, the process will exit immediately after qq.
        For traditional binaries, this will return us to the OPTION prompt.

        After quit(), either:
        - state is DEAD/OPTION and process is alive (traditional)
        - process is dead / isalive() is False (quicklogin EOF)
        """

        if self.state == State.GAME_OVER:
            # The game-over prompt has already moved the state machine on. There is nothing useful left to send, just follow the remaining menu/EOF transition if the process is alive.
            if not self.isalive():
                self.state = State.DEAD
                return
            try:
                self.continue_until([State.DEAD, State.OPTION])
            except RuntimeError:
                if not self.isalive():
                    self.state = State.DEAD
                    return
                raise
            return

        if self.state in [State.GAME, State.TEAROOM, State.TEA_SIPPED]:
            if not self.isalive():
                # Process already gone - nothing to send qq to.
                self.state = State.DEAD
                return

            # mgquit will allow us to quit even if asleep or unconscious
            self.send(b"mgquit")

            # quicklogin binaries exit after quit (EOF). Regular ones reach OPTION.
            try:
                self.continue_until([State.DEAD, State.OPTION])
            except RuntimeError:
                if not self.isalive():
                    self.state = State.DEAD
                    return
                raise
