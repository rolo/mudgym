import re
from collections import deque
from collections.abc import Callable, Collection, Sequence
from typing import Any

import pexpect

from mudgym.connections.prompts import (
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
from mudgym.connections.transitions import TRANSITIONS
from mudgym.featurizers.strings import LINE_BREAK_RE, decode_wire_bytes, encode_command_bytes
from mudgym.logs import get_logger

# Fast lookup from matched pattern bytes -> Prompt enum
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
        *,
        end_of_turn_marker: re.Pattern,
    ):
        self.child = child
        self.account_id = account_id or ""
        self.password = password or ""
        self.max_transition_steps = 15

        # the wire form of the end of turn marker, which closes send_command's read window
        self.end_of_turn_marker_pattern = marker_up_to_next_prompt(end_of_turn_marker)

        self.default_persona_slot = persona_slot
        self.default_db_slot = db_slot
        self.default_name_generator = name_generator

        self.history = deque(maxlen=10)
        self.chunk_history = deque(maxlen=10)
        self.last_prompt: Prompt | None = None

        # the menu answer we have sent but not yet seen echoed back; guards against answering an
        # Option prompt redraw a second time (see transitions.send_db_slot)
        self.pending_menu_answer: str | None = None

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
            raise RuntimeError("Cannot send data to closed child process")

        if isinstance(data, str):
            data = encode_command_bytes(data)

        if add_cr:
            data += b"\r"

        self.child.send(data)
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

    def clear_buffer(self) -> None:
        """Clear the output buffer from the child process.

        This is useful after quit/reset operations to ensure no stale data
        remains in the buffer for the next operation.
        """
        if not self.isalive():
            return
        try:
            self.child.before = b""
            self.child.after = b""
        except (ValueError, AttributeError):
            # Child may be closed, ignore
            pass

    def expect(
        self,
        patterns: Prompt | list[Prompt] | list = EXPECT_LIST,
        timeout: float = 3.0,
        raise_on_eof_timeout: bool = True,
    ) -> tuple[int, Prompt | None]:
        """Expect one of the given patterns and apply its state transition.

        Prefer using the state machine transitions and continue_until methods when possible.

        Args:
            patterns: Prompt enum, list of Prompt enums, or list of raw patterns
            timeout: Maximum time to wait
            raise_on_eof_timeout: If True (default), raise RuntimeError on EOF/TIMEOUT,
                set to False when caller wants to handle these as normal results.

        Returns:
            tuple[int, Prompt | None]: index of matched pattern, matched Prompt enum or None
        """
        # standardise patterns to a list
        if isinstance(patterns, (Prompt, State)):
            patterns = [patterns]

        # convert patterns to values for pexpect.expect()
        # handle both Prompt/State enums (which have .value) and raw values
        pattern_values = []
        pattern_to_enum = {}  # map from value to Prompt enum for return value

        for p in patterns:
            if isinstance(p, (Prompt, State)):
                value = p.value
                pattern_values.append(value)
                if isinstance(p, Prompt):
                    pattern_to_enum[value] = p
            else:
                # already a raw value (exception class, bytes, regex, etc.)
                pattern_values.append(p)
                # try to find matching Prompt enum by value
                if p in PROMPTS_BY_VALUE:
                    pattern_to_enum[p] = PROMPTS_BY_VALUE[p]

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

        # map back to Prompt enum if possible
        matched_prompt = pattern_to_enum.get(matched_pattern)

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
        # any prompt other than the Option menu means the menu moved on, so a pending menu answer
        # is no longer outstanding
        if prompt is not Prompt.OPTION:
            self.pending_menu_answer = None

        # see if we have a transition for this Prompt in this State
        transition = TRANSITIONS[self.state].get(prompt)

        # no transition, nothing to see here
        if not transition:
            logger.debug("sm.transition.missing", state=self.state.name, prompt=prompt.name)
            return

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
        while self.state not in until and steps < self.max_transition_steps:
            # Don't count steps during initialization retry loops (RESETTING <-> OPTION)
            # This prevents tight retry loops from burning through the step limit
            in_init_retry = self.state == State.RESETTING or prev_state == State.RESETTING
            if not in_init_retry:
                steps += 1
            prev_state = self.state

            # Use longer timeout after SESSION_DYING since the session takes time to die
            expect_timeout = 5.0 if last_prompt == Prompt.SESSION_DYING else timeout
            idx, matched = self.expect(timeout=expect_timeout)

            # Update last_prompt with the matched Prompt (or keep existing if None)
            last_prompt = matched
            prompt_enum = matched

            before_buf = self.child.before or b""
            if before_buf:
                logger.debug(
                    "sm.continue.before",
                    before=before_buf.replace(b"\r", b"\\r"),
                    state=self.state.name,
                )
            after_buf = self.child.after
            if after_buf:
                after_str = after_buf.replace(b"\r", b"\\r") if isinstance(after_buf, bytes) else after_buf
                logger.debug(
                    "sm.continue.after",
                    after=after_str,
                    state=self.state.name,
                )
            prompt_name = prompt_enum.name if prompt_enum else "unknown"
            logger.debug(
                "sm.continue.matched",
                prompt=prompt_name,
                pattern=EXPECT_LIST[idx],
                state=self.state.name,
            )

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
                before=self.child.before or b"",
                after=self.child.after,
            )

            debug_parts = []
            debug_parts.append(f"\nTimeout waiting for {until}")
            debug_parts.append(f"Current state: {self.state.name}")
            debug_parts.append(f"Last prompt: {last_prompt.name if last_prompt else 'None'}")
            debug_parts.append(f"Recent state history: {history}")
            debug_parts.append(f"steps_taken: {steps}/{self.max_transition_steps}")
            debug_parts.append("Recent buffer output:")
            debug_parts.append(f"  {last_buffer!r}")
            error_msg = "\n".join(debug_parts)

            raise RuntimeError(error_msg)

        return last_prompt

    def send_command(
        self,
        command: str | Sequence[str],
    ) -> tuple[bytes, bool, bool, dict[str, Any]]:
        """
        Send a command batch and read output until we consider its response complete.

        The batch is one wire line, or several when the caller split it (speech commands take the
        auto commands on a separate line so they aren't spoken). The read anchors on the echo of
        the final line - the one whose batch ends with the marker command - then completes when we
        see the first end of turn marker. Either that marker, a game over prompt, a prompt showing
        we are no longer in the game, transport EOF/TIMEOUT, or a command rejection that aborts
        the rest of the final line.

        Returns the raw bytes including command echoes, prompt markers and so on. Splitting and
        interpretation is left to the caller, but we do pass back a terminated flag (a game-over
        prompt closed the window) and an incomplete flag (the window closed without the marker:
        left the game, or transport EOF/TIMEOUT), and debug info.
        """
        lines = [command] if isinstance(command, str) else list(command)
        if not lines:
            raise ValueError("send_command requires at least one command line; got an empty batch")
        logger.debug("sm.send_command", command=lines, state=self.state.name)

        for line in lines:
            self.send(line)

        buffer = bytearray()
        terminated = False
        incomplete = False
        prompt_enum: Prompt | None = None
        seen_echo = False

        # Every sent line's echo is a trust boundary. Put echo patterns first so a line whose
        # literal command text is also a prompt (eg, "Option:" or "Cheerio!") is consumed as
        # our known echo rather than interpreted as game output. The final line still anchors
        # the window because its batch contains the marker command.
        echo_patterns = [
            re.compile(re.escape(encode_command_bytes(line)) + rb"(?:" + LINE_BREAK_RE + rb")") for line in lines
        ]
        final_echo_pattern = echo_patterns[-1]
        end_of_step_patterns = echo_patterns + END_OF_STEP_PATTERNS + [self.end_of_turn_marker_pattern]

        while True:
            idx, prompt_enum = self.expect(end_of_step_patterns, raise_on_eof_timeout=False)
            matched_pattern = end_of_step_patterns[idx]

            # output from between steps can arrive ahead of, or after, our echo, but either way we include
            # anything before our matched pattern in buffer to return
            buffer += self.child.before or b""

            # a prompt from outside the game (menu system, login screen) means the marker is never coming
            # these are the only ones we don't include in the returned bytes, so we break out of the loop here
            # and return what we have so far
            if prompt_enum in NO_LONGER_IN_GAME_PROMPTS:
                logger.warning(
                    "sm.send_command.left_game",
                    state=self.state.name,
                    matched_prompt=prompt_enum.name if prompt_enum else None,
                )
                incomplete = True
                break

            # otherwise we include the matched prompt in the bytes to return. TIMEOUT/EOF match the pexpect
            # exception classes rather than bytes, so there is nothing to include for those.
            if isinstance(self.child.after, bytes):
                buffer += self.child.after

            if any(matched_pattern is pattern for pattern in echo_patterns):
                if matched_pattern is final_echo_pattern:
                    seen_echo = True
                continue

            # our batch's end of turn marker closes the read window. A marker seen before our echo is stale output
            # from an earlier desynchronised window (eg, a timeout-cut step's responses arriving late): it is
            # buffered like any other pre-echo content and the read continues until our own marker.
            if matched_pattern is self.end_of_turn_marker_pattern and seen_echo:
                break

            # a rejection aborts the rest of its own line only. Before the anchoring echo it
            # belongs to an earlier line of a split batch, or to stale/ambient output (another
            # player speaking a rejection phrase), so the read continues to the marker.
            rejection_closes_window = matched_pattern in INVALID_COMMAND_PROMPTS and seen_echo
            if prompt_enum in GAME_OVER_PROMPTS or prompt_enum in TRANSPORT_BREAK_PROMPTS or rejection_closes_window:
                # the death/rejection text is already buffered above, keep it and return what we have.
                terminated = prompt_enum in GAME_OVER_PROMPTS
                incomplete = prompt_enum in TRANSPORT_BREAK_PROMPTS
                if incomplete and seen_echo:
                    logger.warning(
                        "sm.send_command.marker_missing",
                        state=self.state.name,
                        hint="no end of turn marker before timeout - did the batch end with the marker command?",
                    )
                break

        debug_info: dict[str, Any] = {"matched_prompt": prompt_enum.name if prompt_enum else None}
        logger.debug(
            "sm.send_command.complete",
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
