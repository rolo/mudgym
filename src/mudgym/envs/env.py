import re
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import gymnasium as gym

from mudgym.connections.connection import MudConnection
from mudgym.connections.termination import is_permadeath
from mudgym.db.levels import WIZARD_POINTS
from mudgym.envs.fields import FEScoreField, FieldSpec, ObservationField, instantiate_field
from mudgym.envs.specs import ACTION_CHARSET, ACTION_MAX_LENGTH, INT_DTYPE, TEXT_CHARSET, TEXT_MAX_LENGTH
from mudgym.featurizers.ansi import strip_ansi
from mudgym.featurizers.points import parse_points_changes
from mudgym.featurizers.responses import (
    RESPONSE_PROMPT_RE,
    echo_pattern,
    normalise_lines,
    split_on_echo_lines,
    split_on_prompt,
)
from mudgym.featurizers.strings import decode_text_bytes
from mudgym.logs import get_logger
from mudgym.session import MudSession

logger = get_logger(__name__)

DEFAULT_FIELDS: tuple[FieldSpec, ...] = (FEScoreField(include_keys=()),)


class MudEnv(gym.Env[dict[str, Any], str]):
    """
    A Gymnasium environment for MUD2.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
    }

    def __init__(
        self,
        *,
        field_parsers: Sequence[FieldSpec] | None = None,
        tearoom_commands: str | None = None,
        connection: MudConnection,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.action_space = gym.spaces.Text(
            max_length=ACTION_MAX_LENGTH,
            min_length=1,
            charset=ACTION_CHARSET,
        )

        # None means use default. An explicit empty sequence means empty but it will fail the command check below as we
        # need an end of step marker
        if field_parsers is None:
            field_parsers = DEFAULT_FIELDS
        self.fields = [instantiate_field(field) for field in field_parsers]

        observation_space: dict[str, gym.spaces.Space] = {
            "text": gym.spaces.Text(max_length=TEXT_MAX_LENGTH, min_length=0, charset=TEXT_CHARSET),
        }
        empty_observation: dict[str, Any] = {"text": ""}
        for field in self.fields:
            field_space = field.space()
            duplicates = observation_space.keys() & field_space.keys()
            if duplicates:
                raise ValueError(f"Duplicate observation keys: {sorted(duplicates)}")
            observation_space.update(field_space)
            empty_observation.update(field.empty())

        self.observation_space = gym.spaces.Dict(observation_space)
        self.empty_observation = empty_observation

        command_fields = tuple(field for field in self.fields if field.command is not None)
        commands = tuple(field.command for field in command_fields)
        if not command_fields:
            raise ValueError("At least one observation field must declare a command.")

        final_field = command_fields[-1]
        if final_field.end_of_turn_marker is None:
            raise ValueError(
                "The final commanded observation field must declare an end_of_turn_marker (fei, fes, mgcheats, ...)."
            )

        self.observation_command_fields: tuple[ObservationField, ...] = command_fields
        observation_line = ",".join(commands)

        self.tearoom_commands = tearoom_commands
        self.render_mode = render_mode
        self.last_render_bytes: bytes = b""
        self.step_count = 0

        # keep score independently of any one observation response
        self.points: int | None = None

        self.session = MudSession(
            connection=connection,
            observation_line=observation_line,
            end_of_turn_marker=final_field.end_of_turn_marker,
        )

    @property
    def persona(self) -> str | None:
        """The persona name of the current player. Set during session reset."""
        return self.session.persona

    def bytes_to_observation(
        self,
        raw_bytes: bytes,
        *,
        sent_lines: Sequence[str],
        response_complete: bool,
    ) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
        """Turn a step's response payload into an observation and its renderable bytes."""

        # deepcopy so we don't accidentally mutate
        obs = deepcopy(self.empty_observation)

        # split the response into pre and post echo
        segments = split_on_echo_lines(raw_bytes, sent_lines)
        if segments is not None:
            # anything that came from the game before our echo we don't try and parse into observation fields
            pre_echo_chunks = [chunk for segment in segments[:-1] for chunk in split_on_prompt(segment)]
            chunks = split_on_prompt(segments[-1])
        else:
            pre_echo_chunks = []
            chunks = split_on_prompt(raw_bytes)

        # fields with no command set use the whole step's bytes.
        for field in (field for field in self.fields if field.command is None):
            obs.update(field.extract([raw_bytes], persona=self.persona))

        payload_text_chunks: list[bytes] = []
        field_refusals: dict[str, bytes] = {}
        if response_complete:
            # claim in the same order commands were sent. A refusal still consumes, eg, asleep
            pending_fields = list(self.observation_command_fields)
            for position, chunk in enumerate(chunks):
                field = pending_fields[0] if pending_fields else None
                if field is not None and field.is_refusal(chunk):
                    field_refusals[field.__class__.__name__] = chunk
                    payload_text_chunks.append(chunk)
                    pending_fields.pop(0)
                elif field is not None and field.matches(chunk):
                    obs.update(field.extract([chunk], persona=self.persona))
                    if not field.remove_on_match:
                        payload_text_chunks.append(chunk)
                    pending_fields.pop(0)
                elif position == len(chunks) - 1:
                    # final chunk can be used just as a marker rather than a field
                    pass
                else:
                    payload_text_chunks.append(chunk)
            if pending_fields:
                raise RuntimeError(
                    f"end of step marker arrived but fields {[f.__class__.__name__ for f in pending_fields]} "
                    f"found no matching response among {len(chunks)} window chunks"
                )

        text_chunks = [*pre_echo_chunks, *payload_text_chunks]
        if not response_complete:
            # Without the marker we cannot safely line chunks up with fields. Preserve the bytes as text rather than pretending the structured observation is complete.
            text_chunks.extend(chunks)

        # keeps the game's ANSI colour - text observation space doesn't.
        render_bytes = normalise_lines(b"\n".join(text_chunks))
        text = decode_text_bytes(strip_ansi(render_bytes))

        if len(text) > TEXT_MAX_LENGTH:
            logger.warning(f"text length {len(text)} exceeds TEXT_MAX_LENGTH {TEXT_MAX_LENGTH}, truncating")

        obs["text"] = text[:TEXT_MAX_LENGTH]
        return obs, render_bytes, field_refusals

    def clean_tearoom_exit(self, raw_bytes: bytes, sent_lines: Sequence[str]) -> tuple[bytes, int]:
        """Remove the tearoom setup and return the exit fes score."""
        if len(sent_lines) != 2:
            raise RuntimeError(f"tearoom exit sent {len(sent_lines)} wire lines, expected 2")

        action_echo = echo_pattern(sent_lines[0]).search(raw_bytes)
        observation_echo = echo_pattern(sent_lines[1]).search(
            raw_bytes,
            action_echo.end() if action_echo is not None else 0,
        )
        if action_echo is None or observation_echo is None:
            raise RuntimeError(f"tearoom exit echoes not found in: {raw_bytes!r}")

        action_response = raw_bytes[action_echo.end() : observation_echo.start()]
        prompts = tuple(RESPONSE_PROMPT_RE.finditer(action_response))
        if not prompts:
            raise RuntimeError(f"tearoom exit fes prompt not found in: {action_response!r}")

        fes_prompt = prompts[0]
        fes_response = action_response[: fes_prompt.start()]
        move_response = action_response[fes_prompt.end() :]
        field = FEScoreField(include_keys=())
        if not field.matches(fes_response):
            raise RuntimeError("reset completed without establishing the persona score")

        trim_re = re.compile(rb"\.\.\.\r?\n")
        trim_match = trim_re.search(move_response)
        if trim_match is None:
            raise ValueError(f"tearoom exit marker {trim_re.pattern!r} not found in: {move_response!r}")

        cleaned_bytes = (
            raw_bytes[: action_echo.end()] + move_response[trim_match.end() :] + raw_bytes[observation_echo.start() :]
        )
        values = field.full_extract([fes_response], persona=self.persona)
        return cleaned_bytes, int(values["points"])

    def make_info(
        self,
        *,
        raw_bytes: bytes,
        render_bytes: bytes,
        rejected: bool,
        field_refusals: dict[str, bytes],
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "raw_bytes": raw_bytes,
            "render_bytes": render_bytes,
            "step": self.step_count,
            "persona": self.persona,
            "action_rejected": rejected,
        }
        if field_refusals:
            info["field_refusals"] = field_refusals
        return info

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        cleaned_text = decode_text_bytes(self.last_render_bytes)
        if self.render_mode == "human":
            print(cleaned_text, end="", flush=True)
            return None
        return cleaned_text

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            self.action_space.seed(seed)

        self.step_count = 0

        # reset the session which takes us to the tearoom and sets the persona name via quickscore
        self.session.reset()

        # tearoom commands are episode setup, issued before the exit step
        if self.tearoom_commands:
            raw_bytes, terminated, truncated, _ = self.session.command(self.tearoom_commands)
            if terminated or truncated:
                raise RuntimeError(
                    f"tearoom commands {self.tearoom_commands!r} failed during reset "
                    f"(terminated={terminated}, truncated={truncated}); raw_bytes={raw_bytes!r}"
                )

        # step out of the tearoom and into The Land
        command = "fes,move north"
        raw_bytes, terminated, truncated, debug_info = self.session.command(command)

        if terminated or truncated:
            raise RuntimeError(
                f"step out of the tearoom failed during reset "
                f"(terminated={terminated}, truncated={truncated}); "
                f"bytes_length={len(raw_bytes)}; "
                f"raw_bytes={raw_bytes!r}; "
                f"debug_info={debug_info!r}"
            )

        raw_bytes, self.points = self.clean_tearoom_exit(raw_bytes, debug_info["sent_lines"])

        obs, render_bytes, field_refusals = self.bytes_to_observation(
            raw_bytes,
            sent_lines=debug_info["sent_lines"],
            response_complete=bool(debug_info.get("marker_arrived", False)),
        )
        self.last_render_bytes = render_bytes
        info = self.make_info(
            raw_bytes=raw_bytes,
            render_bytes=render_bytes,
            rejected=bool(debug_info.get("rejected", False)),
            field_refusals=field_refusals,
        )

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self,
        action: str,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Send one action, then receive its marker-framed observation.

        Vector and parallel coordinators call ``act()`` on every child before calling ``observe()``
        on any child, so each returned observation follows the complete joint action batch.
        """
        self.act(action)
        return self.observe()

    def act(self, action: str) -> None:
        """Send an action now, leaving its observation for a later ``observe`` call."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected {self.action_space}.")
        if self.points is None:
            raise RuntimeError("step called before reset established the persona score")
        self.session.send(action)
        self.step_count += 1

    def observe(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Receive everything up to this player's end-of-turn marker.

        This includes the earlier action, the observation-command responses, and anything caused by other players since that action was sent.
        """
        raw_bytes, terminated, truncated, debug_info = self.session.receive()

        # A truncated result means the read window ended without its marker: we left the game, the transport died, timed out, or otherwise stopped for a reason outside the MDP.
        obs, render_bytes, field_refusals = self.bytes_to_observation(
            raw_bytes,
            sent_lines=debug_info["sent_lines"],
            response_complete=bool(debug_info.get("marker_arrived", False)) and not truncated,
        )
        self.last_render_bytes = render_bytes
        info = self.make_info(
            raw_bytes=raw_bytes,
            render_bytes=render_bytes,
            rejected=bool(debug_info.get("rejected", False)),
            field_refusals=field_refusals,
        )

        # points events require colours a player cannot forge through the command echo
        points_before_step = self.points
        if points_before_step is None:
            raise RuntimeError("step called before reset established the persona score")

        points_changes = parse_points_changes(raw_bytes)
        event_points = points_changes["points"]
        # if we end up with points > WIZARD_POINTS, we terminate the episode.
        if event_points is not None and points_changes["delta"] and event_points >= WIZARD_POINTS:
            terminated = True

        # permadeath is a final points event at zero, after any numeric event
        if terminated and is_permadeath(raw_bytes):
            event_points = 0
        if event_points is not None:
            info["points"] = event_points
            self.points = event_points

        reward = float(self.points - points_before_step)

        # report tracked points even when the FES response is stale or missing
        if "points" in obs:
            obs["points"] = INT_DTYPE(self.points)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()
        self.session.close()
