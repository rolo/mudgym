import re
import uuid
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import gymnasium as gym

from mudgym.connections.connection import MudConnection
from mudgym.connections.registry import default_connection
from mudgym.db.levels import WIZARD_POINTS
from mudgym.envs.fields import FEScoreField, FieldSpec, ObservationField, instantiate_field
from mudgym.envs.specs import ACTION_CHARSET, ACTION_MAX_LENGTH, TEXT_CHARSET, TEXT_MAX_LENGTH
from mudgym.envs.validation import validate_field_spaces
from mudgym.featurizers.ansi import strip_ansi
from mudgym.featurizers.points import parse_points_changes
from mudgym.featurizers.responses import (
    normalise_lines,
    split_on_echo_lines,
    split_on_prompt,
)
from mudgym.featurizers.strings import decode_text_bytes
from mudgym.logs import get_logger
from mudgym.session import MudSession, wire_lines

logger = get_logger(__name__)

# A bare MudEnv() has no fields to derive an end-of-turn marker from, so it defaults to a marker-only
# FEScoreField: include_keys=() keeps the observation text-only while still anchoring the fes marker.
DEFAULT_FIELDS: tuple[FieldSpec, ...] = (FEScoreField(include_keys=()),)


class MudEnv(gym.Env[dict[str, Any], str]):
    """
    Gymnasium environment for MUD2.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
    }

    def __init__(
        self,
        *,
        field_parsers: Sequence[FieldSpec] | None = None,
        auto_commands: Sequence[str] | None = None,
        connection: MudConnection | Callable[[], MudConnection] = default_connection,
        render_mode: str | None = None,
    ):
        super().__init__()

        # an action is one logical game input line, so its charset excludes line breaks that the
        # observation text charset allows
        self.action_space = gym.spaces.Text(
            max_length=ACTION_MAX_LENGTH,
            min_length=1,
            charset=ACTION_CHARSET,
        )

        # None uses the default; an explicit empty sequence stays empty and fails the marker check below.
        if field_parsers is None:
            field_parsers = DEFAULT_FIELDS
        self.fields = [instantiate_field(field) for field in field_parsers]
        validate_field_spaces(self.fields)

        field_spaces: dict[str, gym.spaces.Space] = {}
        self.empty_observation: dict[str, Any] = {"text": ""}
        for field in self.fields:
            field_spaces.update(field.space())
            self.empty_observation.update(field.empty())

        self.observation_space = gym.spaces.Dict(
            {
                "text": gym.spaces.Text(max_length=TEXT_MAX_LENGTH, min_length=0, charset=TEXT_CHARSET),
                **field_spaces,
            }
        )

        if auto_commands is None:
            auto_commands = list(dict.fromkeys(field.command for field in self.fields if field.command))

        # the auto commands are appended after the player's command each step.
        # The last one is the end of turn marker to detect when we reach the end of the step's bytes.
        self.auto_commands: list[str] = list(auto_commands)
        fields_by_command = {field.command: field for field in self.fields if field.command}
        final_field = fields_by_command.get(self.auto_commands[-1]) if self.auto_commands else None
        if final_field is None or final_field.end_of_turn_marker is None:
            raise ValueError(
                f"auto_commands {self.auto_commands!r} must end with the command of a configured "
                f"field that declares an end_of_turn_marker (fei, fes, mgcheats, ...) so the "
                f"transport can detect the end of each step"
            )
        self.final_command: str = self.auto_commands[-1]
        self.end_of_turn_marker: re.Pattern = final_field.end_of_turn_marker

        # responses arrive in batch order, so field claiming follows auto_commands order
        self.auto_command_fields: list[ObservationField] = [
            fields_by_command[command] for command in self.auto_commands if command in fields_by_command
        ]

        self.env_id = uuid.uuid4()
        self.episode_id: uuid.UUID | None = None
        self.render_mode: str | None = render_mode
        self.last_bytes: bytes = b""
        self.last_render_bytes: bytes = b""

        self.session = MudSession(
            connection=connection,
            auto_commands=list(self.auto_commands),
            end_of_turn_marker=self.end_of_turn_marker,
        )

    @property
    def persona(self) -> str | None:
        """The persona name of the current player. Set during session reset."""
        return self.session.persona

    def bytes_to_observation(self, raw_bytes: bytes, info: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Turn a step's game bytes into an observation dictionary.
        """
        if info is None:
            info = {}

        # deep-copy the empty observation to avoid mutating shared state
        obs = deepcopy(self.empty_observation)

        # If we have the command echoes we split around them, both to locate the chunks and to
        # remove the echoes from the text observation. The batch goes out on one wire line, or two
        # when the command speaks (the same assembly as the session used), and each line is echoed
        # separately. The splits are stored in the info dict for anything that may need them later.
        echo_lines = wire_lines(info["last_command"], self.auto_commands)
        segments = split_on_echo_lines(raw_bytes, echo_lines)
        if segments is not None:
            # everything before the final echo (async events, plus the speech response when the
            # batch was split) reads as pre-echo; the auto command responses follow the final echo
            pre_echo_chunks = [chunk for segment in segments[:-1] for chunk in split_on_prompt(segment)]
            chunks = split_on_prompt(segments[-1])
            info["pre_echo_chunks"] = pre_echo_chunks
        else:
            pre_echo_chunks = []
            chunks = split_on_prompt(raw_bytes)
        info["chunks"] = chunks

        # fields without a command attribute set get the whole step's bytes. Raw bytes include our
        # own echo and other player-authored text, so anything scanning them for game events must
        # anchor its patterns on what players cannot type (line-exact matches, the ANSI colour
        # around a points total) rather than on the words alone.
        for field in [field for field in self.fields if field.command is None]:
            obs.update(field.extract([raw_bytes]))

        # fields with a command are extracted from the post-echo chunks, claiming in batch order
        info["auto_command_fields"] = self.auto_command_fields

        marker_arrived = (
            bool(chunks)
            and not info.get("incomplete", False)
            and self.end_of_turn_marker.search(strip_ansi(bytes(chunks[-1]))) is not None
        )
        payload_text_chunks: list[bytes] = []
        auto_command_chunks: list[bytes] = []
        if marker_arrived:
            pending_fields = list(self.auto_command_fields)
            for position, chunk in enumerate(chunks):
                field = pending_fields[0] if pending_fields else None
                if field is not None and field.is_refusal(chunk):
                    # a player-state refusal (unconscious, asleep, ...) is the game's real answer to
                    # the auto command: it claims the slot but carries no field data.
                    info.setdefault("field_refusals", {})[field.__class__.__name__] = chunk
                    payload_text_chunks.append(chunk)
                    auto_command_chunks.append(chunk)
                    pending_fields.pop(0)
                elif field is not None and field.matches(chunk):
                    obs.update(field.extract([chunk]))
                    if not field.remove_on_match:
                        payload_text_chunks.append(chunk)
                    auto_command_chunks.append(chunk)
                    pending_fields.pop(0)
                elif position == len(chunks) - 1:
                    # the marker chunk is the final command's own output.
                    # when no field claims it, it is just for protocol rather than observation and stays out of the text.
                    pass
                else:
                    payload_text_chunks.append(chunk)
            assert not pending_fields, (
                f"end-of-turn marker arrived but fields {[f.__class__.__name__ for f in pending_fields]} "
                f"found no matching response among {len(chunks)} window chunks"
            )

        text_chunks = [*pre_echo_chunks, *payload_text_chunks]
        if not marker_arrived:
            text_chunks.extend(chunks)

        info["auto_command_chunks"] = auto_command_chunks
        info["text_chunks"] = text_chunks

        # the human facing payload corresponding to obs["text"] with remove_on_match fields excluded and ANSI colour preserved.
        render_bytes = normalise_lines(b"\n".join(text_chunks))
        info["render_bytes"] = render_bytes

        # strip ANSI escape codes
        text = strip_ansi(render_bytes)

        # decode to a string
        text = decode_text_bytes(text)
        info["text"] = text

        if len(text) > TEXT_MAX_LENGTH:
            logger.warning(f"text length {len(text)} exceeds TEXT_MAX_LENGTH {TEXT_MAX_LENGTH}, truncating")

        obs["text"] = text[:TEXT_MAX_LENGTH]

        return obs

    def make_info(self, raw_bytes: bytes, debug_info: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "ids": {
                "env_id": str(self.env_id),
                "episode_id": str(self.episode_id) if self.episode_id else None,
            },
            "raw_bytes": raw_bytes,
            "bytes_length": len(raw_bytes),
        }
        info["step"] = debug_info.get("step")
        info["persona"] = self.persona
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

        self.episode_id = uuid.uuid4()
        logger.debug("env.reset", env_id=str(self.env_id), episode_id=str(self.episode_id))

        # reset the session which takes us to the tearoom and sets the person name via quickscore
        self.session.reset()

        # step out of the tearoom and into The Land
        command = "move north"
        raw_bytes, terminated, incomplete, debug_info = self.session.send_command(command)

        if terminated or incomplete:
            raise RuntimeError(
                f"step out of the tearoom failed during reset "
                f"(terminated={terminated}, incomplete={incomplete}); "
                f"env_id={self.env_id}; "
                f"episode_id={self.episode_id}; "
                f"bytes_length={len(raw_bytes)}; "
                f"raw_bytes={raw_bytes!r}; "
                f"debug_info={debug_info!r}"
            )

        # we consider the episode started after exiting the tearoom, so we drop the tearoom-exit narration
        # ("shape..." / "nothingness...") that sits between the echoed command and the room text, ending at
        # the "...\n" marker, leaving the echo line intact so the observation split still anchors on it.
        trim_re = re.compile(rb"\.\.\.\r?\n")
        echo_end = raw_bytes.find(b"\n")
        match = trim_re.search(raw_bytes, echo_end + 1)
        if match is None:
            raise ValueError(f"tearoom exit marker {trim_re.pattern!r} not found in: {raw_bytes!r}")
        raw_bytes = raw_bytes[: echo_end + 1] + raw_bytes[match.end() :]

        self.last_bytes = raw_bytes

        info = self.make_info(raw_bytes, debug_info)
        info["last_command"] = command

        obs = self.bytes_to_observation(raw_bytes, info)
        self.last_render_bytes = info["render_bytes"]

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self,
        action: str,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # a line break would smuggle an extra wire command past the batch assembly, desynchronising
        # the step, so it gets its own pointed error ahead of the generic space check
        if "\r" in action or "\n" in action:
            raise ValueError(
                f"action {action!r} contains a line break; an action must be a single logical "
                f"input line (the env appends its own auto command line each step)"
            )

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected {self.action_space}.")

        # a player-issued marker command would emit the marker mid-batch, closing the read window
        # before the auto commands answer and desynchronising the step.
        if any(token.strip().lower() == self.final_command for token in action.split(",")):
            raise ValueError(
                f"action {action!r} contains {self.final_command!r}, "
                f"which is reserved as the end-of-turn marker command"
            )

        # converse mode turns every subsequent input line into speech, auto commands included, so
        # the end-of-turn marker would never run again and the transport would desynchronise for
        # the rest of the episode.
        if any(token.strip().split(None, 1)[0].lower() == "converse" for token in action.split(",") if token.strip()):
            raise ValueError(f"action {action!r} contains 'converse', which would break the step protocol")

        raw_bytes, terminated, incomplete, debug_info = self.session.send_command(action)
        self.last_bytes = raw_bytes

        # an incomplete read window (left the game, transport EOF/TIMEOUT) ends the episode for
        # reasons outside the MDP
        truncated = incomplete

        # build info first, then let bytes_to_observation enrich it (text_chunks, ...) -- mirrors reset().
        info = self.make_info(raw_bytes, debug_info)

        # 'command' refers to the text command, 'action' we use to describe the version the action space uses.
        # in the default case with a text action space they are the same, but if we use a discrete action space or other
        # action space wrappers, they may differ.
        info["last_command"] = action

        # the parse must know when the read window closed without the marker
        info["incomplete"] = incomplete

        obs = self.bytes_to_observation(raw_bytes, info)
        self.last_render_bytes = info["render_bytes"]

        # calculate the reward from any points change events. The scan runs over the raw bytes,
        # echo included: forgery resistance lives in the pattern itself, which requires the ANSI
        # colour a player cannot type (see featurizers.points).
        reward_events = parse_points_changes(raw_bytes)
        points = reward_events["points"]
        if points is not None:
            info["points"] = points
            # if we end up with points > WIZARD_POINTS, we terminate the episode. Only a
            # colour-anchored change event can prove the crossing.
            if reward_events["delta"] and points >= WIZARD_POINTS:
                terminated = True

        # in the, albeit very unusual, case where the player achieves wizard, it may be correct to cap the reward at the
        # delta up to WIZARD_POINTS, but I'm not sure and I think the current behaviour is more predictable at least.
        reward = float(reward_events["delta"] or 0)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()
        self.session.close()
