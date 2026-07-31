import io
import random
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Literal

import gymnasium
from pettingzoo import ParallelEnv

from mudgym.connections.provider import ConnectionProvider

StepOrder = Literal["fixed", "rotate", "shuffle"]
RenderMode = Literal["ansi", "human"]


class MudParallelEnv(ParallelEnv[str, dict[str, Any], Any]):
    """
    PettingZoo ParallelEnv over a set of MudEnv sessions.

    This is parallel at the API edge, but not true concurrency as actions are resolved
    serially according to the step_order.
    """

    metadata = {
        "render_modes": ["ansi", "human"],
        "name": "mud2_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        envs: dict[str, gymnasium.Env],
        provider: ConnectionProvider | None = None,
        render_mode: RenderMode | None = None,
        step_order: StepOrder = "rotate",
    ):
        self.envs = dict(envs)
        self.provider = provider
        self.render_mode = render_mode
        self.step_order = step_order
        self.rng = random.Random()
        self.step_count = 0
        self.episode_count = 0

        self.possible_agents = list(self.envs.keys())
        self.agents = list(self.possible_agents)

        sample_env = next(iter(self.envs.values()))
        sample_observation_space = sample_env.observation_space
        sample_action_space = sample_env.action_space

        mismatched_observation_spaces = [
            agent for agent, env in self.envs.items() if env.observation_space != sample_observation_space
        ]
        mismatched_action_spaces = [
            agent for agent, env in self.envs.items() if env.action_space != sample_action_space
        ]
        if mismatched_observation_spaces or mismatched_action_spaces:
            details = []
            if mismatched_observation_spaces:
                details.append(f"observation spaces differ for {sorted(mismatched_observation_spaces)}")
            if mismatched_action_spaces:
                details.append(f"action spaces differ for {sorted(mismatched_action_spaces)}")
            raise ValueError("MudParallelEnv requires homogeneous child spaces; " + "; ".join(details))

        self.observation_spaces = {agent: env.observation_space for agent, env in self.envs.items()}
        self.action_spaces = {agent: env.action_space for agent, env in self.envs.items()}

    def __getitem__(self, agent: str) -> gymnasium.Env:
        """Access a child env by agent name; unknown names raise KeyError."""
        return self.envs[agent]

    def __contains__(self, agent: object) -> bool:
        return agent in self.envs

    def __iter__(self) -> Iterator[str]:
        # with __getitem__ alone, ``in`` and ``for`` would fall back to the legacy
        # sequence protocol and blow up on env[0]; iterate agent names instead.
        return iter(self.envs)

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict]]:
        self.agents = list(self.possible_agents)
        if seed is not None:
            self.rng.seed(seed)
            self.episode_count = 0

        # each episode starts one agent further along the rotation, so short
        # episodes do not hand the first agent a permanent first go. Seeding
        # restarts the cycle, keeping seeded runs reproducible.
        self.step_count = self.episode_count
        self.episode_count += 1

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            agent_seed = seed + i if seed is not None else None
            # children may mutate options, nested state included, so each gets a full copy
            child_options = deepcopy(options) if options is not None else None
            obs, info = self.envs[agent].reset(seed=agent_seed, options=child_options)
            observations[agent] = obs
            infos[agent] = info

        return observations, infos

    def validate_actions(self, actions: dict[str, Any]) -> None:
        live_agents = set(self.agents)
        given_agents = set(actions)
        missing_agents = live_agents - given_agents
        extra_agents = given_agents - live_agents
        if missing_agents or extra_agents:
            raise ValueError(
                "Action dict mismatch. "
                f"Missing={sorted(missing_agents)} extra={sorted(extra_agents)} live={sorted(live_agents)}"
            )
        for agent in self.agents:
            action = actions[agent]
            action_space = self.action_space(agent)
            if not action_space.contains(action):
                raise ValueError(f"Invalid action for {agent}: {action!r}; expected {action_space}.")

    def ordered_agents(self) -> list[str]:
        ordered_agents = list(self.agents)
        if len(ordered_agents) <= 1 or self.step_order == "fixed":
            return ordered_agents
        if self.step_order == "rotate":
            # anchored to possible_agents so the offset advances one slot per step even
            # after deaths; modding by the live count would jump at a death boundary.
            offset = self.step_count % len(self.possible_agents)
            rotated = self.possible_agents[offset:] + self.possible_agents[:offset]
            live = set(ordered_agents)
            return [agent for agent in rotated if agent in live]

        self.rng.shuffle(ordered_agents)
        return ordered_agents

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        self.validate_actions(actions)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        ordered_agents = self.ordered_agents()
        for agent in ordered_agents:
            action = actions[agent]
            obs, reward, terminated, truncated, info = self.envs[agent].step(action)
            observations[agent] = obs
            # children can hand back numpy scalars; PettingZoo wants plain floats and bools
            rewards[agent] = float(reward)
            terminations[agent] = bool(terminated)
            truncations[agent] = bool(truncated)
            infos[agent] = info

        self.step_count += 1

        # an agent is live until its own child env says it is done.
        self.agents = [
            agent for agent in self.agents if not terminations.get(agent, False) and not truncations.get(agent, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def render_ansi(self) -> str:
        sections = []
        for agent in self.agents:
            result = self.envs[agent].render()
            if isinstance(result, str):
                child_frame = result
            elif isinstance(result, io.StringIO):
                child_frame = result.getvalue()
            else:
                child_frame = None
            section = f"[{agent}]\n"
            if child_frame:
                section += child_frame
                if not section.endswith("\n"):
                    section += "\n"
            sections.append(section)
        return "".join(sections).rstrip("\n")

    def render(self) -> str | None:
        if self.render_mode is None:
            return None

        rendered = self.render_ansi()
        if self.render_mode == "ansi":
            return rendered

        if rendered:
            print(rendered, flush=True)
        return None

    def close(self):
        # Children own their sessions. This wrapper only owns the provider.
        errors: list[BaseException] = []
        for env in self.envs.values():
            try:
                env.close()
            except Exception as exc:
                errors.append(exc)
        if self.provider is not None:
            try:
                self.provider.close()
            except Exception as exc:
                errors.append(exc)

        if errors:
            raise ExceptionGroup("MudParallelEnv close failed", errors)
