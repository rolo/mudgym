from collections.abc import Callable
from typing import Any

from pettingzoo import ParallelEnv

from mudgym.connections.provider import ConnectionProvider
from mudgym.envs.env import MudEnv
from mudgym.envs.lifecycle import close_players, reset_players, reset_worlds, step_players


class MudParallelEnv(ParallelEnv[str, dict[str, Any], str]):
    """Coordinates several named players acting together in one shared MUD world."""

    metadata = {
        "render_modes": ["ansi", "human"],
        "name": "mud2_v0",
    }

    def __init__(
        self,
        envs: dict[str, MudEnv],
        provider: ConnectionProvider,
        render_mode: str | None = None,
        world_ticker: Callable[[], None] | None = None,
    ):
        if not envs:
            raise ValueError("MudParallelEnv requires at least one child MudEnv.")
        self.envs = dict(envs)
        self._provider = provider
        self.render_mode = render_mode
        self.world_ticker = world_ticker

        self.possible_agents = list(self.envs)
        self.agents = list(self.possible_agents)

    def observation_space(self, agent: str):
        return self.envs[agent].observation_space

    def action_space(self, agent: str):
        return self.envs[agent].action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict]]:
        agents = list(self.possible_agents)
        reset_worlds(self.envs, self._provider, seed)
        results = reset_players(
            self.envs,
            {agent: seed + index if seed is not None else None for index, agent in enumerate(agents)},
            options,
        )
        self.agents = agents
        obs = {agent: result[0] for agent, result in results.items()}
        infos = {agent: result[1] for agent, result in results.items()}

        return obs, infos

    def step(
        self,
        actions: dict[str, str],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        obs = {}
        rewards = {}
        terminates = {}
        truncates = {}
        infos = {}

        agents = list(self.agents)
        if any(child.points is None for child in self.envs.values()):
            raise RuntimeError("Parallel environment requires a successful reset before stepping.")
        # Resolve required keys before sending so a missing action leaves sessions ready to retry.
        agent_actions = {agent: actions[agent] for agent in agents}
        for agent, result in step_players(self.envs, agent_actions, self.world_ticker):
            obs[agent], rewards[agent], terminates[agent], truncates[agent], infos[agent] = result

        # An agent stays live until its own child says it is done. Keep the snapshot above for the result dictionaries, then update the public live-agent list for the next step.
        self.agents = [agent for agent in agents if not terminates[agent] and not truncates[agent]]

        return obs, rewards, terminates, truncates, infos

    def render(self) -> str | None:
        if self.render_mode is None:
            return None

        sections = []
        for agent in self.agents:
            child_frame = self.envs[agent].render()
            section = f"[{agent}]\n"
            if child_frame:
                section += child_frame
                if not section.endswith("\n"):
                    section += "\n"
            sections.append(section)
        rendered = "".join(sections).rstrip("\n")
        if self.render_mode == "ansi":
            return rendered

        if rendered:
            print(rendered, flush=True)
        return None

    def close(self) -> None:
        close_players(self.envs, self._provider)
