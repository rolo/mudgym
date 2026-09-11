from collections.abc import Callable
from typing import Any

from pettingzoo import ParallelEnv

from mudgym.connections.provider import ConnectionProvider
from mudgym.envs.env import MudEnv


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
        agent = None
        phase = "provider reset"
        try:
            self._provider.reset(seed=seed)
            self.agents = list(agents)
            phase = "preparation"
            for index, agent in enumerate(agents):
                agent_seed = seed + index if seed is not None else None
                self.envs[agent]._prepare_reset(seed=agent_seed, options=options)
            phase = "entry"
            entries = {}
            for agent in agents:
                entries[agent] = self.envs[agent]._enter_world()
            phase = "observation"
            observations = {}
            infos = {}
            for agent in agents:
                observations[agent], infos[agent] = self.envs[agent]._finish_reset(*entries[agent])
        except BaseException as error:
            error.add_note(f"parallel reset failed during {phase} for agent {agent!r}")
            for selected_agent in agents:
                self.envs[selected_agent]._invalidate_reset(error)
            raise

        return observations, infos

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
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        agents = list(self.agents)
        if any(child.points is None for child in self.envs.values()):
            raise RuntimeError("Parallel environment requires a successful reset before stepping.")
        # Resolve required keys before sending so a missing action leaves sessions ready to retry.
        agent_actions = [(agent, actions[agent]) for agent in agents]
        # This ordering is the point of the coordinator: everybody acts before anybody runs their
        # observation commands. Folding the loops together would make later players invisible to
        # earlier observations from the same PettingZoo step.
        for agent, action in agent_actions:
            self.envs[agent].act(action)

        if self.world_ticker is not None:
            self.world_ticker()

        for agent in agents:
            (
                observations[agent],
                rewards[agent],
                terminations[agent],
                truncations[agent],
                infos[agent],
            ) = self.envs[agent].observe()

        # An agent stays live until its own child says it is done. Keep the snapshot above for the result dictionaries, then update the public live-agent list for the next step.
        self.agents = [agent for agent in agents if not terminations[agent] and not truncations[agent]]

        return observations, rewards, terminations, truncations, infos

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
        # Children own their connections; the provider owns the shared world beneath them. Attempt every close even if one fails so one awkward child does not leak everybody else's state.
        errors: list[Exception] = []
        for env in self.envs.values():
            try:
                env.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self._provider.close()
        except Exception as exc:
            errors.append(exc)

        if errors:
            raise ExceptionGroup("MudParallelEnv close failed", errors)
