from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate

from mudgym.connections.provider import ConnectionProvider
from mudgym.envs.env import MudEnv


class MudVectorEnv(VectorEnv):
    def __init__(
        self,
        envs: Sequence[MudEnv],
        provider: ConnectionProvider,
    ):
        if not envs:
            raise ValueError("MudVectorEnv requires at least one child MudEnv.")
        self.envs = list(envs)
        self._provider = provider
        self.metadata = dict(self.envs[0].metadata)
        self.metadata["autoreset_mode"] = AutoresetMode.DISABLED
        self.render_mode = self.envs[0].render_mode
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._needs_reset = np.zeros(self.num_envs, dtype=np.bool_)

    def batch_observations(self, observations: Sequence[dict[str, Any]]):
        """Put child observations into the vector observation space."""
        output = create_empty_array(self.single_observation_space, n=self.num_envs, fn=np.empty)
        return concatenate(self.single_observation_space, observations, output)

    def batch_infos(self, infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Use Gymnasium's mask convention to combine child info dictionaries."""
        batched: dict[str, Any] = {}
        for index, info in enumerate(infos):
            batched = self._add_info(batched, info, index)
        return batched

    def child_seeds(self, seed: int | list[int | None] | None) -> list[int | None]:
        """Turn the vector seed into one Gym-side seed per child."""
        if seed is None:
            return [None] * self.num_envs
        if isinstance(seed, int):
            return [seed + index for index in range(self.num_envs)]
        if len(seed) != self.num_envs:
            raise ValueError(f"Seed list must contain {self.num_envs} entries, got {len(seed)}.")
        return list(seed)

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ):
        if isinstance(seed, int):
            super().reset(seed=seed)
        self._provider.reset(seed=seed)
        seeds = self.child_seeds(seed)
        for child, child_seed in zip(self.envs, seeds, strict=True):
            child.reset(seed=child_seed, options=options)

        observations = []
        infos = []
        for index, child in enumerate(self.envs):
            observation, _, terminated, truncated, info = child.observe()
            if terminated or truncated:
                raise RuntimeError(
                    f"initial vector observation failed for child {index} "
                    f"(terminated={terminated}, truncated={truncated})"
                )
            observations.append(observation)
            infos.append(info)

        self._needs_reset[:] = False
        return self.batch_observations(observations), self.batch_infos(infos)

    def step(self, actions):
        """Send every child action before receiving any child observation; never autoreset."""
        if self._needs_reset.any():
            indices = np.flatnonzero(self._needs_reset).tolist()
            raise RuntimeError(f"Vector children {indices} are done; call reset() before stepping again.")

        child_actions = list(iterate(self.action_space, actions))
        # Don't fold these loops together. A player can affect another player's observation, so every action must reach the game before any observation commands are sent.
        for child, action in zip(self.envs, child_actions, strict=True):
            child.act(action)

        results = [child.observe() for child in self.envs]
        observations, rewards, terminations, truncations, infos = zip(*results, strict=True)
        terminations = np.asarray(terminations, dtype=np.bool_)
        truncations = np.asarray(truncations, dtype=np.bool_)
        self._needs_reset = np.logical_or(terminations, truncations)
        return (
            self.batch_observations(observations),
            np.asarray(rewards, dtype=np.float64),
            terminations,
            truncations,
            self.batch_infos(infos),
        )

    def render(self):
        return tuple(child.render() for child in self.envs)

    def close_extras(self, **kwargs):
        # Children own their connections and the provider owns whatever sits underneath them. Try every close even if one fails, then report the lot rather than leaking the rest.
        errors: list[Exception] = []
        for child in self.envs:
            try:
                child.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self._provider.close()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise ExceptionGroup("MudVectorEnv close failed", errors)
