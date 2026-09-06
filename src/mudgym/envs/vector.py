from collections.abc import Callable, Sequence
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
        world_ticker: Callable[[], None] | None = None,
        autoreset_mode: AutoresetMode | str = AutoresetMode.DISABLED,
    ):
        if not envs:
            raise ValueError("MudVectorEnv requires at least one child MudEnv.")
        autoreset_mode = AutoresetMode(autoreset_mode)
        if autoreset_mode not in {AutoresetMode.DISABLED, AutoresetMode.NEXT_STEP}:
            raise ValueError(f"Unsupported autoreset_mode {autoreset_mode.value!r}. Use Disabled or NextStep.")
        self.envs = list(envs)
        self._provider = provider
        self.world_ticker = world_ticker
        self.autoreset_mode = autoreset_mode
        self.metadata = dict(self.envs[0].metadata)
        self.metadata["autoreset_mode"] = autoreset_mode
        self.render_mode = self.envs[0].render_mode
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._needs_reset = np.zeros(self.num_envs, dtype=np.bool_)
        self.observations: list[dict[str, Any]] | None = None

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

    def reset_selection(self, options: dict[str, Any] | None) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Validate Gymnasium's reset mask without changing the caller's options."""
        child_options = dict(options or {})
        reset_mask = child_options.pop("reset_mask", None)
        if reset_mask is None:
            return np.ones(self.num_envs, dtype=np.bool_), child_options or None
        if not isinstance(reset_mask, np.ndarray):
            raise TypeError(f"options['reset_mask'] must be a numpy array, got {type(reset_mask)!r}.")
        if reset_mask.shape != (self.num_envs,):
            raise ValueError(f"options['reset_mask'] must have shape ({self.num_envs},), got {reset_mask.shape}.")
        if reset_mask.dtype != np.bool_:
            raise TypeError(f"options['reset_mask'] must have dtype numpy.bool_, got {reset_mask.dtype}.")
        if not reset_mask.any():
            raise ValueError("options['reset_mask'] must select at least one child.")
        if self.observations is None and not reset_mask.all():
            raise RuntimeError("A partial reset_mask requires an earlier full vector reset.")
        return reset_mask, child_options or None

    def reset_children(
        self,
        reset_mask: np.ndarray,
        seeds: Sequence[int | None],
        options: dict[str, Any] | None,
    ) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
        """Discard each reset's first observation until every selected child has entered the world."""
        selected_indices = np.flatnonzero(reset_mask).tolist()
        for index in selected_indices:
            self.envs[index].reset(seed=seeds[index], options=options)

        results = {}
        for index in selected_indices:
            observation, _, terminated, truncated, info = self.envs[index].observe()
            if terminated or truncated:
                raise RuntimeError(
                    f"initial vector observation failed for child {index} "
                    f"(terminated={terminated}, truncated={truncated})"
                )
            results[index] = observation, info
        return results

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ):
        reset_mask, child_options = self.reset_selection(options)
        if isinstance(seed, int):
            super().reset(seed=seed)
        seeds = self.child_seeds(seed)
        if reset_mask.all():
            self._provider.reset(seed=seed)

        reset_results = self.reset_children(reset_mask, seeds, child_options)
        observations = list(self.observations) if self.observations is not None else [{} for _ in self.envs]
        infos = [{} for _ in self.envs]
        for index, (observation, info) in reset_results.items():
            observations[index], infos[index] = observation, info
        self.observations = observations
        self._needs_reset[reset_mask] = False
        return self.batch_observations(self.observations), self.batch_infos(infos)

    def step(self, actions):
        """Send every live child action before observing, then reset children already done."""
        if self.observations is None:
            raise RuntimeError("Vector environment has not been reset.")
        resetting = self._needs_reset.copy()
        if self.autoreset_mode is AutoresetMode.DISABLED and resetting.any():
            indices = np.flatnonzero(self._needs_reset).tolist()
            raise RuntimeError(f"Vector children {indices} are done; call reset() before stepping again.")

        child_actions = list(iterate(self.action_space, actions))
        if len(child_actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(child_actions)}.")
        live_indices = np.flatnonzero(~resetting).tolist()
        # Don't fold these loops together. A player can affect another player's observation, so every action must reach the game before any observation commands are sent.
        for index in live_indices:
            self.envs[index].act(child_actions[index])

        if live_indices and self.world_ticker is not None:
            self.world_ticker()

        results: dict[int, tuple[dict[str, Any], float, bool, bool, dict[str, Any]]] = {
            index: self.envs[index].observe() for index in live_indices
        }
        if resetting.any():
            # A relogin is visible to other players in a shared world. Finish every live observation first so a done
            # child's new episode cannot change another child's preceding transition.
            reset_results = self.reset_children(resetting, [None] * self.num_envs, None)
            for index, (observation, info) in reset_results.items():
                results[index] = observation, 0.0, False, False, info

        ordered_results = [results[index] for index in range(self.num_envs)]
        observations, rewards, terminations, truncations, infos = zip(*ordered_results, strict=True)
        terminations = np.asarray(terminations, dtype=np.bool_)
        truncations = np.asarray(truncations, dtype=np.bool_)
        self._needs_reset = np.logical_or(terminations, truncations)
        self.observations = list(observations)
        return (
            self.batch_observations(self.observations),
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
