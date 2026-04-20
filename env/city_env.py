"""Gymnasium environment for the Eco-City grid simulation."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dynamics import Zone, update_metrics
from env.terrain import WorldLayout, generate_world

GRID_SIZE = 10
NUM_ZONE_TYPES = 7  # one-hot channels 0..6 (matches Zone enum)
NUM_GLOBAL_FEATURES = 4
CANDIDATE_CELLS = 5
MAX_STEPS = 200
# Zone one-hot + per-cell buildable mask + normalized globals
FLAT_OBS_SIZE = (
    GRID_SIZE * GRID_SIZE * NUM_ZONE_TYPES
    + GRID_SIZE * GRID_SIZE
    + NUM_GLOBAL_FEATURES
)


def _normalize_globals(
    population: float,
    pollution: float,
    traffic: float,
    energy_supply: float,
    energy_demand: float,
) -> np.ndarray:
    pop_n = float(np.clip(population / 1000.0, 0.0, 1.0))
    pol_n = float(np.clip(pollution / 500.0, 0.0, 1.0))
    tr_n = float(np.clip(traffic / 200.0, 0.0, 1.0))
    eb = energy_supply - energy_demand
    eb_n = float(np.clip((eb + 500.0) / 1000.0, 0.0, 1.0))
    return np.array([pop_n, pol_n, tr_n, eb_n], dtype=np.float32)


class CityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.2,
        gamma: float = 0.7,
        delta: float = 0.8,
        max_steps: int = MAX_STEPS,
        candidate_cells: int = CANDIDATE_CELLS,
        grid_size: int = GRID_SIZE,
        industrial_emission_factor: float = 1.0,
        residential_pop_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.max_steps = max_steps
        self.candidate_cells_n = candidate_cells
        self.industrial_emission_factor = industrial_emission_factor
        self.residential_pop_multiplier = residential_pop_multiplier

        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.metrics: dict[str, float] = {}
        self.step_count = 0
        self.candidate_cells = np.zeros(candidate_cells, dtype=np.int64)
        self.state_history: list[dict[str, Any]] = []
        # When set to a list, each step appends a JSON-ready snapshot (survives VecEnv autoreset).
        self._export_traj: list[dict[str, Any]] | None = None
        self._rng = np.random.default_rng()
        self._world_layout: WorldLayout | None = None
        self._terrain_seed: int = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(FLAT_OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(candidate_cells * NUM_ZONE_TYPES)

    def _reset_metrics(self) -> None:
        self.metrics = {
            "population": 0.0,
            "pollution": 0.0,
            "traffic": 0.0,
            "energy_supply": 0.0,
            "energy_demand": 0.0,
        }

    def _sample_candidates(self) -> None:
        assert self._world_layout is not None
        flat = self._world_layout.buildable.ravel()
        buildable_idx = np.flatnonzero(flat).astype(np.int64)
        if buildable_idx.size == 0:
            raise RuntimeError("No buildable cells; relax terrain parameters.")
        replace = buildable_idx.size < self.candidate_cells_n
        pick = self._rng.choice(
            buildable_idx,
            size=self.candidate_cells_n,
            replace=replace,
        )
        self.candidate_cells = pick.astype(np.int64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._terrain_seed = int(seed) if seed is not None else int(self._rng.integers(0, 2**31 - 1))
        self._world_layout = generate_world(self.grid_size, self._terrain_seed)
        self.grid.fill(0)
        self._reset_metrics()
        self.step_count = 0
        self.state_history = []
        self._sample_candidates()
        update_metrics(
            self.grid,
            self.metrics,
            industrial_emission_factor=self.industrial_emission_factor,
            residential_pop_multiplier=self.residential_pop_multiplier,
        )
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        one_hot = np.zeros(
            (self.grid_size, self.grid_size, NUM_ZONE_TYPES),
            dtype=np.float32,
        )
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                z = int(self.grid[r, c])
                if 0 <= z < NUM_ZONE_TYPES:
                    one_hot[r, c, z] = 1.0
        flat_grid = one_hot.reshape(-1)
        g = _normalize_globals(
            self.metrics["population"],
            self.metrics["pollution"],
            self.metrics["traffic"],
            self.metrics["energy_supply"],
            self.metrics["energy_demand"],
        )
        assert self._world_layout is not None
        buildable_flat = self._world_layout.buildable.astype(np.float32).reshape(-1)
        return np.concatenate([flat_grid, buildable_flat, g], axis=0).astype(np.float32)

    def _compute_livability(self) -> float:
        return float(min(self.metrics["population"] / 100.0, 1.0))

    def _compute_reward(self) -> float:
        livability = self._compute_livability()
        pollution = float(self.metrics["pollution"])
        traffic = float(self.metrics["traffic"])
        mismatch = abs(self.metrics["energy_demand"] - self.metrics["energy_supply"])
        return (
            self.alpha * livability
            - self.beta * pollution
            - self.gamma * traffic
            - self.delta * mismatch
        )

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        a = int(action)
        cell_slot = a // NUM_ZONE_TYPES
        zone_type = a % NUM_ZONE_TYPES
        cell_idx = int(self.candidate_cells[cell_slot])
        row = cell_idx // self.grid_size
        col = cell_idx % self.grid_size
        self.grid[row, col] = zone_type

        update_metrics(
            self.grid,
            self.metrics,
            industrial_emission_factor=self.industrial_emission_factor,
            residential_pop_multiplier=self.residential_pop_multiplier,
        )
        reward = float(self._compute_reward())
        energy_balance = float(self.metrics["energy_supply"] - self.metrics["energy_demand"])

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        snapshot = {
            "step": self.step_count,
            "grid": self.grid.copy(),
            "population": float(self.metrics["population"]),
            "pollution": float(self.metrics["pollution"]),
            "traffic": float(self.metrics["traffic"]),
            "energy_balance": energy_balance,
            "reward": reward,
            "terrain_seed": self._terrain_seed,
        }
        self.state_history.append(snapshot)
        if self._export_traj is not None:
            self._export_traj.append(
                {
                    "step": snapshot["step"],
                    "grid": self.grid.copy().tolist(),
                    "population": snapshot["population"],
                    "pollution": snapshot["pollution"],
                    "traffic": snapshot["traffic"],
                    "energy_balance": snapshot["energy_balance"],
                    "reward": snapshot["reward"],
                }
            )

        self._sample_candidates()
        obs = self._get_obs()

        info = {
            "population": float(self.metrics["population"]),
            "pollution": float(self.metrics["pollution"]),
            "traffic": float(self.metrics["traffic"]),
            "energy_balance": energy_balance,
            "energy_demand": float(self.metrics["energy_demand"]),
            "energy_supply": float(self.metrics["energy_supply"]),
        }
        return obs, reward, terminated, truncated, info

    def export_history(self, path: str) -> None:
        from visualization.exporter import export

        export(self, path)


def decode_action(action: int, candidate_cells: np.ndarray, grid_size: int) -> tuple[int, int, int]:
    """Debug helper: (row, col, zone_type)."""
    a = int(action)
    cell_slot = a // NUM_ZONE_TYPES
    zone_type = a % NUM_ZONE_TYPES
    cell_idx = int(candidate_cells[cell_slot])
    row = cell_idx // grid_size
    col = cell_idx % grid_size
    return row, col, zone_type
