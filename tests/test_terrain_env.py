"""Terrain / buildable mask and env sampling invariants."""

from __future__ import annotations

import numpy as np

from env.city_env import CityEnv, NUM_ZONE_TYPES
from env.dynamics import Zone
from env.terrain import generate_world


def test_generate_world_deterministic() -> None:
    a = generate_world(10, 42)
    b = generate_world(10, 42)
    assert np.array_equal(a.cell_height, b.cell_height)
    assert np.array_equal(a.buildable, b.buildable)


def test_buildable_subset_of_land() -> None:
    w = generate_world(10, 0)
    assert not w.water_mask[w.buildable].any()


def test_candidates_only_buildable() -> None:
    env = CityEnv()
    env.reset(seed=123)
    assert env._world_layout is not None
    for idx in env.candidate_cells:
        r, c = int(idx // env.grid_size), int(idx % env.grid_size)
        assert env._world_layout.buildable[r, c]


def test_step_only_on_candidates() -> None:
    env = CityEnv(max_steps=10)
    env.reset(seed=0)
    target = int(env.candidate_cells[0])
    r, c = target // env.grid_size, target % env.grid_size
    action = 0 * NUM_ZONE_TYPES + int(Zone.RESIDENTIAL)
    obs, _, _, _, _ = env.step(action)
    assert env.grid[r, c] == Zone.RESIDENTIAL
    assert obs.shape[0] == env.observation_space.shape[0]


def test_world_always_has_buildable_land() -> None:
    """Regression: training must never see zero buildable cells (see terrain fallback)."""
    for s in range(512):
        w = generate_world(10, s)
        assert w.buildable.any()


def test_obs_includes_buildable() -> None:
    env = CityEnv()
    obs, _ = env.reset(seed=1)
    g = env.grid_size
    i0 = g * g * NUM_ZONE_TYPES
    bh = obs[i0 : i0 + g * g]
    assert np.all((bh == 0) | (bh == 1))
