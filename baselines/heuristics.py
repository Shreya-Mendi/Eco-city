"""Baseline policies for Eco-City."""

from __future__ import annotations

import numpy as np

from env.city_env import CityEnv, NUM_ZONE_TYPES
from env.dynamics import Zone


class RandomAgent:
    def predict(self, obs: np.ndarray, env: CityEnv) -> int:
        return int(env.action_space.sample())


class GreedyAgent:
    """Always place Residential on the first candidate cell."""

    def predict(self, obs: np.ndarray, env: CityEnv) -> int:
        return int(0 * NUM_ZONE_TYPES + Zone.RESIDENTIAL)


class HeuristicAgent:
    """Rule-based zoning using global metrics (raw values from env metrics)."""

    def predict(self, obs: np.ndarray, env: CityEnv) -> int:
        pollution = float(env.metrics["pollution"])
        traffic = float(env.metrics["traffic"])
        energy_balance = float(env.metrics["energy_supply"] - env.metrics["energy_demand"])

        if pollution > 50:
            zone = Zone.GREEN
        elif energy_balance < 0:
            zone = Zone.ENERGY
        elif traffic > 30:
            zone = Zone.ROAD
        else:
            zone = Zone.RESIDENTIAL

        return int(0 * NUM_ZONE_TYPES + zone)
