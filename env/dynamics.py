"""Grid transition rules and metric updates."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np


class Zone(IntEnum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    GREEN = 4
    ROAD = 5
    ENERGY = 6


def count_zones(grid: np.ndarray) -> dict[int, int]:
    counts = {z: 0 for z in Zone}
    flat = grid.ravel()
    for v in flat:
        z = int(v)
        if z in counts:
            counts[z] += 1
    return counts


def update_metrics(
    grid: np.ndarray,
    metrics: dict[str, Any],
    *,
    industrial_emission_factor: float = 1.0,
    residential_pop_multiplier: float = 1.0,
) -> dict[str, Any]:
    counts = count_zones(grid)

    metrics["population"] = (
        counts[Zone.RESIDENTIAL] * 10 * residential_pop_multiplier
        + counts[Zone.COMMERCIAL] * 5
    )
    metrics["energy_demand"] = (
        counts[Zone.RESIDENTIAL] * 3
        + counts[Zone.INDUSTRIAL] * 8
        + counts[Zone.COMMERCIAL] * 4
    )
    metrics["energy_supply"] = counts[Zone.ENERGY] * 15
    metrics["pollution"] = max(
        0,
        counts[Zone.INDUSTRIAL] * 5 * industrial_emission_factor - counts[Zone.GREEN] * 3,
    )
    raw_traffic = metrics["population"] * 0.1 - counts[Zone.ROAD] * 2
    metrics["traffic"] = max(0.0, float(raw_traffic))
    return metrics
