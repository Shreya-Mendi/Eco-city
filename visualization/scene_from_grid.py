"""Derive roads, buildings, and static scene data from grid + WorldLayout (visualization only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from env.dynamics import Zone
from env.terrain import WorldLayout, cell_center_world

# Visual scale factors (match legacy cityRenderer.js HEIGHTS vibe)
ZONE_BUILDING_HEIGHT = {
    int(Zone.EMPTY): 0.0,
    int(Zone.RESIDENTIAL): 2.0,
    int(Zone.COMMERCIAL): 1.6,
    int(Zone.INDUSTRIAL): 2.4,
    int(Zone.GREEN): 0.35,
    int(Zone.ROAD): 0.06,
    int(Zone.ENERGY): 1.1,
}


def roads_from_grid(grid: np.ndarray, layout: WorldLayout) -> list[dict[str, Any]]:
    """Flat road pads on terrain for ROAD cells that are buildable."""
    n = layout.grid_size
    cs = layout.cell_size
    out: list[dict[str, Any]] = []
    for r in range(n):
        for c in range(n):
            z = int(grid[r, c])
            if z != int(Zone.ROAD):
                continue
            if not layout.buildable[r, c]:
                continue
            x, z_w = cell_center_world(r, c, n, cs)
            out.append(
                {
                    "x": float(x),
                    "z": float(z_w),
                    "width": float(cs * 0.92),
                    "depth": float(cs * 0.92),
                    "y": 0.03,
                }
            )
    return out


def buildings_from_grid(
    grid: np.ndarray,
    layout: WorldLayout,
    *,
    population: float,
) -> list[dict[str, Any]]:
    """One building per occupied buildable cell; height scaled slightly by global density."""
    n = layout.grid_size
    cs = layout.cell_size
    density_scale = 1.0 + 0.15 * min(float(population) / 500.0, 1.0)
    out: list[dict[str, Any]] = []
    for r in range(n):
        for c in range(n):
            z = int(grid[r, c])
            if z == int(Zone.EMPTY) or z == int(Zone.ROAD):
                continue
            if not layout.buildable[r, c]:
                continue
            base_h = ZONE_BUILDING_HEIGHT.get(z, 0.5)
            if base_h <= 0:
                continue
            x, z_w = cell_center_world(r, c, n, cs)
            h = base_h * density_scale
            # Footprint: industrial wider
            foot = cs * (0.88 if z != int(Zone.INDUSTRIAL) else 0.95)
            out.append(
                {
                    "x": float(x),
                    "z": float(z_w),
                    "zone": z,
                    "height": float(h),
                    "footprint": float(foot),
                    "row": r,
                    "col": c,
                }
            )
    return out


def augment_snapshot(
    grid: np.ndarray,
    layout: WorldLayout,
    *,
    population: float,
) -> dict[str, Any]:
    """Road + building draw lists for one frame."""
    g = grid if isinstance(grid, np.ndarray) else np.array(grid, dtype=np.int32)
    return {
        "roads": roads_from_grid(g, layout),
        "buildings": buildings_from_grid(g, layout, population=population),
    }
