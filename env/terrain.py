"""Deterministic terrain, water, and buildable masks shared by CityEnv and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Must match Three.js `CELL_SIZE` in visualization/threejs/cityRenderer.js
DEFAULT_CELL_SIZE = 2.0


def _bilinear_resize(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize 2D array with bilinear interpolation (numpy only)."""
    in_h, in_w = arr.shape
    if in_h == out_h and in_w == out_w:
        return arr.astype(np.float64)
    y = np.linspace(0.0, in_h - 1.0, out_h, dtype=np.float64)
    x = np.linspace(0.0, in_w - 1.0, out_w, dtype=np.float64)
    yi = np.floor(y).astype(np.int64)
    xi = np.floor(x).astype(np.int64)
    yi = np.clip(yi, 0, in_h - 2)
    xi = np.clip(xi, 0, in_w - 2)
    dy = y - yi
    dx = x - xi
    out = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            a = arr[yi[i], xi[j]]
            b = arr[yi[i], xi[j] + 1]
            c = arr[yi[i] + 1, xi[j]]
            d = arr[yi[i] + 1, xi[j] + 1]
            t = dy[i]
            u = dx[j]
            out[i, j] = (1 - u) * ((1 - t) * a + t * c) + u * ((1 - t) * b + t * d)
    return out


@dataclass(frozen=True)
class WorldLayout:
    """Per-episode static world: heights, water, and where zoning is allowed."""

    grid_size: int
    seed: int
    cell_size: float
    cell_height: np.ndarray  # (N, N) in [0, 1]
    water_mask: np.ndarray  # (N, N) bool
    buildable: np.ndarray  # (N, N) bool
    heightmap: np.ndarray  # (R, R) in [0, 1], R >= N for smoother terrain mesh
    lake_center_row: float
    lake_center_col: float
    lake_radius_cells: float
    water_level: float  # normalized; mesh Y offset derived in viewer
    slope: np.ndarray  # (N, N) float


def generate_world(
    grid_size: int,
    seed: int,
    *,
    cell_size: float = DEFAULT_CELL_SIZE,
    terrain_upsample: int = 4,
    lake_center_row: float | None = None,
    lake_center_col: float | None = None,
    lake_radius_cells: float = 2.4,
    max_slope_buildable: float = 0.42,
    max_height_buildable: float = 0.82,
) -> WorldLayout:
    """
    Build height field, mark lake cells, and compute buildable = not water, not too steep, not peak.

    Roads use the same buildable mask as other zones (v1).
    """
    rng = np.random.default_rng(seed)
    low_n = max(3, grid_size // 3)
    low = rng.uniform(0.15, 0.95, size=(low_n, low_n))
    # Ridge toward +row for a simple "mountain" edge
    ridge = np.linspace(0.0, 0.35, grid_size, dtype=np.float64)[:, None]
    ridge = np.broadcast_to(ridge, (grid_size, grid_size))
    base = _bilinear_resize(low, grid_size, grid_size)
    cell_height = np.clip(base * 0.75 + ridge[:, :], 0.0, 1.0).astype(np.float64)

    if lake_center_row is None:
        lake_center_row = grid_size * 0.22 + 0.5
    if lake_center_col is None:
        lake_center_col = grid_size * 0.28 + 0.5

    rr, cc = np.indices((grid_size, grid_size))
    dist = np.sqrt((rr.astype(np.float64) - lake_center_row) ** 2 + (cc.astype(np.float64) - lake_center_col) ** 2)
    water_mask = dist < lake_radius_cells

    gy, gx = np.gradient(cell_height)
    slope = np.sqrt(gx**2 + gy**2)
    smax = float(np.max(slope)) + 1e-6
    slope_norm = slope / smax

    buildable = (
        (~water_mask)
        & (slope_norm < max_slope_buildable)
        & (cell_height < max_height_buildable)
    )

    res = grid_size * terrain_upsample
    heightmap = _bilinear_resize(cell_height.astype(np.float64), res, res)

    return WorldLayout(
        grid_size=grid_size,
        seed=seed,
        cell_size=cell_size,
        cell_height=cell_height.astype(np.float32),
        water_mask=water_mask,
        buildable=buildable,
        heightmap=heightmap.astype(np.float32),
        lake_center_row=float(lake_center_row),
        lake_center_col=float(lake_center_col),
        lake_radius_cells=float(lake_radius_cells),
        water_level=0.12,
        slope=slope_norm.astype(np.float32),
    )


def cell_center_world(row: int, col: int, grid_size: int, cell_size: float) -> tuple[float, float]:
    """World (x, z) for cell center; y is terrain height at viewer."""
    offset = grid_size * cell_size / 2.0
    x = col * cell_size - offset + cell_size / 2.0
    z = row * cell_size - offset + cell_size / 2.0
    return x, z


def world_layout_to_json_dict(layout: WorldLayout) -> dict[str, Any]:
    """Subset of WorldLayout for JSON `static` block."""
    n = layout.grid_size
    return {
        "cellSize": layout.cell_size,
        "gridSize": n,
        "worldExtent": n * layout.cell_size,
        "terrain": {
            "resolution": int(layout.heightmap.shape[0]),
            "heights": layout.heightmap.tolist(),
        },
        "water": {
            "kind": "circle",
            "centerRow": layout.lake_center_row,
            "centerCol": layout.lake_center_col,
            "radiusCells": layout.lake_radius_cells,
            "level": layout.water_level,
        },
        "buildable": layout.buildable.astype(np.int32).tolist(),
    }
