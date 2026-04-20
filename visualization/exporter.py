"""Export city state history to JSON for the Three.js viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from env.city_env import CityEnv
from env.terrain import generate_world, world_layout_to_json_dict
from visualization.scene_from_grid import augment_snapshot


def export(env: CityEnv, output_path: str) -> None:
    layout = env._world_layout
    if layout is None:
        seed = 0
        if env.state_history:
            seed = int(env.state_history[0].get("terrain_seed", 0))
        layout = generate_world(env.grid_size, seed)

    history: list[dict[str, Any]] = []
    for snapshot in env.state_history:
        grid = snapshot["grid"]
        if isinstance(grid, np.ndarray):
            g = grid
            grid_list = grid.tolist()
        else:
            g = np.array(grid, dtype=np.int32)
            grid_list = grid

        extra = augment_snapshot(g, layout, population=float(snapshot["population"]))
        history.append(
            {
                "step": snapshot["step"],
                "grid": grid_list,
                "population": float(snapshot["population"]),
                "pollution": float(snapshot["pollution"]),
                "traffic": float(snapshot["traffic"]),
                "energy_balance": float(snapshot["energy_balance"]),
                "reward": float(snapshot["reward"]),
                "terrain_seed": int(snapshot.get("terrain_seed", layout.seed)),
                "roads": extra["roads"],
                "buildings": extra["buildings"],
            }
        )

    payload: dict[str, Any] = {
        "formatVersion": "1.0",
        "static": world_layout_to_json_dict(layout),
        "history": history,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Exported {len(history)} steps to {output_path}")
