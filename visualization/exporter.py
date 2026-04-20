"""Export city state history to JSON for the Three.js viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from env.city_env import CityEnv


def export(env: CityEnv, output_path: str) -> None:
    history: list[dict[str, Any]] = []
    for snapshot in env.state_history:
        grid = snapshot["grid"]
        if isinstance(grid, np.ndarray):
            grid_list = grid.tolist()
        else:
            grid_list = grid
        history.append(
            {
                "step": snapshot["step"],
                "grid": grid_list,
                "population": snapshot["population"],
                "pollution": snapshot["pollution"],
                "traffic": snapshot["traffic"],
                "energy_balance": snapshot["energy_balance"],
                "reward": snapshot["reward"],
            }
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"Exported {len(history)} steps to {output_path}")
