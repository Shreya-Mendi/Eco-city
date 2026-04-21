"""Run N rollouts with different seeds and save the top-K trajectories for the viewer.

Each saved rollout uses the enriched JSON format (formatVersion + static terrain +
history with roads/buildings), matching visualization/exporter.export().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.vec_env import VecNormalize

from env.city_env import CityEnv
from env.terrain import WorldLayout, generate_world, world_layout_to_json_dict
from training.vec_env import load_ppo_for_eval, unwrap_city_env
from visualization.scene_from_grid import augment_snapshot


def _enrich_history(traj: list[dict[str, Any]], layout: WorldLayout) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for snap in traj:
        grid = np.array(snap["grid"], dtype=np.int32)
        extra = augment_snapshot(grid, layout, population=float(snap["population"]))
        enriched.append(
            {
                "step": int(snap["step"]),
                "grid": grid.tolist(),
                "population": float(snap["population"]),
                "pollution": float(snap["pollution"]),
                "traffic": float(snap["traffic"]),
                "energy_balance": float(snap["energy_balance"]),
                "reward": float(snap["reward"]),
                "terrain_seed": int(snap.get("terrain_seed", layout.seed)),
                "roads": extra["roads"],
                "buildings": extra["buildings"],
            }
        )
    return enriched


def _write_enriched_rollout(
    path: str,
    traj: list[dict[str, Any]],
    layout: WorldLayout,
) -> None:
    payload = {
        "formatVersion": "1.0",
        "static": world_layout_to_json_dict(layout),
        "history": _enrich_history(traj, layout),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _rollout_once(
    model,
    wrapped: VecNormalize | CityEnv,
    seed: int,
) -> tuple[float, list[dict[str, Any]], WorldLayout]:
    """Deterministic rollout; returns (total_reward, trajectory, terrain_layout)."""
    if isinstance(wrapped, VecNormalize):
        wrapped.training = False
        wrapped.norm_reward = False
        wrapped.venv.seed(seed)
        base = unwrap_city_env(wrapped)
        base._export_traj = []
        obs = wrapped.reset()
        # Capture layout right after reset before any autoreset replaces it.
        assert base._world_layout is not None
        layout = base._world_layout
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = wrapped.step(action)
            total += float(rewards[0])
            if dones[0]:
                break
        traj = list(base._export_traj or [])
        base._export_traj = None
        return total, traj, layout

    env = wrapped
    env._export_traj = []
    obs, _ = env.reset(seed=seed)
    assert env._world_layout is not None
    layout = env._world_layout
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(int(action))
        total += float(reward)
    traj = list(env._export_traj or [])
    env._export_traj = None
    return total, traj, layout


def _fallback_layout(traj: list[dict[str, Any]], grid_size: int) -> WorldLayout:
    """If we couldn't capture the layout, regenerate from the stored terrain seed."""
    seed = int(traj[0].get("terrain_seed", 0)) if traj else 0
    return generate_world(grid_size, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/ppo_eco_city", help="PPO checkpoint base path (no .zip)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of rollouts to run")
    parser.add_argument("--top-k", type=int, default=3, help="How many best rollouts to keep")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        default="visualization/threejs/rollouts",
        help="Where to write run_{i}.json + index.json",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, wrapped = load_ppo_for_eval(args.model, max_steps=args.max_steps)
    inner = unwrap_city_env(wrapped) if isinstance(wrapped, VecNormalize) else wrapped
    grid_size = inner.grid_size

    runs: list[dict[str, Any]] = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        total, traj, layout = _rollout_once(model, wrapped, seed=seed)
        if layout is None:
            layout = _fallback_layout(traj, grid_size)
        runs.append({"seed": seed, "total_reward": total, "traj": traj, "layout": layout})
        print(f"[rollout] seed={seed} reward={total:.4f} steps={len(traj)}")

    runs.sort(key=lambda r: r["total_reward"], reverse=True)
    top = runs[: args.top_k]

    for f in out_dir.glob("run_*.json"):
        f.unlink()

    index: list[dict[str, Any]] = []
    for rank, r in enumerate(top):
        fname = f"run_{rank}.json"
        _write_enriched_rollout(str(out_dir / fname), r["traj"], r["layout"])
        index.append(
            {
                "file": fname,
                "rank": rank,
                "seed": r["seed"],
                "total_reward": r["total_reward"],
                "steps": len(r["traj"]),
            }
        )

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Saved top {len(top)} of {args.episodes} to {out_dir}")


if __name__ == "__main__":
    main()
