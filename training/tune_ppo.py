"""Small PPO hyperparameter grid with VecNormalize; keeps best checkpoint by rollout return."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import make_ppo_agent, train
from evaluation.experiments import _rollout_ppo
from training.callbacks import MetricsCallback
from training.vec_env import make_vec_train_env, save_vecnormalize, vecnorm_save_path

# name + SB3/PPO kwargs (subset supported by make_ppo_agent)
TUNING_CONFIGS: list[dict[str, object]] = [
    {"name": "default", "learning_rate": 3e-4, "ent_coef": 0.01, "n_steps": 2048, "batch_size": 64},
    {"name": "high_ent", "learning_rate": 3e-4, "ent_coef": 0.05, "n_steps": 2048, "batch_size": 64},
    {"name": "low_lr", "learning_rate": 1e-4, "ent_coef": 0.01, "n_steps": 2048, "batch_size": 64},
    {"name": "larger_minibatch", "learning_rate": 3e-4, "ent_coef": 0.01, "n_steps": 2048, "batch_size": 128},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=80_000, help="Train steps per config")
    parser.add_argument(
        "--out-best",
        type=str,
        default="results/ppo_eco_city_tuned",
        help="Where to save the best run (SB3 adds .zip)",
    )
    args = parser.parse_args()

    Path("results/tuning_runs").mkdir(parents=True, exist_ok=True)
    Path("logs/tuning").mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    best_reward: float | None = None
    best_name: str | None = None
    best_ckpt_dir: Path | None = None

    for cfg in TUNING_CONFIGS:
        name = str(cfg["name"])
        hparams = {k: v for k, v in cfg.items() if k != "name"}
        run_dir = Path("results/tuning_runs") / name
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_base = str(run_dir / "model")

        log_csv = Path("logs/tuning") / f"{name}_monitor.csv"
        vec_env = make_vec_train_env(log_csv=log_csv)
        model = make_ppo_agent(vec_env, tensorboard_log="logs/tuning", **hparams)
        train(model, total_timesteps=args.timesteps, callback=MetricsCallback())
        model.save(ckpt_base)
        save_vecnormalize(vec_env, vecnorm_save_path(ckpt_base))

        roll = _rollout_ppo(ckpt_base, max_steps=200, seed=42)
        reward = float(roll["total_reward"])
        results.append({"name": name, "hparams": hparams, "rollout_reward": reward, "ckpt": ckpt_base})
        print(f"[tune] {name}: rollout_reward={reward:.4f}")

        if best_reward is None or reward > best_reward:
            best_reward = reward
            best_name = name
            best_ckpt_dir = run_dir

    assert best_ckpt_dir is not None and best_name is not None

    best_zip = best_ckpt_dir / "model.zip"
    best_pkl = vecnorm_save_path(str(best_ckpt_dir / "model"))
    out_zip = Path(args.out_best).with_suffix(".zip")
    out_pkl = vecnorm_save_path(args.out_best)
    shutil.copy2(best_zip, out_zip)
    if best_pkl.is_file():
        shutil.copy2(best_pkl, out_pkl)

    summary = {
        "timesteps_per_run": args.timesteps,
        "best_config_name": best_name,
        "best_rollout_reward": best_reward,
        "best_checkpoint_copied_to": str(out_zip),
        "vecnormalize_copied_to": str(out_pkl) if out_pkl.is_file() else None,
        "all_runs": results,
    }
    out_json = Path("results") / "tuning_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Best: {best_name} ({best_reward:.4f}) -> {out_zip}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
