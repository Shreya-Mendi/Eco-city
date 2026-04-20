"""CLI entry point for PPO training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import make_ppo_agent, save, train
from env.city_env import CityEnv
from training.callbacks import MetricsCallback
from training.vec_env import make_vec_train_env, save_vecnormalize, unwrap_city_env, vecnorm_save_path
from visualization.exporter import export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs/ppo")
    parser.add_argument("--save-path", type=str, default="results/ppo_eco_city")
    args = parser.parse_args()

    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_train_env(log_csv=log_dir / "monitor.csv")
    base = unwrap_city_env(vec_env)
    model = make_ppo_agent(vec_env, tensorboard_log="logs")

    callback = MetricsCallback()
    train(model, total_timesteps=args.timesteps, callback=callback)
    save(model, args.save_path)
    save_vecnormalize(vec_env, vecnorm_save_path(args.save_path))

    if args.export:
        export(base, "visualization/threejs/city_data.json")


if __name__ == "__main__":
    main()
