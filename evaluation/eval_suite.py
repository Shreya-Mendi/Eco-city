"""Run evaluation suite, compare PPO to baselines, write eval_summary.json."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments import (
    main as run_all_experiments,
    run_experiment_1,
)


def ensure_default_model(base_path: str = "results/ppo_eco_city") -> Path:
    """Ensure `results/ppo_eco_city.zip` exists: copy from repo root or raise."""
    dest_zip = Path(base_path).with_suffix(".zip")
    if dest_zip.is_file():
        return dest_zip
    root_zip = Path("ppo_eco_city.zip")
    if root_zip.is_file():
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root_zip, dest_zip)
        print(f"Copied model: {root_zip} -> {dest_zip}")
        return dest_zip
    raise FileNotFoundError(
        "No model found at results/ppo_eco_city.zip or ./ppo_eco_city.zip. "
        "Train with: python training/train.py --timesteps 500000 --export "
        "or pass --train-if-missing 200000"
    )


def _summarize_exp1(exp1: dict[str, object]) -> dict[str, object]:
    baselines = ("random", "greedy", "heuristic")
    br = {k: float(exp1[k]["total_reward"]) for k in baselines}
    best_name = max(br, key=br.get)
    best_base = br[best_name]
    ppo_r = float(exp1["ppo"]["total_reward"])
    return {
        "baseline_rewards": br,
        "best_baseline_name": best_name,
        "best_baseline_reward": best_base,
        "ppo_reward": ppo_r,
        "ppo_beats_best_baseline": ppo_r > best_base,
        "margin_vs_best_baseline": ppo_r - best_base,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eco-City eval suite + summary JSON")
    parser.add_argument(
        "--model",
        type=str,
        default="results/ppo_eco_city",
        help="PPO checkpoint base path (SB3 adds .zip)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only experiment 1 (baselines vs PPO)",
    )
    parser.add_argument(
        "--train-if-missing",
        type=int,
        nargs="?",
        const=200_000,
        default=None,
        metavar="TIMESTEPS",
        help="If no model exists, train PPO for this many timesteps (default 200000)",
    )
    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    dest_zip = Path(args.model).with_suffix(".zip")
    if not dest_zip.is_file():
        if args.train_if_missing is not None:
            print("No checkpoint found; training with VecNormalize...")
            run_experiment_1(model_path=args.model, timesteps_train=args.train_if_missing)
        else:
            ensure_default_model(args.model)

    if args.quick:
        exp1 = run_experiment_1(model_path=args.model)
        out = {"experiment_1": exp1, "summary": _summarize_exp1(exp1)}
    else:
        run_all_experiments()
        with open(Path("results") / "experiment_results.json", encoding="utf-8") as f:
            full = json.load(f)
        out = {**full, "summary": _summarize_exp1(full["experiment_1"])}

    summary_path = Path("results") / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {summary_path}")
    s = out["summary"]
    print(
        f"PPO reward: {s['ppo_reward']:.2f} | best baseline ({s['best_baseline_name']}): "
        f"{s['best_baseline_reward']:.2f} | beats best: {s['ppo_beats_best_baseline']}"
    )


if __name__ == "__main__":
    main()
