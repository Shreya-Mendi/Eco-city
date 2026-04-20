"""Experiment runners from plan.md (RL vs baselines, reward sensitivity, generalization, analysis)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.monitor import Monitor

from agents.ppo_agent import load, make_ppo_agent, train
from baselines.heuristics import GreedyAgent, HeuristicAgent, RandomAgent
from env.city_env import CityEnv
from training.callbacks import MetricsCallback


def _resolve_model_path(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    z = p.with_suffix(".zip")
    if z.is_file():
        return str(z)
    z2 = Path(str(path) + ".zip")
    if z2.is_file():
        return str(z2)
    return str(path)


def _rollout_baseline(
    agent: object,
    name: str,
    *,
    max_steps: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    env = CityEnv(max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.predict(obs, env)  # type: ignore[attr-defined]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
    return {
        "agent": name,
        "total_reward": total_reward,
        "final_population": info.get("population", 0.0),
        "final_pollution": info.get("pollution", 0.0),
        "final_traffic": info.get("traffic", 0.0),
        "final_energy_balance": info.get("energy_balance", 0.0),
    }


def _rollout_ppo(
    model_path: str,
    *,
    max_steps: int = 200,
    seed: int = 42,
    env_kwargs: dict | None = None,
) -> dict[str, object]:
    env = CityEnv(**(env_kwargs or {}), max_steps=max_steps)
    mp = _resolve_model_path(model_path)
    model = load(mp, env)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
    return {
        "agent": "ppo",
        "total_reward": total_reward,
        "final_population": info.get("population", 0.0),
        "final_pollution": info.get("pollution", 0.0),
        "final_traffic": info.get("traffic", 0.0),
        "final_energy_balance": info.get("energy_balance", 0.0),
    }


def run_experiment_1(
    model_path: str = "results/ppo_eco_city",
    *,
    max_steps: int = 200,
    timesteps_train: int | None = None,
) -> dict[str, object]:
    """RL vs baselines on cumulative reward and final city metrics."""
    results: dict[str, object] = {}
    results["random"] = _rollout_baseline(RandomAgent(), "random", max_steps=max_steps)
    results["greedy"] = _rollout_baseline(GreedyAgent(), "greedy", max_steps=max_steps)
    results["heuristic"] = _rollout_baseline(HeuristicAgent(), "heuristic", max_steps=max_steps)

    mp = model_path
    if not Path(_resolve_model_path(mp)).is_file():
        if timesteps_train is None:
            timesteps_train = 200_000
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        inner = CityEnv(max_steps=max_steps)
        log_dir = Path("logs/experiments")
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(inner, filename=str(log_dir / "exp1_monitor.csv"))
        model = make_ppo_agent(env, tensorboard_log="logs")
        train(model, total_timesteps=timesteps_train, callback=MetricsCallback())
        model.save(mp)

    results["ppo"] = _rollout_ppo(mp, max_steps=max_steps)
    return results


def run_experiment_2(
    *,
    train_timesteps: int = 100_000,
    model_dir: str = "results",
) -> dict[str, object]:
    """Train PPO with sustainability-focused vs growth-focused reward weights."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    out: dict[str, object] = {}

    for label, alpha, beta in [
        ("sustainability", 0.5, 2.0),
        ("growth", 2.0, 0.3),
    ]:
        inner = CityEnv(alpha=alpha, beta=beta)
        env = Monitor(inner, filename=str(Path("logs") / f"exp2_{label}.csv"))
        model = make_ppo_agent(env, tensorboard_log="logs")
        train(model, total_timesteps=train_timesteps, callback=MetricsCallback())
        save_path = str(Path(model_dir) / f"ppo_eco_city_{label}")
        model.save(save_path)

        ev = CityEnv(alpha=alpha, beta=beta)
        obs, _ = ev.reset(seed=42)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = ev.step(int(action))
            total_reward += float(reward)

        grid = ev.grid.copy()
        flat = grid.ravel().tolist()
        dist = Counter(flat)
        out[label] = {
            "alpha": alpha,
            "beta": beta,
            "rollout_reward": total_reward,
            "zone_counts": {int(z): int(dist.get(z, 0)) for z in range(7)},
            "final_grid": grid.tolist(),
        }
    return out


def run_experiment_3(
    model_path: str = "results/ppo_eco_city",
    *,
    max_steps: int = 200,
) -> dict[str, object]:
    """Train on default dynamics; evaluate on shifted growth / industrial emissions."""
    mp = _resolve_model_path(model_path)
    if not Path(mp).is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path!r}. Train with training/train.py or the Colab notebook first."
        )

    baseline = _rollout_ppo(model_path, max_steps=max_steps)
    high_growth = _rollout_ppo(
        model_path,
        max_steps=max_steps,
        env_kwargs={"residential_pop_multiplier": 1.5},
    )
    high_emissions = _rollout_ppo(
        model_path,
        max_steps=max_steps,
        env_kwargs={"industrial_emission_factor": 1.5},
    )
    return {
        "train_match_eval": baseline,
        "test_high_pop_growth": high_growth,
        "test_high_industrial_emissions": high_emissions,
    }


def run_experiment_4(
    model_path: str = "results/ppo_eco_city",
    *,
    max_steps: int = 200,
) -> dict[str, object]:
    """Policy behavior: per-step zone-type counts over one deterministic PPO rollout."""
    mp = _resolve_model_path(model_path)
    if not Path(mp).is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path!r}. Train first, then rerun experiment 4."
        )

    env = CityEnv(max_steps=max_steps)
    model = load(mp, env)
    obs, _ = env.reset(seed=42)
    terminated = truncated = False
    per_step: list[dict[str, object]] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        dist = Counter(env.grid.ravel().tolist())
        per_step.append(
            {
                "step": env.step_count,
                "zone_counts": {int(z): int(dist.get(z, 0)) for z in range(7)},
            }
        )
    return {"per_step_zone_counts": per_step}


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    exp1 = run_experiment_1()
    exp2 = run_experiment_2()
    exp3 = run_experiment_3()
    exp4 = run_experiment_4()

    out = {
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4,
    }
    out_path = Path("results/experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote aggregated results to {out_path}")


if __name__ == "__main__":
    main()
