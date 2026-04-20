"""Experiment runners from plan.md (RL vs baselines, reward sensitivity, generalization, analysis)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.vec_env import VecNormalize

from agents.ppo_agent import make_ppo_agent, save, train
from baselines.heuristics import GreedyAgent, HeuristicAgent, RandomAgent
from env.city_env import CityEnv
from training.callbacks import MetricsCallback
from training.vec_env import (
    load_ppo_for_eval,
    make_vec_train_env,
    save_vecnormalize,
    vecnorm_save_path,
)


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


def _unwrap_city(wrapped: VecNormalize | CityEnv) -> CityEnv:
    if isinstance(wrapped, VecNormalize):
        return wrapped.venv.envs[0].env
    return wrapped


def _rollout_ppo(
    model_path: str,
    *,
    max_steps: int = 200,
    seed: int = 42,
    env_kwargs: dict | None = None,
) -> dict[str, object]:
    mp = _resolve_model_path(model_path)
    kw = dict(env_kwargs or {})
    kw.setdefault("max_steps", max_steps)
    model, wrapped = load_ppo_for_eval(mp, **kw)

    if isinstance(wrapped, VecNormalize):
        if seed is not None:
            wrapped.venv.seed(seed)
        obs = wrapped.reset()
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = wrapped.step(action)
            total_reward += float(rewards[0])
            if dones[0]:
                info = infos[0]
                break
    else:
        env = wrapped
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
    city = _unwrap_city(wrapped)
    return {
        "agent": "ppo",
        "total_reward": total_reward,
        "final_population": info.get("population", 0.0),
        "final_pollution": info.get("pollution", 0.0),
        "final_traffic": info.get("traffic", 0.0),
        "final_energy_balance": info.get("energy_balance", 0.0),
        "final_grid": city.grid.tolist(),
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
        log_dir = Path("logs/experiments")
        log_dir.mkdir(parents=True, exist_ok=True)
        vec_env = make_vec_train_env(log_csv=log_dir / "exp1_monitor.csv", max_steps=max_steps)
        model = make_ppo_agent(vec_env, tensorboard_log="logs")
        train(model, total_timesteps=timesteps_train, callback=MetricsCallback())
        model.save(mp)
        save_vecnormalize(vec_env, vecnorm_save_path(mp))

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
        log_csv = Path("logs") / f"exp2_{label}.csv"
        vec_env = make_vec_train_env(log_csv=log_csv, alpha=alpha, beta=beta)
        model = make_ppo_agent(vec_env, tensorboard_log="logs")
        train(model, total_timesteps=train_timesteps, callback=MetricsCallback())
        save_path = str(Path(model_dir) / f"ppo_eco_city_{label}")
        model.save(save_path)
        save_vecnormalize(vec_env, vecnorm_save_path(save_path))

        roll = _rollout_ppo(save_path, env_kwargs={"alpha": alpha, "beta": beta})
        flat = [c for row in roll["final_grid"] for c in row]
        dist = Counter(flat)
        out[label] = {
            "alpha": alpha,
            "beta": beta,
            "rollout_reward": roll["total_reward"],
            "zone_counts": {int(z): int(dist.get(z, 0)) for z in range(7)},
            "final_grid": roll["final_grid"],
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

    model, wrapped = load_ppo_for_eval(mp, max_steps=max_steps)
    city = _unwrap_city(wrapped)
    per_step: list[dict[str, object]] = []

    if isinstance(wrapped, VecNormalize):
        wrapped.venv.seed(42)
        obs = wrapped.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, _infos = wrapped.step(action)
            dist = Counter(city.grid.ravel().tolist())
            per_step.append(
                {
                    "step": city.step_count,
                    "zone_counts": {int(z): int(dist.get(z, 0)) for z in range(7)},
                }
            )
            if dones[0]:
                break
    else:
        obs, _ = wrapped.reset(seed=42)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = wrapped.step(int(action))
            dist = Counter(city.grid.ravel().tolist())
            per_step.append(
                {
                    "step": city.step_count,
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
