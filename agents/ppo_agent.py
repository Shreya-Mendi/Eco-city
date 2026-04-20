"""PPO training helpers using Stable-Baselines3."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.city_env import CityEnv


def make_ppo_agent(
    env: CityEnv | None = None,
    *,
    tensorboard_log: str = "logs/ppo",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    policy_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
) -> PPO:
    if env is None:
        env = CityEnv()
    kwargs: dict[str, Any] = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    if policy_kwargs is not None:
        kwargs["policy_kwargs"] = policy_kwargs
    if device is not None:
        kwargs["device"] = device
    return PPO(**kwargs)


def train(
    model: PPO,
    total_timesteps: int = 500_000,
    callback: BaseCallback | None = None,
) -> PPO:
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model


def save(model: PPO, path: str = "results/ppo_eco_city") -> None:
    model.save(path)


def load(path: str, env: CityEnv) -> PPO:
    from pathlib import Path

    p = Path(path)
    if p.suffix != ".zip":
        z = p.with_suffix(".zip")
        if z.is_file():
            path = str(z)
    return PPO.load(path, env=env)
