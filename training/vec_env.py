"""DummyVecEnv + VecNormalize for stable PPO value targets (reward/obs scaling)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.city_env import CityEnv

# Must match PPO `gamma` in `make_ppo_agent` (default 0.99).
VEC_GAMMA = 0.99


def make_vec_train_env(
    *,
    log_csv: Path | None = None,
    gamma: float = VEC_GAMMA,
    **city_kwargs: Any,
) -> VecNormalize:
    def _thunk() -> Monitor:
        inner = CityEnv(**city_kwargs)
        if log_csv is not None:
            return Monitor(inner, filename=str(log_csv))
        return Monitor(inner, filename=None)

    venv = DummyVecEnv([_thunk])
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )


def unwrap_city_env(vec_env: VecNormalize) -> CityEnv:
    mon = vec_env.venv.envs[0]
    inner = mon.env
    assert isinstance(inner, CityEnv)
    return inner


def save_vecnormalize(vec_env: VecNormalize, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vec_env.save(str(path))


def load_vec_eval_env(vecnorm_path: str | Path, **city_kwargs: Any) -> VecNormalize:
    def _thunk() -> Monitor:
        return Monitor(CityEnv(**city_kwargs), filename=None)

    venv = DummyVecEnv([_thunk])
    loaded = VecNormalize.load(str(vecnorm_path), venv)
    loaded.training = False
    loaded.norm_reward = False
    return loaded


def resolve_zip(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file() and p.suffix == ".zip":
        return p
    z = p.with_suffix(".zip")
    if z.is_file():
        return z
    z2 = Path(str(path) + ".zip")
    if z2.is_file():
        return z2
    return p


def vecnorm_path_next_to_model(model_zip: Path) -> Path | None:
    """Prefer `{stem}_vecnormalize.pkl`, then legacy `vec_normalize.pkl`."""
    for name in (f"{model_zip.stem}_vecnormalize.pkl", "vec_normalize.pkl"):
        p = model_zip.parent / name
        if p.is_file():
            return p
    return None


def vecnorm_save_path(model_save_path: str | Path) -> Path:
    """VecNormalize stats next to `model.save(...)` base path (no `.zip` suffix)."""
    p = Path(model_save_path)
    return p.parent / f"{p.stem}_vecnormalize.pkl"


def load_ppo_for_eval(
    model_path: str,
    *,
    vecnorm_path: str | Path | None = None,
    **city_kwargs: Any,
) -> tuple[PPO, VecNormalize | CityEnv]:
    """Load PPO with VecNormalize if a matching `.pkl` exists next to the `.zip`."""
    mp = resolve_zip(model_path)
    if not mp.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vnp: Path | None = None
    if vecnorm_path is not None:
        cand = Path(vecnorm_path)
        vnp = cand if cand.is_file() else None
    else:
        vnp = vecnorm_path_next_to_model(mp)

    if vnp is not None:
        venv = load_vec_eval_env(vnp, **city_kwargs)
        model = PPO.load(str(mp), env=venv)
        return model, venv

    env = CityEnv(**city_kwargs)
    model = PPO.load(str(mp), env=env)
    return model, env
