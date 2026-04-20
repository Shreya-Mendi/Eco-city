# Evaluation & tuning

## Default checkpoint

Training writes:

- `results/ppo_eco_city.zip`
- `results/ppo_eco_city_vecnormalize.pkl` (when using `training/train.py` or the Colab notebook with VecNormalize)

A copy of an older checkpoint may exist at the repo root as `ppo_eco_city.zip`; `eval_suite.py` copies it into `results/` if needed.

## Quick eval (baselines vs PPO)

```bash
python evaluation/eval_suite.py --quick
```

Writes `results/eval_summary.json` (includes `summary.ppo_beats_best_baseline` and margins).

Full suite (all four experiments):

```bash
python evaluation/eval_suite.py
```

Same as `python evaluation/experiments.py` plus `eval_summary.json`.

## If PPO underperforms heuristics

1. **Observation/reward scale:** Ensure training used **VecNormalize** (`DummyVecEnv` stack in `training/vec_env.py`).
2. **Train longer:** e.g. `--timesteps 500000` locally or on Colab GPU.
3. **Hyperparameter grid:** run a small search:

```bash
python training/tune_ppo.py --timesteps 80000
```

Produces `results/tuning_results.json` and copies the best run to `results/ppo_eco_city_tuned.zip` (+ `_vecnormalize.pkl`).

Point evaluations at the tuned model:

```bash
python evaluation/eval_suite.py --quick --model results/ppo_eco_city_tuned
```
