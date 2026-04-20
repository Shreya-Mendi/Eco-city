# Cursor chat prompt — continue Eco-City RL after Colab + local clone

**Copy everything below the line into a new Cursor chat** (adjust branch name and paths if needed).

---

## Context for the assistant

I’m continuing the **Eco-City** RL project (grid city sim, **PPO** via Stable-Baselines3, **VecNormalize** for stable training, Three.js viewer). I trained on **Google Colab** using `training/train_eco_city_colab.ipynb` and **downloaded artifacts**. I also **cloned the GitHub repo** locally and want to align my machine with that workflow.

### What I already did (or will do)

1. **Clone** the repo: `https://github.com/Shreya-Mendi/Eco-city.git` (or my fork).
2. **Checkout** the branch where the latest code lives: **`REPLACE_WITH_BRANCH`** (e.g. `eval-tuning`, `main`, or the branch we just pushed).
3. **Colab downloads** — I have these files from Colab’s `results/` and `visualization/threejs/`:
   - `ppo_eco_city.zip`
   - `ppo_eco_city_vecnormalize.pkl` (must sit next to the zip for correct eval)
   - optionally `city_data.json` for the 3D viewer

### What I need help with

1. **Place downloaded files correctly** under the repo root:
   - `results/ppo_eco_city.zip`
   - `results/ppo_eco_city_vecnormalize.pkl`
   - optional: `visualization/threejs/city_data.json`

2. **Local setup:** Python venv, `pip install -r requirements.txt`, confirm imports work.

3. **Run evaluation** so I get fresh metrics:
   - Quick: `python evaluation/eval_suite.py --quick` → `results/eval_summary.json`
   - Full: `python evaluation/experiments.py` → `results/experiment_results.json`

4. **Optional:** static server for the viewer: `python -m http.server 8000 --directory visualization/threejs` and open `http://localhost:8000`.

5. If something breaks (missing `.pkl`, wrong paths, PPO scores weird), **debug using** `training/vec_env.py` (`load_ppo_for_eval`), `evaluation/experiments.py`, and the export path that uses `CityEnv._export_traj` + `training/train.py` rollout for `city_data.json`.

### Important implementation notes (for you, assistant)

- Evaluation expects **both** the SB3 **`.zip`** and **`{stem}_vecnormalize.pkl`** next to it when training used VecNormalize.
- `experiment_results.json` with **all PPO zeros** was an old artifact; fixed export used **`_export_traj`** so VecEnv autoreset doesn’t clear history before JSON export.
- Baseline comparison: higher **total_reward** is better (returns are often negative).

### Ask me to do next

Please walk me through **exact terminal commands** from a clean clone through **dropped Colab files → eval → viewer**, and call out anything that must match **branch / file layout** in this repo.

---

## Repo paths (reference)

| Topic | Path |
|--------|------|
| Colab notebook | `training/train_eco_city_colab.ipynb` |
| Train CLI | `training/train.py` |
| VecNormalize helpers | `training/vec_env.py` |
| Eval suite | `evaluation/eval_suite.py`, `evaluation/experiments.py` |
| Presentation outline | `presentation_outline_eco_city_rl.md` |
| Eval README | `evaluation/README_EVAL.md` (if present on branch) |

---

*Last intent: hand off after Colab training + Git clone; continue development and evaluation locally.*
