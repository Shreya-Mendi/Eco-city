# Eco-City RL — 7-Minute Presentation Outline
**Course:** AIPI 590 (Modern RL) | **Format:** Copy each `##` block into one slide (title + bullets/tables)

**Timing:** ~50s × 8 content slides + title/thanks ≈ 7 min.

---

## Slide 1 — Title

**Eco-City RL**  
*Learning sustainable urban zoning policies with PPO — reward design, alignment, and the path from toy grid to digital twin.*

**Tech stack:**  
`PPO · Stable-Baselines3` `Gymnasium` `VecNormalize` `Three.js viewer` `Google Colab GPU (A100)`

| Task | Algorithm | State Space | Action Space | Horizon |
|------|-----------|-------------|--------------|---------|
| City zoning under multi-objective reward | PPO (SB3) | **804-d** (10×10 zone one-hot + buildable mask + globals) | **35** discrete (5 candidate cells × 7 zone types) | 200 steps |

*Repo:* `github.com/Shreya-Mendi/Eco-city` · *Branch:* `colab-local-handoff`

---

## Slide 2 — Applied use case (RL for Impact)

**Problem statement:** urban land-use planning is a sequential, multi-objective decision problem where early zoning choices constrain long-horizon outcomes (livability, pollution, congestion, grid load).

**Why it matters (real world):**
- Cities produce **~70%** of global CO₂ and host **>55%** of the population (UN Habitat).
- Land-use decisions lock in emissions, traffic, and equity outcomes for **decades**.
- Standard urban-planning tools are **simulation-based** (e.g., UrbanSim, ABM-style models) but lack optimized policies.

**Why RL (vs. supervised / heuristics):**
- **Sequential:** path-dependent — you cannot retroactively un-zone.
- **Delayed reward:** welfare emerges over many steps.
- **No labeled dataset:** there is no ground-truth "optimal" city per state.
- **Policy vs. prediction:** we want a **decision rule**, not a forecaster.

---

## Slide 3 — MDP Formulation

**State · Action · Transition · Reward · Termination**

| **STATE s** | **ACTION a** | **TRANSITION** | **REWARD r** | **EPISODE END** |
|-------------|--------------|----------------|--------------|-----------------|
| 10×10 grid one-hot (7 zones) | Discrete `{0..34}` | Rule-based deterministic: | Weighted objective (α,β,γ,δ): | 200 steps (fixed) |
| Per-cell **buildable** mask (100-d) | `cell_slot ∈ {0..4}` × | • industry ↑ pollution | `+α · livability` | No early termination |
| Global: **population, pollution, traffic, energy balance** (4-d, normalized) | `zone_type ∈ {EMPTY, RES, COM, IND, GREEN, ROAD, ENERGY}` | • green ↓ pollution | `−β · pollution` | Terrain regenerated on reset |
| **Total: 804-d float32** | Only buildable cells sampled as candidates | • roads ↑ traffic pressure | `−γ · traffic` | |
| | | • energy zones ↑ supply, residential ↑ demand | `−δ · |supply−demand|` | |
| | | | `+0.01 · built_cells` (post-fix) | |

---

## Slide 4 — PPO Training Setup

**Policy architecture · hyperparameters · infrastructure**

**Policy Architecture**
- Type: **MLP Actor-Critic** (SB3 `MlpPolicy`)
- Hidden: `[64, 64]` ReLU, separate heads
- Output: **Categorical** over 35 discrete actions
- Observation normalization via **VecNormalize** (running mean/std, clip 10)

**PPO Hyperparameters**
| lr | `n_steps` | batch | `γ` | `λ_GAE` | `ε_clip` | `ent_coef` | epochs |
|----|-----------|-------|-----|---------|----------|------------|--------|
| 3e-4 | 2048 | 64 | 0.99 | 0.95 | 0.2 | 0.01 | 10 |

**Training budget:** 500k timesteps, ~20 min on Colab A100; DummyVecEnv (single env + Monitor).

**Value-function health:** `explained_variance` grew from ~0 to **~0.7** after VecNormalize calibration; `ep_rew_mean` monotonically rising (−6k → −1.6k over 500k steps).

---

## Slide 5 — Reward Function Design

Why each term exists — one failure gap per component.

| **Primary signal · Livability** | **Tracking signal · Pollution penalty** | **Flow signal · Traffic penalty** |
|---|---|---|
| `α · population/100` (uncapped post-fix). Rewards actual residents. Without this, no reason to build. | `−β · total_pollution`. Drives agent to avoid industrial sprawl + to intersperse green. | `−γ · total_traffic`. Pressure based on residential/commercial density near roads; penalises unbalanced layouts. |

| **Grid-load signal · Energy mismatch** | **Anti-laziness · Build bonus** |
|---|---|
| `−δ · |supply − demand|`. Forces agent to co-plan ENERGY zones alongside demand-generating zones. | `+0.01 × built_cells`. Small, uniform bonus so the empty grid is strictly worse than any reasonable city. Prevents the do-nothing equilibrium. |

*Weights:* default `α=1.0, β=1.2, γ=0.7, δ=0.8`. Experiment 2 sweeps `sustainability (α=0.5, β=2.0)` vs `growth (α=2.0, β=0.3)`.

---

## Slide 6 — Results

**PPO vs. baselines (cumulative reward, 200 steps, deterministic eval):**

| Agent | Total Reward | Notes |
|-------|-------------:|-------|
| random | −4,750 | places arbitrary zones, accumulates penalties |
| greedy | −6,324 | argmax livability, ignores pollution/traffic |
| **heuristic** (best baseline) | **−1,911** | hand-coded balanced placement |
| **PPO (ours, post-fix)** | **+21.08** | **+1,932** vs. best baseline |

**Hyperparameter sweep** (`training/tune_ppo.py`, 80k steps each, 4 configs):

| Config | Reward | Takeaway |
|--------|-------:|----------|
| **low_lr (1e-4)** | **+21.28** | ✓ winner — stabilises aggressive updates seen in training |
| default (3e-4) | −265.67 | baseline |
| larger_minibatch (128) | −48.38 | marginal |
| high_ent (0.05) | −2,906 | excessive exploration → hits penalty states |

**Top-3 rollouts out of 20 seeds:** `seed 18: +43.70` · `seed 10: +20.45` · `seed 17: +7.84` (all positive → policy is **reproducibly** better than baselines, not a lucky eval).

---

## Slide 7 — Reward Misalignment: What Broke & Why

**01 — Reward hacking: the "do nothing" Nash equilibrium**  
First PPO run converged to an **all-empty grid** (total_reward = 0). Livability was clipped at 1.0 while pollution/traffic/energy penalties were unbounded, so any non-trivial city scored negative. Inaction dominated.  
**Fix:** uncapped livability (`pop/100`, no clip) + `+0.01 × built_cells` build bonus.

**02 — Post-fix: the "clean but sterile" minimalist policy**  
Experiment 4 (per-step zone distribution) shows the learned policy stabilises at **8 ROAD + 3 GREEN cells** — no residential, commercial, or industrial. Zero population means zero pollution, zero traffic, zero energy mismatch. The agent is "farming" the build bonus.  
**Reading:** still partial reward hacking, but now quantifiable. Next iteration: make livability scale *per-resident* not per-cell.

**03 — Training diagnostics that predicted the fix**  
`clip_fraction ≈ 0.45` and `approx_kl ≈ 0.06` across training → PPO updates routinely saturated the trust region. HP sweep confirmed: `low_lr = 1e-4` gives **+21.28** vs `3e-4` at `−265`. Textbook signal-driven tuning.

---

## Slide 8 — Alignment, Safety & Scaling to Real Cities

**What transfers to a real city digital twin:**

| **Challenge** | **Mitigation in our prototype** | **Real-world extension** |
|---|---|---|
| **Reward specification** — "sustainability" is a proxy | Multi-objective scalar (α,β,γ,δ); ablation (Exp. 2) shows sensitivity | Constrained RL (CMDP); reward modelling from stakeholder preferences |
| **Distribution shift** — dynamics change over years | Exp. 3: tested on `pop_multiplier=1.5` and `emission_factor=1.5` → reward stable at +21 | Domain randomisation across growth rates, climate regimes; online fine-tuning |
| **Equity / fairness** — per-neighbourhood welfare | Global metrics only (current prototype limitation) | Per-cell or per-demographic welfare terms; Pareto-front policies |
| **Observability** — real cities are partially observed | Full-state MDP (clean lab setup) | POMDP with sensor/census lag; recurrent policy (LSTM PPO) |
| **Safe exploration** — cannot "try random zoning" IRL | Only simulator | Offline RL from historical land-use records; simulation-then-deploy |
| **Scale** — 10×10 toy vs. 1000× real cells | 804-d obs, 35 actions | Hierarchical RL (district-level manager + cell-level executor); GNN policy for spatial structure |

**What doesn't transfer (honesty slide):** real cities have politics, budgets, legal constraints, and multi-agent stakeholders — single-agent PPO is the *technical core* of a decision-support tool, not the full solution.

---

## Slide 9 — Conclusion & future work

**Takeaway**
- RL is a **natural fit** for sequential multi-objective planning under uncertainty — exactly what urban sustainability is.
- **Reward design is the research surface.** Our "do nothing → minimalist road+green" trajectory is modern RL alignment in miniature.
- Baselines lose; PPO wins (+21 vs. −1,911) **only after** reward re-spec + HP tuning — both driven by explicit diagnostics.

**Next steps (concrete)**
1. **Per-resident livability** (instead of per-cell bonus) to unlock mixed-use cities.
2. **Hierarchical PPO** — district-level planner + cell-level executor for scaling to 100×100 grids.
3. **SFT → PPO warm-start** from heuristic demonstrations (classic modern RL recipe).
4. **Offline RL** on OpenStreetMap + census land-use traces for real-city pretraining.

**Thank you · Questions**

---

## Speaker notes (not on slides)

- **Pacing (7 min):** 1→2 (1:30), 3 (0:50), 4 (0:50), 5 (0:50), 6 (1:10), 7 (1:10), 8 (0:50), 9 (0:30).
- **Demo option:** 10–15s screen capture of seed-18 rollout (reward +43.70) in 3D viewer. Only if timing.
- **Common questions:**
  - *"Why not supervised?"* → no fixed labels, actions change distribution, path-dependent.
  - *"Why PPO not DQN?"* → discrete but large action space, continuous-valued features, need stable value targets → PPO + VecNormalize is standard.
  - *"Does it actually work in a real city?"* → no. This is a prototype/digital-twin precursor. Scaling path is on Slide 8.
  - *"Why did tuning barely improve reward?"* → both +21.08 and +21.28 hit the same "minimalist" ceiling of the current reward function; the next gain comes from reward re-spec, not more tuning.

---

## Rubric checklist

| Requirement | Covered on |
|---|---|
| Applied use case in structured domain | Slides 1, 2 |
| Justify RL | Slide 2 |
| Prototype / simulation | Slides 3, 4 |
| Relevant metrics + literature | Slides 2 (UN Habitat / UrbanSim), 4 (VecNormalize, PPO-Schulman), 6 (quantitative) |
| Alignment / safety | Slides 7, 8 |
| Failure modes + training | Slide 7 |
| Real-world scaling | Slide 8 |
| Concrete results | Slide 6 |
| Future work | Slide 9 |
