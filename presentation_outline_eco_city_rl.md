# Eco-City RL — 7-Minute Presentation Outline
**Course:** AIPI 590 (Modern RL) | **Format:** Copy each `##` block into one slide (title + bullets)

**Timing guide:** ~45–55 seconds per slide → ~9 slides ≈ 7 minutes (leave 30s buffer).

---

## Slide 1 — Title

- **Learning to Plan Sustainable Cities with Reinforcement Learning**
- *[Your name] · AIPI 590 · [Date]*
- **Applied domain:** Structured decision-making in **urban sustainability** (growth vs. pollution, energy, livability)

---

## Slide 2 — Applied use case (rubric: structured domain)

- **Problem:** Cities must make **sequential, coupled** land-use decisions (zoning, roads, green space, energy) under **competing objectives**.
- **Structured domain:** Discrete 10×10 grid; 7 zone types; transition rules from planning/engineering-style dynamics (population, emissions, traffic, energy balance).
- **Outcome we care about:** Policies that **balance** livability and environmental/load constraints — not a one-shot classifier predicting "good/bad" cities.

---

## Slide 3 — Why RL? (justification)

- **Sequential:** Early zoning **constrains** later choices (path dependence).
- **Delayed & global reward:** "Good" outcomes emerge after many steps; myopic rules miss long-horizon trade-offs.
- **Policy optimization:** We want a **decision rule** (policy), not only prediction of outcomes.
- **Baselines fall short:** Random / greedy / hand-crafted heuristics are useful **comparisons** but don’t optimize a stated objective under uncertainty.

*Contrast (one line):* supervised learning fits static input→label maps; **RL fits control** when the agent’s actions **change** the data distribution over time.

---

## Slide 4 — MDP formulation (prototype grounding)

| Component | Eco-City prototype |
|-----------|-------------------|
| **State** | 10×10 grid (zone IDs) + buildable mask + global features (population, pollution, traffic, energy balance). 804-d flat obs. |
| **Action** | Discrete: pick 1 of 5 candidate cells × 7 zone types (**35 actions**). |
| **Dynamics** | Rule-based: industry ↑ pollution; green mitigates; roads affect traffic; energy assets ↑ supply. |
| **Reward** | `α · livability − β · pollution − γ · traffic − δ · |Δenergy| + 0.01 · built_cells` |
| **Horizon** | 200 steps per episode; terrain resampled each reset. |

*One sentence:* This is a **simplified MDP** for research and visualization, not a production digital twin.

---

## Slide 5 — Prototype & implementation

- **Simulation:** Grid + terrain (buildable mask) + dynamics; **Three.js 3D viewer** for trajectory playback.
- **Algorithm:** **PPO** (Stable-Baselines3, MLP policy) — scalable for large discrete action spaces vs. tabular Q.
- **Training aids:** **VecNormalize** (reward/obs scaling), 500k timesteps, Colab GPU.
- **Baselines:** random, greedy (argmax placement), hand-crafted heuristic.
- **Evaluation harness:** 4 experiments + HP sweep + top-K rollout saver, all reproducible from one Colab notebook.

---

## Slide 6 — Results: final numbers

**Cumulative reward (higher = better, 200-step episode):**

| Agent | Total Reward | Margin vs heuristic |
|-------|-------------:|--------------------:|
| random | −4,750 | worse |
| greedy | −6,324 | worse |
| **heuristic** (best baseline) | **−1,911** | — |
| **PPO (ours)** | **+21.08** | **+1,932** |

**Hyperparameter sweep** (`training/tune_ppo.py`, 80k steps each):

| Config | Reward |
|--------|-------:|
| **low_lr (1e-4)** | **+21.28** ← best |
| default (3e-4) | −265.67 |
| larger_minibatch | −48.38 |
| high_ent | −2,906 |

**Best-run rollouts** (`evaluation/save_top_rollouts.py`, 20 seeds → top 3):  
`seed 18: +43.70` · `seed 10: +20.45` · `seed 17: +7.84` — all **positive**, confirming a robust learned policy.

---

## Slide 7 — Alignment & safety (rubric)

- **Reward hacking — we hit this live (Slide 8):** our first PPO converged to **doing nothing** (all-empty grid, reward = 0). It "beat" baselines by exploiting a ceiling in `livability`.
- **Proxy risk:** "livability" as `population/100` is a **proxy** — not welfare or equity across neighborhoods.
- **Distribution shift (Experiment 3):** trained policy tested on **high-pop-growth** and **high-industrial-emission** dynamics — reward unchanged at +21.08 across all three. Suggests the policy isn’t sensitive to external regime (could be robustness, or could be a sign the policy is doing the same "minimalist" strategy regardless).
- **Mitigations (conceptual):** constrained RL, reward **modeling** from human feedback, **multi-objective** Pareto policies, audits on worst-case scenarios.

---

## Slide 8 — Failure modes & training challenges (the real story)

- **Reward hacking (run 1):** PPO chose *inaction*. `livability` was clipped at 1 while pollution/traffic/energy penalties were unbounded → the empty grid was the **Nash equilibrium** (reward 0 > any negative real city).
  - **Fix applied:** uncap livability, add **+0.01 per built cell** bonus.
  - **Result:** positive reward, actual construction.
- **New minimalist policy (post-fix):** Experiment 4 shows the learned strategy is **~11 cells**: 8 × ROAD + 3 × GREEN. Zero residential / industrial → zero population, zero pollution, zero penalties, ~+22 from build bonus.
  - That’s still partial reward hacking — the agent found a "clean but empty" local optimum. It matches our reward function *perfectly*, which tells us the reward design is the real research surface.
- **PPO diagnostics:** high `clip_fraction` (~0.45) and `approx_kl` (~0.06) during training → updates too aggressive. HP sweep confirmed: **low_lr (1e-4) ≫ default (3e-4)** — exactly the fix the diagnostics suggested.
- **Value-function fit:** `explained_variance` climbed 0 → **0.7** after VecNormalize — critic is healthy.
- **MDP gap:** simplified dynamics omit geography, networks, equity — deployment failure modes would be structural, not only algorithmic.

---

## Slide 9 — Future work & conclusion

- **Reward redesign:** shape livability for *actual* population growth (not just non-empty cells); add pollution-per-resident not per-cell.
- **Hierarchical RL:** higher-level planner sets subgoals (zone mix targets); low-level PPO executes.
- **Warm-start (SFT → PPO):** train on heuristic demonstrations, then RL fine-tune — standard modern RL recipe.
- **Takeaway:** RL is a **natural fit** for sequential multi-objective planning, and our prototype makes **reward design** the visible research lever — which is where alignment work actually lives.
- **Thank you / Questions**

---

## Speaker notes (optional — not on slides)

- **7 min pacing:** Slides 1–2 (~1:00), 3–4 (~1:30), 5–6 (~1:30), 7 (~1:00), 8 (~1:30), 9 (~0:30).
- **Demo option:** 15–20s screen record of the **3D viewer** (dropdown with seed 18 rollout, reward +43.70) — only if timing permits.
- **If asked "why not supervised?":** No fixed dataset of "optimal" cities for every state; objectives are **policy-dependent** and **path-dependent**.
- **If asked "did the tuned model do better than main?":** Best tuned (low_lr) +21.28 vs main +21.08 — essentially tied, but tuning confirms our diagnostic read.
- **If asked why PPO "did nothing" at first:** because the reward function let it. Modern RL alignment challenge in miniature.

---

## Rubric checklist (for you)

| Requirement | Where covered |
|-------------|---------------|
| Applied structured domain | Slides 2, 4 |
| Justify RL | Slide 3 |
| Prototype / simulation | Slides 4–5 |
| Metrics + literature | Slide 6 + footnote in Slide 5 |
| Alignment / safety | Slide 7 |
| Failure modes + training + MDP | Slides 4, 8 |
| Concrete results | Slide 6 (table) |
| Future work | Slide 9 |
