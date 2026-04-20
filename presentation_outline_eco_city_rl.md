# Eco-City RL — 7-Minute Presentation Outline  
**Course:** RL class | **Format:** Copy each `##` block into one slide (title + bullets)

**Timing guide:** ~45–55 seconds per slide → ~9 slides ≈ 7 minutes (leave 30s buffer).

---

## Slide 1 — Title

- **Learning to Plan Sustainable Cities with Reinforcement Learning**
- *[Your name] · AIPI 590 · [Date]*
- **Applied domain:** Structured decision-making in **urban sustainability** (growth vs. pollution, energy, livability)

---

## Slide 2 — Applied use case (rubric: structured domain)

- **Problem:** Cities must make **sequential, coupled** land-use decisions (zoning, roads, green space, energy) under **competing objectives**.
- **Structured domain:** Discrete grid; zone types; transition rules from planning/engineering-style dynamics (population, emissions, traffic, energy balance).
- **Outcome we care about:** Policies that **balance** livability and environmental/load constraints—not a one-shot classifier predicting “good/bad” cities.

---

## Slide 3 — Why RL? (justification)

- **Sequential:** Early zoning **constrains** later choices (path dependence).
- **Delayed & global reward:** “Good” outcomes emerge after many steps; myopic rules miss long-horizon trade-offs.
- **Policy optimization:** We want a **decision rule** (policy), not only prediction of outcomes.
- **Baselines fall short:** Random / greedy / hand-crafted heuristics are useful **comparisons** but don’t optimize a stated objective under uncertainty.

*Contrast (one line):* supervised learning fits static input→label maps; **RL fits control** when the agent’s actions **change** the data distribution over time.

---

## Slide 4 — MDP formulation (prototype grounding)

| Component | Eco-City prototype |
|-----------|-------------------|
| **State** | Grid (zone IDs) + global features (population, pollution, traffic, energy imbalance)—flattened for PPO. |
| **Action** | Choose **cell** (from sampled candidates) + **zone type** (residential, industrial, green, road, energy, …). |
| **Dynamics** | Rule-based updates: e.g. industry ↑ pollution; green mitigates; roads affect traffic pressure; energy assets ↑ supply. |
| **Reward** | Multi-objective scalar: weighted **livability** minus penalties on pollution, traffic, energy mismatch (weights α, β, γ, δ). |
| **Horizon** | Fixed episode length (e.g. 200 steps). |

*One sentence:* This is a **simplified MDP** for research and visualization, not a production digital twin.

---

## Slide 5 — Prototype & implementation

- **Simulation:** Grid world + metrics; **3D viewer** (e.g. Three.js) to **inspect** emergent layouts.
- **Algorithm:** **PPO** (Stable-Baselines3)—scalable for continuous-ish observations and large discrete action spaces vs. tabular Q methods.
- **Training aids:** **VecNormalize** (normalize obs/rewards) to align value targets with return scale; baselines: random, greedy, heuristic.
- **Metrics:** Episode return; final pollution, traffic, population; zone distribution; (optional) export episode trajectory for the viewer.

---

## Slide 6 — Metrics & literature (rubric)

- **Metrics:** Cumulative reward; pollution; congestion proxy; energy balance; zoning balance / growth.
- **Core RL:** Sutton & Barto — MDPs, policy optimization, value bootstrapping.
- **PPO:** Schulman et al. — stable policy-gradient updates for continuous control and many simulators.
- **Urban / simulation context (optional cite):** Urban analytics and land-use simulation literature (e.g. zoning and ABM-style models)—our prototype is **inspired by** that structure, **simplified** for RL.

---

## Slide 7 — Alignment & safety (rubric)

- **Reward misspecification:** Optimizing the scalar can yield **undesirable** equilibria (e.g. industrial clustering, no green space) if weights don’t match stakeholder values.
- **Proxy risk:** “Livability” as a function of population/metrics is a **proxy**—not full welfare or equity across neighborhoods.
- **Distribution shift:** Policies trained in one growth/emission regime may **fail** when dynamics or objectives change.
- **Mitigations (conceptual):** Constrained RL, reward **modeling** from human feedback, **multi-objective** Pareto policies, **audits** on worst-case scenarios.

---

## Slide 8 — Failure modes & training challenges (your story)

- **Learning signal:** Very negative raw returns → value network **hard to fit**; **explained variance** near zero early; **VecNormalize** helps align scales.
- **PPO diagnostics:** **clip_fraction ≈ 0** can mean updates are **tiny** (policy near initial / not yet pushing ratios)—often a **symptom** of slow learning, not necessarily wrong `clip_range`.
- **Degenerate policies:** Short training or bad scaling can collapse to **repetitive actions** (e.g. always same zone type)—shows need for **longer training**, normalization, or architecture tuning.
- **MDP gap:** Grid + hand rules **omit** geography, roads network, equity, and social dynamics—**failure modes** in deployment would be structural, not only algorithmic.

---

## Slide 9 — Future work & conclusion

- **Future work:** Richer dynamics (network traffic, budget constraints), **multi-agent** stakeholders, **offline RL** from real urban data, **safe exploration** under constraints.
- **Takeaway:** RL is a **natural fit** for sequential multi-objective planning; our **prototype** makes trade-offs **visible** and forces **explicit** reward design—where **alignment** work actually lives.
- **Thank you / Questions**

---

## Speaker notes (optional — not on slides)

- **7 min pacing:** Slides 1–2 (~1:00), 3–4 (~1:30), 5–6 (~1:30), 7 (~1:00), 8 (~1:30), 9 (~0:30).
- **Demo (if allowed):** 15–20s screen record of **3D playback** or TensorBoard curve—**only if** it doesn’t break the clock.
- **If asked “why not supervised?”:** No fixed dataset of “optimal” cities for every state; objectives are **policy-dependent** and **path-dependent**.

---

## Rubric checklist (for you)

| Requirement | Where covered |
|-------------|----------------|
| Applied structured domain | Slides 2, 4 |
| Justify RL | Slide 3 |
| Prototype / simulation | Slides 4–5 |
| Alignment / safety | Slide 7 |
| Metrics + literature | Slide 6 |
| Failure modes + training + MDP | Slides 4, 8 |
| Future work | Slide 9 |
