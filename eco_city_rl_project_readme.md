# 🌍 Eco-City RL Simulation (3D + Policy Optimization)

## 🚀 Project Vision

This project builds a **3D interactive eco-city simulation** where a Reinforcement Learning (RL) agent learns to design and manage a sustainable city over time.

Unlike static simulations, the agent:
- makes **sequential urban planning decisions**
- balances **competing objectives** (growth vs sustainability)
- adapts to **dynamic environmental feedback**

The environment is rendered using **Three.js (3D visualization)** for intuitive understanding of emergent city structures.

---

## 🎯 Core Objective

Train an RL agent to:

> **Construct and manage a city that maximizes sustainability, livability, and efficiency over time**

---

## 🧠 Why Reinforcement Learning?

Urban planning is:
- sequential (early zoning decisions constrain future ones)
- long-horizon (effects emerge over time)
- multi-objective (conflicting goals)

RL is required because:
- we optimize **policies**, not static predictions
- rewards are **delayed and global**
- decisions influence **future state distributions**

---

## 🏗️ Environment Design

### Grid-Based City (Back-end)
- 2D grid (e.g., 10x10 → extendable)
- Each cell has a discrete type

### 3D Visualization (Front-end)
- Built with **Three.js**
- Each grid cell maps to a 3D object:
  - Residential → buildings
  - Industrial → factories
  - Green → trees/parks
  - Roads → flat meshes

---

## 🧩 State Representation

### Grid Tensor
Shape: (N, N, K)
- One-hot encoding of zone types

### Global Features
- population
- pollution
- traffic
- energy demand/supply

Final state:
```
state = [grid_tensor, global_features]
```

---

## 🎮 Action Space

Each step:
```
action = (cell_index, zone_type)
```

Zone types:
- Residential
- Commercial
- Industrial
- Green
- Road
- Energy

To reduce complexity:
- sample subset of candidate cells per step

---

## 🔄 Transition Dynamics (Simulation Rules)

Simple but structured rules:

- Residential → +population, +energy demand
- Industrial → +jobs, +pollution
- Green → −pollution
- Roads → −traffic locally
- Energy → +supply

Global metrics updated after every action.

---

## 🏆 Reward Function (Multi-Objective)

We use a weighted reward:

R = α * livability − β * pollution − γ * traffic − δ * energy_mismatch

Where:

- livability = population satisfaction proxy
- pollution = emissions score
- traffic = congestion metric
- energy_mismatch = |demand − supply|

### Default Weights
- α = 1.0
- β = 1.2
- γ = 0.7
- δ = 0.8

---

## 🤖 Algorithm Choice (IMPORTANT)

### ❌ Why NOT Q-Learning / DQN
- Action space is **large (grid × zones)**
- State is **high-dimensional (spatial + global)**
- Poor scalability

### ✅ Best Choice: PPO (Proximal Policy Optimization)

#### Why PPO:
- handles **large state spaces**
- stable policy updates
- works well for **continuous + structured environments**
- widely used in simulation/control tasks

### Optional: GRPO (if implemented)
- improves sample efficiency
- better for constrained policies

👉 **Final Choice: PPO (baseline), GRPO (optional extension)**

---

## 🧪 Experimental Design (VERY SPECIFIC)

### Experiment 1: RL vs Baselines

Agents:
- Random policy
- Greedy (maximize population)
- Heuristic (balanced zoning)
- PPO agent

Metrics:
- total reward
- pollution
- traffic
- energy efficiency

---

### Experiment 2: Reward Sensitivity

Vary weights:

Case A: sustainability-focused
- β high (pollution penalty ↑)

Case B: growth-focused
- α high (population ↑)

Compare resulting city layouts.

---

### Experiment 3: Generalization

Train on:
- fixed demand pattern

Test on:
- higher population growth
- higher pollution penalties

Goal:
- evaluate robustness of learned policy

---

### Experiment 4: Policy Behavior Analysis

Track over time:
- zoning distribution
- infrastructure allocation
- emergent patterns

---

## 📊 Metrics

Primary:
- cumulative reward
- pollution level
- traffic congestion
- energy efficiency

Secondary:
- zoning balance
- city growth rate

---

## 🎨 3D Visualization (Three.js)

### Features
- real-time rendering of city grid
- color-coded zones
- height variation (population density)
- animation over time

### Tech Stack
- Three.js
- React (optional)
- WebSocket / JSON export from Python

---

## ⚠️ Alignment & Safety Analysis

### Reward Misspecification
- agent may overbuild one zone type

### Emergent Failures
- extreme industrial clustering
- zero-green cities

### Trade-off Instability
- optimizing one metric harms others

---

## 🛠️ Implementation Plan

### Phase 1: Core Environment
- grid + rules
- reward function

### Phase 2: Baselines
- random
- greedy
- heuristic

### Phase 3: RL Agent
- PPO implementation (Stable-Baselines3 recommended)

### Phase 4: Visualization
- export states → JSON
- render using Three.js

### Phase 5: Experiments
- run all experiment sets
- log metrics

---

## 📁 Project Structure

```
project/
│
├── env/
│   └── city_env.py
├── agents/
│   └── ppo_agent.py
├── baselines/
│   └── heuristics.py
├── training/
│   └── train.py
├── visualization/
│   └── threejs/
├── evaluation/
│   └── experiments.py
└── README.md
```

---

## 🚀 Final Outcome

You will demonstrate:

- RL learns **better planning policies than heuristics**
- policies adapt to **multi-objective trade-offs**
- emergent city structures reflect **reward design**

---

## 🏁 Getting Started

1. Build environment
2. Implement baselines
3. Train PPO agent
4. Run experiments
5. Visualize in 3D

---

## 📚 References

- Sutton & Barto — RL
- PPO (Schulman et al.)
- Urban simulation research

---

## ⭐ Key Contribution

A fully simulated **3D RL-driven eco-city planner** that demonstrates how intelligent policies emerge in complex, structured sustainability domains.

