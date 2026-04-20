# Eco-City RL Simulation — Full Build Plan

## Project Overview

A 3D interactive eco-city simulation where a PPO-based RL agent learns to design and manage a sustainable city on a grid. The backend is Python (Gymnasium + Stable-Baselines3). The frontend is a Three.js web app that reads exported JSON state history and renders the city in 3D.

---

## Directory Structure (Create Exactly This)

```
eco-city/
├── env/
│   ├── __init__.py
│   ├── city_env.py          # Gymnasium custom environment
│   └── dynamics.py          # Transition rules
├── agents/
│   ├── __init__.py
│   └── ppo_agent.py         # PPO wrapper using Stable-Baselines3
├── baselines/
│   ├── __init__.py
│   └── heuristics.py        # Random, Greedy, Heuristic agents
├── training/
│   ├── __init__.py
│   ├── train.py             # Main training loop
│   └── callbacks.py         # SB3 custom callbacks for logging
├── evaluation/
│   ├── __init__.py
│   └── experiments.py       # All 4 experiment runners
├── visualization/
│   ├── exporter.py          # Export city state history → JSON
│   └── threejs/
│       ├── index.html
│       ├── main.js          # Three.js scene setup
│       ├── cityRenderer.js  # Builds 3D meshes from grid state
│       ├── controls.js      # OrbitControls + playback UI
│       └── style.css
├── logs/                    # Auto-created at runtime
├── results/                 # Auto-created at runtime
├── requirements.txt
└── plan.md
```

---

## Step 1: Environment Setup

### `requirements.txt`

```
gymnasium>=0.29.0
stable-baselines3>=2.2.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
torch>=2.0.0
tensorboard>=2.13.0
```

Install: `pip install -r requirements.txt`

---

## Step 2: Core Environment — `env/city_env.py`

### Constants

```python
GRID_SIZE = 10          # 10x10 grid
NUM_ZONE_TYPES = 6      # 0=Empty, 1=Residential, 2=Commercial, 3=Industrial, 4=Green, 5=Road, 6=Energy
NUM_GLOBAL_FEATURES = 4 # population, pollution, traffic, energy_balance
CANDIDATE_CELLS = 5     # cells sampled per step to reduce action space
MAX_STEPS = 200
```

### Zone Type Enum

```python
class Zone(IntEnum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    GREEN = 4
    ROAD = 5
    ENERGY = 6
```

### State Space

- `grid`: shape `(GRID_SIZE, GRID_SIZE, 7)` — one-hot encoded zone types, dtype float32
- `global_features`: shape `(4,)` — [population, pollution, traffic, energy_balance], normalized to [0,1]
- Observation: `Dict` space with keys `"grid"` and `"global_features"`, or flattened to Box

Use `gymnasium.spaces.Dict` for the observation space.

### Action Space

```python
# Discrete action: index into (candidate_cell × zone_type)
# At each step, sample CANDIDATE_CELLS cells from the grid
# action = integer in [0, CANDIDATE_CELLS * NUM_ZONE_TYPES)
# Decode: cell_idx = action // NUM_ZONE_TYPES, zone = action % NUM_ZONE_TYPES
action_space = gymnasium.spaces.Discrete(CANDIDATE_CELLS * NUM_ZONE_TYPES)
```

### `CityEnv` Class (inherits `gymnasium.Env`)

**`__init__`:**
- Initialize `grid` as `np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)`
- Initialize global metrics: `population=0`, `pollution=0`, `traffic=0`, `energy_supply=0`, `energy_demand=0`
- Set reward weights: `alpha=1.0, beta=1.2, gamma=0.7, delta=0.8`
- `step_count = 0`
- `state_history = []` — list of snapshots for export

**`reset(seed=None)`:**
- Clear grid, reset all metrics to 0
- Sample initial `candidate_cells` from flat grid indices
- Return observation dict and info dict

**`step(action)`:**
- Decode action → `(cell_row, cell_col, zone_type)`
- Apply zone to grid
- Call `dynamics.update_metrics(grid, global_features)` → returns updated metrics
- Compute reward: `R = alpha*livability - beta*pollution - gamma*traffic - delta*|energy_demand - energy_supply|`
- Save snapshot to `state_history`
- Increment `step_count`; done if `step_count >= MAX_STEPS`
- Sample new `candidate_cells` for next step
- Return `obs, reward, terminated, truncated, info`

**`_get_obs()`:**
- One-hot encode grid → shape `(GRID_SIZE, GRID_SIZE, 7)`
- Normalize global features → shape `(4,)`
- Flatten both and concatenate → 1D array for SB3 compatibility
- Total size: `GRID_SIZE*GRID_SIZE*7 + 4 = 704`

**`_compute_livability()`:**
- `livability = min(population / 100.0, 1.0)` — normalized population satisfaction proxy

**`export_history(path)`:**
- Serialize `state_history` to JSON at `path`

---

## Step 3: Transition Dynamics — `env/dynamics.py`

### `update_metrics(grid, metrics)` function

Scan entire grid, count each zone type, then apply rules:

```python
def update_metrics(grid, metrics):
    counts = count_zones(grid)  # dict: zone -> count

    # Population: each residential adds 10, each commercial adds 5
    metrics['population'] = counts[Zone.RESIDENTIAL] * 10 + counts[Zone.COMMERCIAL] * 5

    # Energy demand
    metrics['energy_demand'] = counts[Zone.RESIDENTIAL] * 3 + counts[Zone.INDUSTRIAL] * 8 + counts[Zone.COMMERCIAL] * 4

    # Energy supply
    metrics['energy_supply'] = counts[Zone.ENERGY] * 15

    # Pollution: industrial creates, green removes
    metrics['pollution'] = max(0, counts[Zone.INDUSTRIAL] * 5 - counts[Zone.GREEN] * 3)

    # Traffic: reduced by roads, increased by population density
    raw_traffic = metrics['population'] * 0.1 - counts[Zone.ROAD] * 2
    metrics['traffic'] = max(0, raw_traffic)

    return metrics
```

---

## Step 4: Baseline Agents — `baselines/heuristics.py`

### Three classes, all with a `.predict(obs, env)` method

**`RandomAgent`:**
- `return env.action_space.sample()`

**`GreedyAgent`:**
- Always place Residential in the first available candidate cell
- `action = 0 * NUM_ZONE_TYPES + Zone.RESIDENTIAL`

**`HeuristicAgent`:**
- Balanced zoning policy:
  - If `pollution > 50`: place Green
  - Elif `energy_balance < 0`: place Energy
  - Elif `traffic > 30`: place Road
  - Else: place Residential
- Encodes this as an action on candidate_cells[0]

---

## Step 5: PPO Agent — `agents/ppo_agent.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def make_ppo_agent(env, tensorboard_log="logs/ppo"):
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    return model

def train(model, total_timesteps=500_000, callback=None):
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model

def save(model, path="results/ppo_eco_city"):
    model.save(path)

def load(path, env):
    return PPO.load(path, env=env)
```

---

## Step 6: Training Callbacks — `training/callbacks.py`

```python
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class MetricsCallback(BaseCallback):
    """Logs custom city metrics to tensorboard every episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.logger.record("city/population", info.get("population", 0))
                self.logger.record("city/pollution", info.get("pollution", 0))
                self.logger.record("city/traffic", info.get("traffic", 0))
                self.logger.record("city/energy_balance", info.get("energy_balance", 0))
        return True
```

Pass city metrics back through `info` in `env.step()`.

---

## Step 7: Training Script — `training/train.py`

```python
import argparse
from env.city_env import CityEnv
from agents.ppo_agent import make_ppo_agent, train, save
from training.callbacks import MetricsCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    env = CityEnv()
    model = make_ppo_agent(env)
    callback = MetricsCallback()
    train(model, total_timesteps=args.timesteps, callback=callback)
    save(model)

    if args.export:
        from visualization.exporter import export
        export(env, "visualization/threejs/city_data.json")

if __name__ == "__main__":
    main()
```

Run: `python training/train.py --timesteps 500000 --export`

---

## Step 8: Experiments — `evaluation/experiments.py`

### Structure

```python
def run_experiment_1(timesteps=200_000):
    """RL vs Baselines: compare cumulative reward, pollution, traffic, energy."""
    results = {}
    for agent_name, agent in [("random", RandomAgent()), ("greedy", GreedyAgent()), ("heuristic", HeuristicAgent())]:
        env = CityEnv()
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(200):
            action = agent.predict(obs, env)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            if done: break
        results[agent_name] = {"reward": total_reward, **info}

    # Load trained PPO
    ppo_env = CityEnv()
    model = PPO.load("results/ppo_eco_city", env=ppo_env)
    obs, _ = ppo_env.reset()
    total_reward = 0
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = ppo_env.step(action)
        total_reward += reward
        if done: break
    results["ppo"] = {"reward": total_reward, **info}
    return results

def run_experiment_2():
    """Reward sensitivity: sustainability-focused vs growth-focused."""
    # Run two CityEnv instances with different alpha/beta weights
    # sustainability: alpha=0.5, beta=2.0
    # growth: alpha=2.0, beta=0.3
    # Train PPO on each, compare final city layouts

def run_experiment_3():
    """Generalization: train on fixed demand, test on higher growth/pollution."""
    # Train on default env, test by modifying env dynamics multipliers

def run_experiment_4():
    """Policy behavior analysis: track zoning distribution over training."""
    # Log zone counts at each step, plot distribution evolution
```

---

## Step 9: State Exporter — `visualization/exporter.py`

```python
import json
import numpy as np

def export(env, output_path):
    history = []
    for snapshot in env.state_history:
        history.append({
            "step": snapshot["step"],
            "grid": snapshot["grid"].tolist(),   # 2D list of zone ints
            "population": snapshot["population"],
            "pollution": snapshot["pollution"],
            "traffic": snapshot["traffic"],
            "energy_balance": snapshot["energy_balance"],
            "reward": snapshot["reward"],
        })
    with open(output_path, "w") as f:
        json.dump(history, f)
    print(f"Exported {len(history)} steps to {output_path}")
```

Call this after a trained episode to generate `city_data.json`.

---

## Step 10: Three.js Visualization

### `visualization/threejs/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Eco-City RL Viewer</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div id="ui">
    <h2>Eco-City RL</h2>
    <div id="stats"></div>
    <div id="controls">
      <button id="play">Play</button>
      <button id="pause">Pause</button>
      <input type="range" id="step-slider" min="0" value="0" />
      <span id="step-label">Step: 0</span>
    </div>
  </div>
  <canvas id="canvas"></canvas>
  <script type="importmap">
    { "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js",
                   "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/" } }
  </script>
  <script type="module" src="main.js"></script>
</body>
</html>
```

### `visualization/threejs/style.css`

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a1a; color: #e0e0e0; font-family: monospace; overflow: hidden; }
#canvas { display: block; width: 100vw; height: 100vh; }
#ui { position: fixed; top: 20px; left: 20px; z-index: 10; background: rgba(0,0,0,0.7);
      padding: 16px; border-radius: 8px; min-width: 220px; }
#ui h2 { color: #4caf50; margin-bottom: 10px; }
#stats div { margin: 4px 0; font-size: 13px; }
#controls { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
button { background: #4caf50; color: #000; border: none; padding: 4px 10px;
         border-radius: 4px; cursor: pointer; font-family: monospace; }
button:hover { background: #81c784; }
#step-slider { width: 100%; }
```

### `visualization/threejs/main.js`

```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CityRenderer } from './cityRenderer.js';
import { setupControls } from './controls.js';

// Scene setup
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);
scene.fog = new THREE.Fog(0x0a0a1a, 50, 200);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(30, 40, 30);
camera.lookAt(0, 0, 0);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

// Lighting
const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xfff8e1, 1.2);
sun.position.set(30, 60, 20);
sun.castShadow = true;
scene.add(sun);

// Ground plane
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(100, 100),
  new THREE.MeshLambertMaterial({ color: 0x1a1a2e })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// Load city data and start
const cityRenderer = new CityRenderer(scene);

fetch('city_data.json')
  .then(r => r.json())
  .then(data => {
    setupControls(data, cityRenderer, updateStats);
    cityRenderer.render(data[0]);
    updateStats(data[0]);
    document.getElementById('step-slider').max = data.length - 1;
  });

function updateStats(snapshot) {
  document.getElementById('stats').innerHTML = `
    <div>Step: ${snapshot.step}</div>
    <div>Population: ${snapshot.population}</div>
    <div>Pollution: ${snapshot.pollution.toFixed(2)}</div>
    <div>Traffic: ${snapshot.traffic.toFixed(2)}</div>
    <div>Energy Balance: ${snapshot.energy_balance.toFixed(2)}</div>
    <div>Reward: ${snapshot.reward.toFixed(3)}</div>
  `;
}

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Render loop
function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
}
animate();
```

### `visualization/threejs/cityRenderer.js`

```javascript
import * as THREE from 'three';

const CELL_SIZE = 2;
const COLORS = {
  0: 0x222233,   // Empty — dark base
  1: 0x1565c0,   // Residential — blue
  2: 0xf57f17,   // Commercial — amber
  3: 0xb71c1c,   // Industrial — red
  4: 0x2e7d32,   // Green — forest green
  5: 0x424242,   // Road — grey
  6: 0xf9a825,   // Energy — yellow
};

const HEIGHTS = {
  0: 0.05,
  1: 2.0,   // Residential: tall
  2: 1.5,   // Commercial: medium
  3: 2.5,   // Industrial: tallest
  4: 0.3,   // Green: low
  5: 0.05,  // Road: flat
  6: 1.0,   // Energy: medium
};

export class CityRenderer {
  constructor(scene) {
    this.scene = scene;
    this.meshes = [];
  }

  render(snapshot) {
    // Clear previous meshes
    this.meshes.forEach(m => this.scene.remove(m));
    this.meshes = [];

    const grid = snapshot.grid;
    const N = grid.length;
    const offset = (N * CELL_SIZE) / 2;

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const zone = grid[row][col];
        const height = HEIGHTS[zone];
        const color = COLORS[zone];

        const geo = new THREE.BoxGeometry(CELL_SIZE * 0.9, height, CELL_SIZE * 0.9);
        const mat = new THREE.MeshLambertMaterial({ color });
        const mesh = new THREE.Mesh(geo, mat);

        mesh.position.set(
          col * CELL_SIZE - offset + CELL_SIZE / 2,
          height / 2,
          row * CELL_SIZE - offset + CELL_SIZE / 2
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        this.scene.add(mesh);
        this.meshes.push(mesh);
      }
    }
  }
}
```

### `visualization/threejs/controls.js`

```javascript
export function setupControls(data, cityRenderer, updateStats) {
  const slider = document.getElementById('step-slider');
  const playBtn = document.getElementById('play');
  const pauseBtn = document.getElementById('pause');
  const stepLabel = document.getElementById('step-label');

  let currentStep = 0;
  let playing = false;
  let interval = null;

  function goToStep(idx) {
    currentStep = Math.max(0, Math.min(idx, data.length - 1));
    slider.value = currentStep;
    stepLabel.textContent = `Step: ${currentStep}`;
    cityRenderer.render(data[currentStep]);
    updateStats(data[currentStep]);
  }

  playBtn.addEventListener('click', () => {
    if (playing) return;
    playing = true;
    interval = setInterval(() => {
      if (currentStep >= data.length - 1) {
        clearInterval(interval);
        playing = false;
        return;
      }
      goToStep(currentStep + 1);
    }, 200);  // 200ms per step
  });

  pauseBtn.addEventListener('click', () => {
    playing = false;
    clearInterval(interval);
  });

  slider.addEventListener('input', () => {
    goToStep(parseInt(slider.value));
  });
}
```

---

## Step 11: Build Order for Cursor

Execute these phases in order. Each phase must pass before moving to the next.

### Phase 1 — Environment (do this first)

1. Create all `__init__.py` files
2. Implement `env/dynamics.py` with `count_zones()` and `update_metrics()`
3. Implement `env/city_env.py` with `CityEnv(gymnasium.Env)`
4. Verify with: `python -c "from env.city_env import CityEnv; from stable_baselines3.common.env_checker import check_env; env = CityEnv(); check_env(env); print('ENV OK')"`

### Phase 2 — Baselines

5. Implement `baselines/heuristics.py` with all three agents
6. Quick smoke test: run each agent for 200 steps, print total reward

### Phase 3 — PPO Agent + Training

7. Implement `agents/ppo_agent.py`
8. Implement `training/callbacks.py`
9. Implement `training/train.py`
10. Run: `python training/train.py --timesteps 100000` (quick sanity check)
11. Full training: `python training/train.py --timesteps 500000 --export`

### Phase 4 — Experiments

12. Implement `evaluation/experiments.py` with all 4 experiment functions
13. Add a `main()` that runs all experiments and saves results to `results/experiment_results.json`
14. Run: `python evaluation/experiments.py`

### Phase 5 — Visualization

15. Implement `visualization/exporter.py`
16. Create all Three.js files exactly as specified above
17. Serve locally: `python -m http.server 8000 --directory visualization/threejs/`
18. Open `http://localhost:8000` and verify the city renders + playback works

---

## Step 12: Key Implementation Notes for Cursor

- **Observation flattening**: SB3 MlpPolicy requires a 1D Box observation. Flatten the one-hot grid and concatenate with global_features in `_get_obs()`.
- **Candidate cells**: at each `reset()` and each `step()`, call `np.random.choice(GRID_SIZE*GRID_SIZE, CANDIDATE_CELLS, replace=False)` to get candidate cell indices. Store as `self.candidate_cells`. Decode action as `candidate_cells[action // NUM_ZONE_TYPES]` → row/col, and `action % NUM_ZONE_TYPES` → zone.
- **State history snapshot**: append to `self.state_history` in `step()` with `{"step": self.step_count, "grid": self.grid.copy(), "population": ..., "pollution": ..., "traffic": ..., "energy_balance": energy_supply - energy_demand, "reward": reward}`.
- **Three.js**: use ES module imports with importmap. No npm/bundler needed — runs from a static file server.
- **Experiment reproducibility**: pass `seed=42` to `env.reset()` for all baseline comparisons.
- **Tensorboard**: run `tensorboard --logdir logs/` during training to monitor progress.

---

## Expected Outputs

| File | Description |
|------|-------------|
| `results/ppo_eco_city.zip` | Saved PPO model weights |
| `visualization/threejs/city_data.json` | Episode state history for 3D viewer |
| `results/experiment_results.json` | All 4 experiment results |
| `logs/` | Tensorboard logs |

---

## Success Criteria

- `check_env(env)` passes with no errors
- PPO agent achieves higher cumulative reward than all 3 baselines after 500k timesteps
- 3D city viewer renders the full episode and playback controls work
- Experiment 2 shows visibly different city layouts for sustainability vs growth weights
