### This is the fork from original tbp.monty that uses [hybrid RL library](https://github.com/h555yand/hybrid_rl/tree/monty) as an external module to focus on benefits of replacing teleportation with realistic navigation using reinforcement learning. 

## Introduction and current results
### Background

The Thousand Brains Project's **Monty** system recognizes objects by accumulating evidence for hypotheses (object identity × pose) as a sensor agent explores object surfaces. The standard approach uses **teleportation** (`JumpToGoalState`) to instantly move the agent to target points selected by the Goal State Generator (GSG). We replaced teleportation with our RL surface navigation system, preserving Monty's recognition pipeline while adding biologically plausible movement.


### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MONTY RECOGNITION LOOP                     │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌────────┐ │
│  │ Evidence  │──▶│   GSG    │──▶│ RL Policy  │──▶│Sensors │ │
│  │ GraphLM   │   │ Goal Gen │   │ Selector   │   │(patch) │──┘
│  │ (evidence │◀──│(discrim. │   │            │   └────────┘
│  │  update)  │   │ points)  │   │            │
│  └──────────┘   └──────────┘   └─────┬──────┘
│       ▲                              │
│       │                              ▼
│       │                     ┌────────────────┐
│       │                     │  RLGoalPolicy  │
│       │                     │  ┌───────────┐ │
│       │                     │  │RL Surface │ │
│       │                     │  │Controller │ │
│       │                     │  │(Q+SAC)    │ │
│       │                     │  └───────────┘ │
│       │                     │  ┌───────────┐ │
│       └─────────────────────│  │ MuJoCo    │ │
│        intermediate obs     │  │ Adapter   │ │
│                             │  └───────────┘ │
│                             └────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### Integration Components

| Component | Role |
|-----------|------|
| **RLPolicySelector** | Routes GSG goals to RL navigation, SM goals to LookAt, no goals to default crawl. Drop-in replacement for DistantPolicySelector |
| **RLGoalPolicy** | Navigates agent along object surface to GSG target using Q+SAC hybrid controller |
| **MuJoCoEnvAdapter** | Bridges RL controller with shared MuJoCo simulator via ray-cast surface snapping |

### Challenge: Goal Suppression During Navigation

During RL navigation, Monty's GSG runs on every intermediate observation and may generate new goals. These goals are ignored by the motor system (navigation is in progress), but the GSG logs them as "attempted" and marks them as "not achieved" — creating phantom failures in metrics.

**Solution**: A `navigation_active` flag suppresses goal generation during RL navigation while allowing evidence updates to continue normally.
Changes here: src/tbp/monty/frameworks/models/goal_generation.py

**Impact**: Eliminated phantom goals (27 → 16 total goals), fixed false "not achieved" entries, resolved a timeout failure caused by wasted matching steps.

### Directed Exploration: Learning While Moving

With teleportation, the path to a goal is instant. With RL navigation, it takes 20–80 steps. We turn this "dead time" into productive exploration by sending intermediate observations to the Learning Module during navigation.

**Adaptive observation triggers:**
- Surface normal changed >30° → observe (new face detected)
- Principal curvature changed significantly → observe
- Distance-adaptive interval: every 10 steps far from goal, every 2 steps near goal
- Quality filters: skip off-object, bad depth, or stuck observations

**Why this works:**
- Each observation is a full evidence update — the LM doesn't distinguish intermediate from target observations
- Early recognition is possible: if intermediate evidence is sufficient, the episode ends before reaching the goal
- Failed navigations still contribute: the agent observes at its current position instead of silently failing

### Results: RL Navigation vs Teleportation

**Benchmark**: 2 objects × 3 rotations × 3 epochs = 6 episodes

| Configuration | Accuracy | Matching Steps | Total Steps | Goals/ep | Evidence |
|--------------|:--------:|:--------------:|:-----------:|:--------:|:--------:|
| Baseline (teleport) | 6/6 (100%) | 26.0 ± 4.3 | 101 ± 18 | 3.5 | 23.7 ± 4.2 |
| **RL + directed exploration** | **6/6 (100%)** | **31.7 ± 4.3** | **198 ± 50** | **2.7** | **26.6 ± 7.3** |

### Key Findings

1. **Accuracy parity**: RL matches baseline teleportation at 100% on all test cases.

2. **Higher evidence accumulation**: +12% more evidence (26.6 vs 23.7) thanks to intermediate observations — the agent learns while moving.

3. **Fewer goals needed**: −23% hypothesis-testing goals per episode (2.7 vs 3.5) because directed exploration provides additional evidence that accelerates convergence.

4. **Navigation cost**: ~2× more total steps — the expected cost of replacing instant teleportation with realistic navigation. This is the price of biological plausibility.

5. **Matching steps comparable**: Only +5.7 additional matching steps, and these are productive — they contribute to evidence accumulation.

---
## How to setup and run experiment


src/tbp/monty/frameworks/experiments/object_recognition_experiments.py

### Create environment
I've changed environment.yml
- set python=3.9 for mujoco==3.3.3
- comment aihabitat as a heavy module that I don't use
- add dependensies for hybrid_rl

conda env create -f environment.yml --solver=libmamba
conda activate tbp.monty.hybrid_rl
pip install numpy==1.23.5 


## Choose your favorite YCB MuJoCo objects
I describe my experience for 2 mujoco objects: cracker_box, master_chef_can. You can use it as an example.
YCB MuJoCo objects are placed here ~/tbp/data/mujoco/objects/ycb
I used these files: 
src/tbp/monty/conf/env_interface/tutorial_train_2obj_predefined.yaml
src/tbp/monty/conf/experiment/tutorial/surf_agent_2obj_train_mujoco.yaml
src/tbp/monty/conf/experiment/tutorial/surf_agent_2obj_eval_mujoco.yaml

### Train Monty model
python run.py experiment=tutorial/surf_agent_2obj_train_mujoco

### Run benchmark_rl vs teleportation baseline experiment
MODES = {
    "baseline": {"description": "Standard Monty with JumpToGoal"},
    "rl_de5":   {"description": "RL Navigation + directed exploration every 5 steps",
                 "observe_every_n_steps": 5},
}

python src/tbp/monty/integrations/hybrid_rl/benchmark_rl_vs_baseline.py

Results will be 
- ~tbp/results/monty/projects/rl_benchmark/20261014_170525/comparison.csv
20261014_170525 - dynamic folder with date and time of experiment
Сomparison of baseline vs rl_de5 similar to described above
- ~tbp/results/monty/projects/monty_integration_logs
Detailed logs with pictures of RL navigations

### Known issue: MuJoCo qhull intermittent error
You may see "qhull internal error (qh_maxsimplex)" when loading certain YCB meshes (e.g. master_chef_can). This is a non-deterministic MuJoCo/qhull issue related to process memory state.
Fix: close and reopen your terminal (or VSCode), then re-run.
The error typically resolves on the next attempt.

---

## 15. Future Directions

### 15.1 Coverage-Driven Training with RL Navigation

**Problem**: Current object model building (in systems like Monty) uses random surface crawling — 14,000 steps per object with uneven coverage.

**Proposed**: Replace random crawling with goal-directed exploration using the same RL navigation system. A `CoverageGoalGenerator` analyzes the growing object model and directs the agent to unexplored areas.

Three-level goal generation:
- **Level 1** (< 10 points): Direction-based — move along surface to build initial cluster
- **Level 2** (10–200 points): Frontier-based — push beyond the boundary of explored area
- **Level 3** (> 200 points): Gap-based — voxelize explored space, target largest uncovered regions

**Expected benefit**: ~300 steps per object (vs 14,000) with uniform coverage.

### 15.2 Model-Based Navigation Using Learned Object Models

Once a recognition system has a confident hypothesis about object identity, use the learned object model as a **world model** for navigation planning:

- Graph-based path planning along known surface topology
- Waypoint navigation with predicted edge transitions
- Confidence-gated: fall back to model-free when uncertain

### 15.3 Online Model Enrichment

During deployment, add high-quality observations to the object model in real-time. Each recognition episode makes future recognition faster — a self-improving system.

### 15.4 Real Robot Deployment

The architecture is designed for this transition:

```
Day 1:   Load trimesh-trained policy → basic navigation works (~80%+ at L0)
Days 1-N: Online Q-learning and SAC adapt to real sensor noise and physics
When needed: Offline retrain in fast trimesh simulation
Convergence: Arbitrator learns real-world source reliability
```

The `RobotEnvAdapter` implements the same `RLEnvironment` protocol:
- `get_pose()` from robot kinematics / SLAM
- `get_sensor_data()` from depth camera via point cloud processing
- `step_continuous()` maps to motor commands via inverse kinematics



