# FlyNav Rover

A simulation-only robotics proof of concept that uses **connectome-derived Drosophila navigation circuitry** to steer a Webots E-puck rover toward a target.

> Scientific scope: this project uses MaleCNS neuron identities, topology and relative anatomical connectivity from neuPrint, combined with an engineered Brian2 LIF surrogate and robotics control interfaces. It is **not** a literal simulation of a fly brain.

## Current frozen milestone

**Goal Navigation v1 — FROZEN**

The current baseline:

- uses a 220-neuron selected MaleCNS navigation subgraph;
- preserves 3,231 selected functional edges in the frozen surrogate topology;
- uses EPG, Delta7, FC2, PFL2, PFL3, DNa03 and DNa02 populations;
- uses a frozen 16 × 9 EPG × FC2 steering map;
- drives a Webots E-puck through an engineered world-to-neural-state interface;
- passed automated Webots goal validation in **8/8 world-space directions**;
- keeps the earlier obstacle-avoidance controller frozen separately for the next combined-control milestone.

## What is biological vs engineered

### Connectome-derived
- MaleCNS neuron IDs and neuron types
- selected network topology
- raw anatomical edge weights
- relative synaptic-weight scaling
- left/right motor pathway structure

### Engineering approximations
- leaky integrate-and-fire dynamics in Brian2
- inhibitory interpretation of Delta7 → PFL in the surrogate
- EPG world-heading phase convention
- FC2 goal-state/action interface
- motor decoder and wheel-speed mapping
- motor smoothing and heading-row fallback

## Project structure

```text
flynav-rover/
├─ connectome/
│  ├─ 01_find_navigation_neurons.py
│  ├─ ...
│  ├─ 39_validate_discrete_action_map.py
│  └─ output/
│     ├─ frozen_fc2_steering_matrix.csv
│     ├─ frozen_fc2_population_map.csv
│     └─ webots_goal_validation_results.csv
├─ controllers/
│  ├─ baseline_controller/
│  ├─ flynav_neural_v1/              # frozen obstacle-avoidance baseline
│  ├─ flynav_goal_v1/                # frozen goal-navigation controller
│  └─ flynav_goal_validation_v1/     # automated 8-direction validation
├─ worlds/
│  └─ flynav-rover.wbt
├─ src/
│  └─ flynav_brain.py
├─ .env                              # local only, never commit
├─ .env.example
├─ README.md
├─ SETUP.md
└─ FROZEN_BASELINE.md
```

## Key frozen controllers

### Obstacle avoidance
`controllers/flynav_neural_v1/`

This is the previously frozen connectome-derived obstacle-avoidance controller. Do not modify it while building the combined controller.

### Goal navigation
`controllers/flynav_goal_v1/flynav_goal_v1.py`

Uses the frozen steering matrix:

```text
connectome/output/frozen_fc2_steering_matrix.csv
```

### Goal validation
`controllers/flynav_goal_validation_v1/flynav_goal_validation_v1.py`

Automatically tests E, NE, N, NW, W, SW, S and SE goal positions from a fixed start pose.

Latest frozen result:

```text
Successful goal directions: 8/8
PASS: all 8 world-space goal directions were reached.
```

## External resources

- Male CNS Connectome: https://male-cns.janelia.org/
- MaleCNS data/programmatic access: https://male-cns.janelia.org/download/
- neuPrint: https://neuprint.janelia.org/
- neuprint-python quickstart: https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
- neuprint-python GitHub: https://github.com/connectome-neuprint/neuprint-python
- Webots: https://cyberbotics.com/
- Webots documentation: https://cyberbotics.com/doc/guide/index
- Brian2: https://brian2.readthedocs.io/
- uv: https://docs.astral.sh/uv/

## Dataset

The project currently connects to:

```python
Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)
```

The neuPrint token must be supplied locally through `.env`. Never commit the token.

## Next milestone

**Combined Navigation v1**

Merge the two already-frozen behaviors at the robotics-control layer:

```text
goal navigation
      +
obstacle avoidance
      ↓
behaviour arbitration
      ↓
E-puck wheel commands
```

The frozen neural maps/controllers should not be retuned during the first combined-controller implementation.

## License / data note

MaleCNS data is provided by the Janelia FlyEM ecosystem. Check the source project pages for current dataset licensing and citation requirements before publication or redistribution.
