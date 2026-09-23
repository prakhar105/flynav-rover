# Frozen Baseline

## Baseline name

**FlyNav Goal Navigation v1**

Recommended Git tag:

```text
goal-nav-v1
```

Recommended release title:

```text
FlyNav Goal Navigation v1 — 8/8 Webots Validation
```

## Freeze status

Goal-navigation v1 is frozen after successful Webots validation.

Do not change the frozen neural map or its scientific calibration during the first combined-navigation implementation.

## Frozen scientific/engineering baseline

### Selected circuit

- 220 neurons
- 3,231 selected functional edges
- populations:
  - EPG
  - Delta7
  - FC2A / FC2B / FC2C
  - PFL2
  - PFL3
  - DNa03
  - DNa02

### Frozen surrogate parameters

```text
EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN    = 3.0
SPIKE_SCALE        = 5.0
ACTIVITY_SCALE     = 1.10599
```

Delta7 → PFL inhibition remains a **model hypothesis**, not a claim that neuPrint directly specifies the effective sign in this surrogate.

### Frozen discrete goal map

The accepted runtime representation is the discrete:

```text
16 EPG heading states × 9 FC2 goal populations
```

stored in:

```text
connectome/output/frozen_fc2_steering_matrix.csv
```

Continuous FC2 interpolation experiments are not part of the frozen runtime baseline.

## Why continuous interpolation was not frozen

The research sequence tested amplitude blending, temporal multiplexing, population recruitment, alternative DNa temporal readouts, PFL/DNa shadow dynamics, PFL3 threshold sweeps and a graded PFL3 relay.

The useful conclusion was:

- pure discrete states are reproducible and usable;
- interpolation through the current LIF surrogate introduced threshold/event discontinuities;
- lowering the PFL3 threshold damaged endpoint preservation;
- the graded PFL3 relay improved smoothness but did not preserve downstream motor behavior sufficiently.

Therefore the v1 rover uses the discrete neural action map and performs smoothing at the robotics-control layer.

## Step 39 action-map audit

Result:

```text
Heading states: 16
FC2 goal states per heading: 9

Headings with both negative and positive steering: 14/16
Headings with near-neutral action: 16/16
```

The two incomplete heading rows are handled at runtime with a generic bounded one-bin neighboring-heading fallback.

This fallback is an engineering interface, not a biological claim.

## Webots validation

Automated validation tested:

```text
E
NE
N
NW
W
SW
S
SE
```

Frozen result:

```text
Successful goal directions: 8/8
Total neighbour-fallback activations: 108

PASS: all 8 world-space goal directions were reached.
```

The fallback was therefore exercised during real simulated navigation, not merely present as dead code.

## Frozen controllers

### Goal navigation

```text
controllers/flynav_goal_v1/flynav_goal_v1.py
```

### Goal validation

```text
controllers/flynav_goal_validation_v1/flynav_goal_validation_v1.py
```

### Obstacle avoidance

```text
controllers/flynav_neural_v1/
```

This remains separately frozen and will be merged at the behavior-arbitration layer.

## Do not change before Combined Navigation v1

Unless a critical regression is found, do not modify:

- the 220-neuron selected topology;
- the 3,231-edge functional topology;
- valid low-weight MaleCNS edges merely because they are small;
- the frozen EPG × FC2 steering matrix;
- the 3× EPG→Delta7 surrogate gain;
- the 3× PFL→DNa surrogate gain;
- the frozen DNa decoder calibration;
- the goal-controller neural action values;
- the frozen obstacle controller.

## Allowed next-layer work

The next implementation may add:

- goal/obstacle arbitration;
- motor-command blending;
- hysteresis;
- motor smoothing;
- runtime logging;
- combined Webots validation.

These belong to the robotics-control layer and should not require retuning the frozen neural baselines.

## Scientific wording for README/presentations

Use:

> “A Webots rover controlled by a connectome-derived Drosophila navigation circuit using an engineered Brian2 surrogate.”

Avoid:

> “We simulated a complete fly brain.”

or:

> “The rover is directly controlled by the biological fly brain.”
