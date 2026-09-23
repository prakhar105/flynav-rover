# Setup

This guide reproduces the current FlyNav Rover development environment on Windows.

## 1. Prerequisites

Install:

- Git
- Python compatible with the repository `pyproject.toml`
- uv
- Webots R2025a
- a neuPrint account/token if rebuilding the connectome-derived datasets

Useful links:

- Webots: https://cyberbotics.com/
- Webots docs: https://cyberbotics.com/doc/guide/index
- uv: https://docs.astral.sh/uv/
- neuPrint: https://neuprint.janelia.org/
- MaleCNS: https://male-cns.janelia.org/
- MaleCNS programmatic access: https://male-cns.janelia.org/download/
- neuprint-python quickstart: https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
- Brian2: https://brian2.readthedocs.io/

## 2. Clone

```powershell
git clone https://github.com/prakhar105/flynav-rover.git
cd flynav-rover
```

## 3. Install Python dependencies

The project uses `uv`.

```powershell
uv sync
```

If starting from a minimal environment and the dependency file has not yet been updated, the key scientific packages used during development are:

```powershell
uv add neuprint-python networkx pandas matplotlib python-dotenv brian2
```

Do not blindly rerun `uv add` on an already frozen lockfile unless you intentionally want to update dependencies.

## 4. neuPrint configuration

Create:

```text
.env
```

from:

```text
.env.example
```

Example:

```env
NEUPRINT_TOKEN=replace_with_your_personal_token
```

The token is obtained from your neuPrint account.

The project uses:

```python
from neuprint import Client

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)
```

Never commit `.env`.

## 5. Webots setup

Current development baseline:

```text
Webots R2025a
```

Open:

```text
worlds/flynav-rover.wbt
```

The E-puck must have:

```text
supervisor TRUE
```

for the goal-navigation/validation controllers because they read the simulated robot pose and, during validation, reset the robot position between trials.

### Goal-navigation controller

Set the E-puck controller to:

```text
flynav_goal_v1
```

Controller file:

```text
controllers/flynav_goal_v1/flynav_goal_v1.py
```

The controller loads:

```text
connectome/output/frozen_fc2_steering_matrix.csv
```

If the world contains a node:

```text
DEF GOAL
```

the controller uses that target position. Otherwise the current fallback target is `(0.30, 0.30)`.

### Automated 8-direction validation

Set the controller to:

```text
flynav_goal_validation_v1
```

Place the E-puck near the center of an empty arena before running.

The validator tests:

```text
E, NE, N, NW, W, SW, S, SE
```

and writes:

```text
connectome/output/webots_goal_validation_results.csv
```

Expected frozen result:

```text
Successful goal directions: 8/8
PASS: all 8 world-space goal directions were reached.
```

## 6. Important controller conventions

The current engineered phase convention is:

```text
L1 = +X world heading
```

The current motor-sign convention is:

```text
positive neural steering = right turn
negative neural steering = left turn
```

Do not silently change either convention in the frozen v1 baseline.

## 7. Rebuilding connectome outputs

Only rebuild the connectome analysis if intentionally reproducing or extending the research pipeline.

The scripts in `connectome/` document the analysis sequence.

The frozen goal-navigation controller should use the committed frozen matrix rather than requiring a neuPrint call at runtime.

This separation is intentional:

```text
neuPrint / Brian2 research pipeline
              ↓
      frozen steering artifact
              ↓
        Webots runtime
```

That makes Webots runs reproducible and does not require live neuPrint credentials during simulation.

## 8. Security

Ensure `.gitignore` contains at least:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.brian_debug_*
```

Never commit:

- `NEUPRINT_TOKEN`
- personal access tokens
- GitHub tokens
- local absolute paths containing secrets
