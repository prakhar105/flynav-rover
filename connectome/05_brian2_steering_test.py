from pathlib import Path
from brian2 import prefs

prefs.codegen.target = "numpy"

import numpy as np
import pandas as pd

from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    defaultclock,
    ms,
    run,
    start_scope,
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

NEURON_FILE = OUTPUT_DIR / "steering_core_neurons.csv"
EDGE_FILE = OUTPUT_DIR / "steering_core_edges.csv"

RESULT_FILE = OUTPUT_DIR / "brian2_steering_test.csv"


# =========================================================
# Load real MaleCNS data
# =========================================================

neurons = pd.read_csv(NEURON_FILE)
edges = pd.read_csv(EDGE_FILE)

neurons["bodyId"] = neurons["bodyId"].astype("int64")

edges["source_bodyId"] = (
    edges["source_bodyId"]
    .astype("int64")
)

edges["target_bodyId"] = (
    edges["target_bodyId"]
    .astype("int64")
)

edges["weight"] = pd.to_numeric(
    edges["weight"],
    errors="coerce",
).fillna(0.0)


# =========================================================
# Keep downstream steering circuit
#
# PFL3 -> DNa03
# PFL3 -> DNa02
# DNa03 -> DNa02
#
# PFL2 will be added in the next experiment.
# =========================================================

keep_types = [
    "PFL3",
    "DNa03",
    "DNa02",
]

core_neurons = neurons[
    neurons["type"].isin(keep_types)
].copy()

core_ids = set(
    core_neurons["bodyId"]
)

core_edges = edges[
    edges["source_bodyId"].isin(core_ids)
    &
    edges["target_bodyId"].isin(core_ids)
].copy()


# =========================================================
# Keep only biologically relevant feed-forward routes
# =========================================================

valid_path = (
    (
        (core_edges["source_type"] == "PFL3")
        &
        (
            core_edges["target_type"]
            .isin(["DNa03", "DNa02"])
        )
    )
    |
    (
        (core_edges["source_type"] == "DNa03")
        &
        (core_edges["target_type"] == "DNa02")
    )
)

core_edges = core_edges[
    valid_path
].copy()


print("=" * 80)
print("BRIAN2 STEERING CORE")
print("=" * 80)

print(
    f"Neurons: {len(core_neurons)}"
)

print(
    f"Synaptic edges: {len(core_edges)}"
)


# =========================================================
# Metadata
# =========================================================

side_map = dict(
    zip(
        neurons["bodyId"],
        neurons["somaSide"],
    )
)


# =========================================================
# Determine which motor hemisphere each PFL3 neuron drives
#
# IMPORTANT:
# We do NOT use PFL3 soma side here.
#
# We classify a PFL3 neuron according to the biological
# descending side that its actual synapses target.
# =========================================================

pfl3_edges = core_edges[
    core_edges["source_type"] == "PFL3"
].copy()

pfl3_edges["target_side"] = (
    pfl3_edges["target_bodyId"]
    .map(side_map)
)

projection_strength = (
    pfl3_edges[
        pfl3_edges["target_side"]
        .isin(["L", "R"])
    ]
    .groupby(
        [
            "source_bodyId",
            "target_side",
        ]
    )
    ["weight"]
    .sum()
    .unstack(
        fill_value=0
    )
)

if "L" not in projection_strength.columns:
    projection_strength["L"] = 0.0

if "R" not in projection_strength.columns:
    projection_strength["R"] = 0.0


pfl3_left_turn = []

pfl3_right_turn = []


for body_id, row in projection_strength.iterrows():

    left_strength = float(
        row["L"]
    )

    right_strength = float(
        row["R"]
    )

    if right_strength > left_strength:
        pfl3_right_turn.append(
            int(body_id)
        )

    elif left_strength > right_strength:
        pfl3_left_turn.append(
            int(body_id)
        )


print()
print(
    "PFL3 neurons assigned to LEFT-turn channel:",
    len(pfl3_left_turn),
)

print(
    "PFL3 neurons assigned to RIGHT-turn channel:",
    len(pfl3_right_turn),
)


# =========================================================
# Brian2 simulation
# =========================================================

def run_condition(
    condition_name,
    stimulated_body_ids,
):

    start_scope()

    defaultclock.dt = 1 * ms

    sim_neurons = (
        core_neurons
        .reset_index(drop=True)
        .copy()
    )

    body_to_index = {
        int(body_id): index
        for index, body_id
        in enumerate(
            sim_neurons["bodyId"]
        )
    }

    N = len(sim_neurons)

    # -----------------------------------------------------
    # Simple LIF surrogate
    #
    # This is NOT claiming to reproduce exact fly
    # electrophysiology.
    #
    # The connectome structure is biological.
    # The neuron dynamics are our simulation model.
    # -----------------------------------------------------

    equations = """
    dv/dt = (-v + drive) / (20*ms) : 1
    drive : 1
    """

    G = NeuronGroup(
        N,
        equations,
        threshold="v > 1.0",
        reset="v = 0.0",
        refractory=3 * ms,
        method="euler",
    )

    G.v = 0.0
    G.drive = 0.05


    # -----------------------------------------------------
    # Stimulate selected PFL3 population
    # -----------------------------------------------------

    stimulated_indices = [
        body_to_index[body_id]
        for body_id
        in stimulated_body_ids
        if body_id in body_to_index
    ]

    G.drive[
        stimulated_indices
    ] = 1.30


    # -----------------------------------------------------
    # Actual connectome edges
    # -----------------------------------------------------

    S = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )

    source_indices = []
    target_indices = []
    raw_weights = []

    for _, row in core_edges.iterrows():

        source = int(
            row["source_bodyId"]
        )

        target = int(
            row["target_bodyId"]
        )

        if (
            source not in body_to_index
            or target not in body_to_index
        ):
            continue

        source_indices.append(
            body_to_index[source]
        )

        target_indices.append(
            body_to_index[target]
        )

        raw_weights.append(
            float(row["weight"])
        )


    S.connect(
        i=source_indices,
        j=target_indices,
    )


    # -----------------------------------------------------
    # Convert anatomical synapse counts into simulation
    # weights.
    #
    # This scaling is an engineering modelling choice.
    # Relative biological strengths are preserved.
    # -----------------------------------------------------

    raw_weights = np.asarray(
        raw_weights,
        dtype=float,
    )

    max_raw = max(
        raw_weights.max(),
        1.0,
    )

    scaled_weights = (
        0.05
        +
        0.45
        * np.sqrt(
            raw_weights
            / max_raw
        )
    )

    S.w = scaled_weights


    # -----------------------------------------------------
    # Record spikes
    # -----------------------------------------------------

    spike_monitor = SpikeMonitor(
        G
    )

    run(
        500 * ms
    )


    # -----------------------------------------------------
    # DNa02 output
    # -----------------------------------------------------

    dna02_left_indices = []

    dna02_right_indices = []


    for idx, row in sim_neurons.iterrows():

        if row["type"] != "DNa02":
            continue

        side = row["somaSide"]

        if side == "L":
            dna02_left_indices.append(idx)

        elif side == "R":
            dna02_right_indices.append(idx)


    spike_counts = np.asarray(
        spike_monitor.count
    )


    left_spikes = int(
        spike_counts[
            dna02_left_indices
        ].sum()
    )

    right_spikes = int(
        spike_counts[
            dna02_right_indices
        ].sum()
    )


    steering_score = (
        right_spikes
        - left_spikes
    )


    if steering_score > 0:
        predicted_turn = "RIGHT"

    elif steering_score < 0:
        predicted_turn = "LEFT"

    else:
        predicted_turn = "BALANCED"


    return {
        "condition":
            condition_name,

        "DNa02_L_spikes":
            left_spikes,

        "DNa02_R_spikes":
            right_spikes,

        "steering_score_R_minus_L":
            steering_score,

        "predicted_turn":
            predicted_turn,
    }


# =========================================================
# Test both channels
# =========================================================

results = []


results.append(
    run_condition(
        "stimulate_LEFT_turn_PFL3",
        pfl3_left_turn,
    )
)


results.append(
    run_condition(
        "stimulate_RIGHT_turn_PFL3",
        pfl3_right_turn,
    )
)


results_df = pd.DataFrame(
    results
)


# =========================================================
# Results
# =========================================================

print()
print("=" * 80)
print("STEERING SIMULATION RESULTS")
print("=" * 80)
print()

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    RESULT_FILE,
    index=False,
)


print()
print("Saved:")
print(RESULT_FILE)