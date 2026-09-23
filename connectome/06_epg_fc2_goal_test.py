import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from brian2 import (
    Network,
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    prefs,
    ms,
)


# =========================================================
# Brian2 configuration
# =========================================================

prefs.codegen.target = "numpy"


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

NEURON_FILE = (
    OUTPUT_DIR
    / "steering_core_neurons.csv"
)

EDGE_FILE = (
    OUTPUT_DIR
    / "steering_core_edges.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "epg_fc2_goal_test.csv"
)


# =========================================================
# Load MaleCNS data
# =========================================================

neurons = pd.read_csv(
    NEURON_FILE
)

edges = pd.read_csv(
    EDGE_FILE
)

neurons["bodyId"] = (
    neurons["bodyId"]
    .astype("int64")
)

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
# Neuron families
# =========================================================

NETWORK_TYPES = [
    "EPG",
    "FC2A",
    "FC2B",
    "FC2C",
    "PFL2",
    "PFL3",
    "DNa03",
    "DNa02",
]


core_neurons = neurons[
    neurons["type"].isin(
        NETWORK_TYPES
    )
].copy()

core_neurons = (
    core_neurons
    .reset_index(drop=True)
)


core_ids = set(
    core_neurons["bodyId"]
)


core_edges = edges[
    edges["source_bodyId"].isin(
        core_ids
    )
    &
    edges["target_bodyId"].isin(
        core_ids
    )
].copy()


# =========================================================
# Keep functional navigation paths
#
# EPG / FC2
#       ↓
# PFL2 / PFL3
#       ↓
# DNa03 / DNa02
#
# DNa03
#       ↓
# DNa02
# =========================================================

input_types = [
    "EPG",
    "FC2A",
    "FC2B",
    "FC2C",
]

pfl_types = [
    "PFL2",
    "PFL3",
]


valid_edges = (
    (
        core_edges["source_type"]
        .isin(input_types)
        &
        core_edges["target_type"]
        .isin(pfl_types)
    )
    |
    (
        core_edges["source_type"]
        .isin(pfl_types)
        &
        core_edges["target_type"]
        .isin(
            [
                "DNa03",
                "DNa02",
            ]
        )
    )
    |
    (
        (
            core_edges["source_type"]
            == "DNa03"
        )
        &
        (
            core_edges["target_type"]
            == "DNa02"
        )
    )
)


core_edges = core_edges[
    valid_edges
].copy()


print(
    "=" * 90
)

print(
    "EPG + FC2 GOAL NAVIGATION TEST"
)

print(
    "=" * 90
)

print(
    f"Neurons: {len(core_neurons)}"
)

print(
    f"Functional edges: "
    f"{len(core_edges)}"
)


# =========================================================
# Utilities
# =========================================================

def wrap_angle(angle):

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def circular_tuning(
    neuron_phase,
    stimulus_phase,
    sharpness=3.0,
):

    difference = wrap_angle(
        neuron_phase
        - stimulus_phase
    )

    value = math.exp(
        sharpness
        * math.cos(
            difference
        )
    )

    maximum = math.exp(
        sharpness
    )

    return (
        value
        / maximum
    )


# =========================================================
# Biological column phase mapping
#
# IMPORTANT:
#
# These phases are an engineered interface between
# angular information and connectome columns.
#
# The synaptic topology remains MaleCNS-derived.
#
# This first experiment checks whether the biological
# connectivity produces sensible steering from these
# population patterns.
# =========================================================

def get_epg_phase(instance):

    if not isinstance(
        instance,
        str,
    ):
        return None

    match = re.search(
        r"_([LR])(\d+)$",
        instance,
    )

    if not match:
        return None

    side = match.group(1)

    number = int(
        match.group(2)
    )

    if not (
        1 <= number <= 8
    ):
        return None

    # Arrange PB columns around a circular compass:
    #
    # L1 L2 ... L8 R8 ... R2 R1

    if side == "L":

        index = (
            number
            - 1
        )

    else:

        index = (
            16
            - number
        )

    phase = (
        -math.pi
        +
        (
            2.0
            * math.pi
            * index
            / 16.0
        )
    )

    return phase


def get_fc2_phase(instance):

    if not isinstance(
        instance,
        str,
    ):
        return None

    match = re.search(
        r"_C(\d+)_",
        instance,
    )

    if not match:
        return None

    column = int(
        match.group(1)
    )

    if not (
        1 <= column <= 9
    ):
        return None

    phase = (
        -math.pi
        +
        (
            2.0
            * math.pi
            * (
                column
                - 1
            )
            / 9.0
        )
    )

    return phase


# =========================================================
# Pre-calculate phases
# =========================================================

epg_phase = {}

fc2_phase = {}


for index, row in core_neurons.iterrows():

    neuron_type = row["type"]
    instance = row["instance"]

    if neuron_type == "EPG":

        phase = get_epg_phase(
            instance
        )

        if phase is not None:

            epg_phase[index] = (
                phase
            )

    elif neuron_type in [
        "FC2A",
        "FC2B",
        "FC2C",
    ]:

        phase = get_fc2_phase(
            instance
        )

        if phase is not None:

            fc2_phase[index] = (
                phase
            )


print()

print(
    f"EPG neurons with compass phase: "
    f"{len(epg_phase)}"
)

print(
    f"FC2 neurons with goal phase: "
    f"{len(fc2_phase)}"
)


# =========================================================
# Build lookup
# =========================================================

body_to_index = {
    int(body_id): index
    for index, body_id
    in enumerate(
        core_neurons[
            "bodyId"
        ]
    )
}


# =========================================================
# DNa02 indices
# =========================================================

dna_left = []
dna_right = []


for index, row in core_neurons.iterrows():

    if row["type"] != "DNa02":
        continue

    if row["somaSide"] == "L":

        dna_left.append(
            index
        )

    elif row["somaSide"] == "R":

        dna_right.append(
            index
        )


# =========================================================
# Run one heading-error condition
#
# We keep current heading at 0°.
#
# The goal direction is represented relative to it.
#
# Therefore:
#
# error +90°
#     goal lies to LEFT
#
# error -90°
#     goal lies to RIGHT
#
# =========================================================

def run_condition(
    heading_error_deg,
):

    heading_error = math.radians(
        heading_error_deg
    )

    N = len(
        core_neurons
    )


    # -----------------------------------------------------
    # Neuron model
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

    G.drive = 0.03


    # -----------------------------------------------------
    # Current-heading representation
    #
    # Reference current heading = 0°
    # -----------------------------------------------------

    EPG_GAIN = 1.30


    for index, phase in epg_phase.items():

        activity = (
            circular_tuning(
                phase,
                0.0,
                sharpness=3.5,
            )
        )

        G.drive[index] = (
            0.03
            +
            EPG_GAIN
            * activity
        )


    # -----------------------------------------------------
    # Goal-direction representation
    #
    # FC2 bump is shifted according to heading error.
    # -----------------------------------------------------

    FC2_GAIN = 1.35


    for index, phase in fc2_phase.items():

        activity = (
            circular_tuning(
                phase,
                heading_error,
                sharpness=3.0,
            )
        )

        G.drive[index] = (
            0.03
            +
            FC2_GAIN
            * activity
        )


    # -----------------------------------------------------
    # MaleCNS synaptic network
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

        source_body = int(
            row[
                "source_bodyId"
            ]
        )

        target_body = int(
            row[
                "target_bodyId"
            ]
        )

        if (
            source_body
            not in body_to_index
            or
            target_body
            not in body_to_index
        ):
            continue

        source_indices.append(
            body_to_index[
                source_body
            ]
        )

        target_indices.append(
            body_to_index[
                target_body
            ]
        )

        raw_weights.append(
            float(
                row["weight"]
            )
        )


    S.connect(
        i=source_indices,
        j=target_indices,
    )


    raw_weights = np.asarray(
        raw_weights,
        dtype=float,
    )


    max_raw = max(
        raw_weights.max(),
        1.0,
    )


    # Preserve relative anatomical connection
    # strengths but scale into LIF range.
    scaled_weights = (
        0.025
        +
        0.32
        * np.sqrt(
            raw_weights
            / max_raw
        )
    )


    S.w = scaled_weights


    # -----------------------------------------------------
    # Monitor
    # -----------------------------------------------------

    monitor = SpikeMonitor(
        G
    )


    network = Network(
        G,
        S,
        monitor,
    )


    network.run(
        400 * ms
    )


    counts = np.asarray(
        monitor.count
    )


    left_spikes = int(
        counts[
            dna_left
        ].sum()
    )

    right_spikes = int(
        counts[
            dna_right
        ].sum()
    )


    # -----------------------------------------------------
    # DNa02 membrane state
    # -----------------------------------------------------

    voltage = np.asarray(
        G.v
    )


    left_v = float(
        voltage[
            dna_left
        ].mean()
    )

    right_v = float(
        voltage[
            dna_right
        ].mean()
    )


    # -----------------------------------------------------
    # Steering output
    #
    # Positive:
    #     RIGHT turn
    #
    # Negative:
    #     LEFT turn
    # -----------------------------------------------------

    spike_total = (
        left_spikes
        +
        right_spikes
    )


    if spike_total > 0:

        spike_signal = (
            right_spikes
            - left_spikes
        ) / spike_total

    else:

        spike_signal = 0.0


    membrane_signal = (
        right_v
        - left_v
    )


    steering = (
        0.85
        * spike_signal
        +
        0.15
        * membrane_signal
    )


    steering = float(
        np.clip(
            steering,
            -1.0,
            1.0,
        )
    )


    # -----------------------------------------------------
    # Expected direction
    #
    # Positive heading error:
    #     goal is to robot's LEFT
    #
    # Negative heading error:
    #     goal is to robot's RIGHT
    # -----------------------------------------------------

    if heading_error_deg > 10:

        expected = "LEFT"

    elif heading_error_deg < -10:

        expected = "RIGHT"

    else:

        expected = "STRAIGHT"


    if steering < -0.05:

        predicted = "LEFT"

    elif steering > 0.05:

        predicted = "RIGHT"

    else:

        predicted = "STRAIGHT"


    return {
        "heading_error_deg":
            heading_error_deg,

        "DNa02_L_spikes":
            left_spikes,

        "DNa02_R_spikes":
            right_spikes,

        "DNa02_L_v":
            left_v,

        "DNa02_R_v":
            right_v,

        "steering":
            steering,

        "expected":
            expected,

        "predicted":
            predicted,

        "correct":
            expected
            == predicted,
    }


# =========================================================
# Test angular errors
# =========================================================

TEST_ERRORS = [
    -150,
    -120,
    -90,
    -60,
    -30,
    0,
    30,
    60,
    90,
    120,
    150,
]


results = []


for error in TEST_ERRORS:

    print(
        f"Testing heading error "
        f"{error:+4d} degrees..."
    )

    result = run_condition(
        error
    )

    results.append(
        result
    )


results_df = pd.DataFrame(
    results
)


# =========================================================
# Results
# =========================================================

print()

print(
    "=" * 110
)

print(
    "EPG + FC2 GOAL STEERING RESULTS"
)

print(
    "=" * 110
)

print()

print(
    results_df.to_string(
        index=False
    )
)


accuracy = (
    results_df[
        "correct"
    ].mean()
)


print()

print(
    f"Direction accuracy: "
    f"{accuracy * 100:.1f}%"
)


results_df.to_csv(
    RESULT_FILE,
    index=False,
)


print()

print(
    "Saved:"
)

print(
    RESULT_FILE
)