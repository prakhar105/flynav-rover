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
# Brian2
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
    / "goal_phase_calibration.csv"
)


# =========================================================
# Load MaleCNS
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
# Navigation network
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


core_neurons = (
    neurons[
        neurons["type"].isin(
            NETWORK_TYPES
        )
    ]
    .copy()
    .reset_index(drop=True)
)


core_ids = set(
    core_neurons["bodyId"]
)


core_edges = edges[
    edges["source_bodyId"]
    .isin(core_ids)
    &
    edges["target_bodyId"]
    .isin(core_ids)
].copy()


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


core_edges = (
    core_edges[
        valid_edges
    ]
    .copy()
)


print("=" * 90)
print("GOAL PHASE CALIBRATION")
print("=" * 90)

print(
    f"Neurons: {len(core_neurons)}"
)

print(
    f"Edges: {len(core_edges)}"
)


# =========================================================
# Angle helpers
# =========================================================

def wrap_angle(angle):

    while angle > math.pi:
        angle -= (
            2.0 * math.pi
        )

    while angle < -math.pi:
        angle += (
            2.0 * math.pi
        )

    return angle


def circular_tuning(
    neuron_phase,
    stimulus_phase,
    sharpness,
):

    difference = wrap_angle(
        neuron_phase
        - stimulus_phase
    )

    numerator = math.exp(
        sharpness
        * math.cos(
            difference
        )
    )

    denominator = math.exp(
        sharpness
    )

    return (
        numerator
        / denominator
    )


# =========================================================
# EPG phase
#
# Keep our current mapping fixed for this experiment.
#
# We are only calibrating FC2 relative to EPG.
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

    return (
        -math.pi
        +
        2.0
        * math.pi
        * index
        / 16.0
    )


# =========================================================
# FC2 phase
#
# Two unknowns:
#
# direction:
#     +1 or -1
#
# offset:
#     arbitrary rotation relative to EPG
# =========================================================

def get_fc2_base_phase(
    instance,
    direction,
):

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

    normalized = (
        column
        - 1
    ) / 9.0

    return (
        direction
        * 2.0
        * math.pi
        * normalized
    )


# =========================================================
# Static metadata
# =========================================================

body_to_index = {
    int(body_id): index
    for index, body_id
    in enumerate(
        core_neurons["bodyId"]
    )
}


epg_phase = {}


for index, row in (
    core_neurons
    .iterrows()
):

    if row["type"] != "EPG":
        continue

    phase = get_epg_phase(
        row["instance"]
    )

    if phase is not None:

        epg_phase[index] = (
            phase
        )


dna_left = []
dna_right = []


for index, row in (
    core_neurons
    .iterrows()
):

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
# Pre-build synapse arrays
# =========================================================

source_indices = []
target_indices = []
raw_weights = []


for _, row in (
    core_edges
    .iterrows()
):

    source = int(
        row["source_bodyId"]
    )

    target = int(
        row["target_bodyId"]
    )

    if (
        source not in body_to_index
        or
        target not in body_to_index
    ):
        continue

    source_indices.append(
        body_to_index[
            source
        ]
    )

    target_indices.append(
        body_to_index[
            target
        ]
    )

    raw_weights.append(
        float(
            row["weight"]
        )
    )


raw_weights = np.asarray(
    raw_weights,
    dtype=float,
)


max_raw = max(
    raw_weights.max(),
    1.0,
)


scaled_weights = (
    0.025
    +
    0.32
    * np.sqrt(
        raw_weights
        / max_raw
    )
)


# =========================================================
# Run one simulated angle
# =========================================================

def simulate(
    heading_error_deg,
    fc2_offset_deg,
    fc2_direction,
):

    N = len(
        core_neurons
    )

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
    # EPG = current heading
    #
    # Reference heading = 0 degrees.
    # -----------------------------------------------------

    EPG_GAIN = 1.30


    for index, phase in (
        epg_phase.items()
    ):

        activity = circular_tuning(
            phase,
            0.0,
            sharpness=3.5,
        )

        G.drive[index] = (
            0.03
            +
            EPG_GAIN
            * activity
        )


    # -----------------------------------------------------
    # FC2 = goal direction
    # -----------------------------------------------------

    goal_phase = math.radians(
        heading_error_deg
    )

    offset = math.radians(
        fc2_offset_deg
    )

    FC2_GAIN = 1.35


    for index, row in (
        core_neurons
        .iterrows()
    ):

        if row["type"] not in [
            "FC2A",
            "FC2B",
            "FC2C",
        ]:
            continue

        phase = (
            get_fc2_base_phase(
                row["instance"],
                fc2_direction,
            )
        )

        if phase is None:
            continue

        phase = wrap_angle(
            phase
            + offset
        )

        activity = circular_tuning(
            phase,
            goal_phase,
            sharpness=3.0,
        )

        G.drive[index] = (
            0.03
            +
            FC2_GAIN
            * activity
        )


    # -----------------------------------------------------
    # Connectome
    # -----------------------------------------------------

    S = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )

    S.connect(
        i=source_indices,
        j=target_indices,
    )

    S.w = (
        scaled_weights
    )


    monitor = SpikeMonitor(
        G
    )


    network = Network(
        G,
        S,
        monitor,
    )


    network.run(
        300 * ms
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


    return float(
        steering
    )


# =========================================================
# Calibration set
#
# These angles choose phase alignment.
# =========================================================

CALIBRATION_ERRORS = [
    -120,
    -60,
    0,
    60,
    120,
]


# =========================================================
# Held-out validation set
#
# These angles are NOT used to select the mapping.
# =========================================================

VALIDATION_ERRORS = [
    -150,
    -90,
    -30,
    30,
    90,
    150,
]


# =========================================================
# Target steering
#
# Smooth desired angular relation:
#
# RIGHT = positive
# LEFT  = negative
#
# target = -sin(error)
# =========================================================

def target_steering(
    error_deg,
):

    return (
        -math.sin(
            math.radians(
                error_deg
            )
        )
    )


# =========================================================
# Score a configuration
# =========================================================

def evaluate_configuration(
    offset_deg,
    direction,
    errors,
):

    predicted = []
    target = []


    for error in errors:

        steering = simulate(
            heading_error_deg=error,
            fc2_offset_deg=offset_deg,
            fc2_direction=direction,
        )

        predicted.append(
            steering
        )

        target.append(
            target_steering(
                error
            )
        )


    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )


    # -----------------------------------------------------
    # Correlation with desired steering shape
    # -----------------------------------------------------

    if (
        np.std(predicted)
        < 1e-8
    ):

        correlation = -1.0

    else:

        correlation = float(
            np.corrcoef(
                predicted,
                target,
            )[0, 1]
        )


    # -----------------------------------------------------
    # Sign accuracy
    #
    # Ignore error = 0.
    # -----------------------------------------------------

    correct = 0
    total = 0


    for error, output in zip(
        errors,
        predicted,
    ):

        if error == 0:
            continue

        expected_sign = (
            1
            if error < 0
            else -1
        )

        actual_sign = (
            1
            if output > 0
            else (
                -1
                if output < 0
                else 0
            )
        )

        if (
            actual_sign
            == expected_sign
        ):
            correct += 1

        total += 1


    sign_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )


    # -----------------------------------------------------
    # Neutral bias at zero
    # -----------------------------------------------------

    neutral_output = 0.0

    if 0 in errors:

        zero_index = (
            errors.index(0)
        )

        neutral_output = abs(
            predicted[
                zero_index
            ]
        )


    # -----------------------------------------------------
    # Combined calibration score
    # -----------------------------------------------------

    score = (
        correlation
        +
        0.50
        * sign_accuracy
        -
        0.50
        * neutral_output
    )


    return {
        "score":
            score,

        "correlation":
            correlation,

        "sign_accuracy":
            sign_accuracy,

        "neutral_bias":
            neutral_output,

        "outputs":
            predicted,
    }


# =========================================================
# Coarse search
# =========================================================

rows = []


OFFSETS = list(
    range(
        0,
        360,
        15,
    )
)


for direction in [
    -1,
    1,
]:

    for offset in OFFSETS:

        print(
            f"Testing "
            f"direction={direction:+d}, "
            f"offset={offset:3d}deg"
        )

        result = (
            evaluate_configuration(
                offset_deg=offset,
                direction=direction,
                errors=CALIBRATION_ERRORS,
            )
        )


        rows.append(
            {
                "direction":
                    direction,

                "offset_deg":
                    offset,

                "calibration_score":
                    result["score"],

                "calibration_correlation":
                    result[
                        "correlation"
                    ],

                "calibration_sign_accuracy":
                    result[
                        "sign_accuracy"
                    ],

                "neutral_bias":
                    result[
                        "neutral_bias"
                    ],
            }
        )


results = pd.DataFrame(
    rows
)


results = results.sort_values(
    "calibration_score",
    ascending=False,
).reset_index(
    drop=True
)


# =========================================================
# Best configuration
# =========================================================

best = results.iloc[0]

BEST_DIRECTION = int(
    best["direction"]
)

BEST_OFFSET = float(
    best["offset_deg"]
)


print()
print("=" * 100)
print("BEST CALIBRATION")
print("=" * 100)

print(
    results.head(10)
    .to_string(
        index=False
    )
)


print()
print(
    f"BEST DIRECTION: "
    f"{BEST_DIRECTION:+d}"
)

print(
    f"BEST OFFSET: "
    f"{BEST_OFFSET:.1f} degrees"
)


# =========================================================
# Held-out validation
# =========================================================

print()
print("=" * 100)
print("HELD-OUT VALIDATION")
print("=" * 100)


validation_rows = []


for error in (
    VALIDATION_ERRORS
):

    steering = simulate(
        heading_error_deg=error,
        fc2_offset_deg=BEST_OFFSET,
        fc2_direction=BEST_DIRECTION,
    )

    target = target_steering(
        error
    )


    expected = (
        "RIGHT"
        if error < 0
        else "LEFT"
    )


    predicted = (
        "RIGHT"
        if steering > 0
        else (
            "LEFT"
            if steering < 0
            else "STRAIGHT"
        )
    )


    validation_rows.append(
        {
            "heading_error_deg":
                error,

            "target":
                target,

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
    )


validation = pd.DataFrame(
    validation_rows
)


print()
print(
    validation.to_string(
        index=False
    )
)


accuracy = float(
    validation[
        "correct"
    ].mean()
)


correlation = float(
    np.corrcoef(
        validation[
            "steering"
        ],
        validation[
            "target"
        ],
    )[0, 1]
)


print()
print(
    f"Validation direction accuracy: "
    f"{accuracy * 100:.1f}%"
)

print(
    f"Validation steering correlation: "
    f"{correlation:.3f}"
)


# =========================================================
# Save
# =========================================================

results.to_csv(
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