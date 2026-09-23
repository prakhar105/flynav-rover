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

STEERING_NEURON_FILE = (
    OUTPUT_DIR
    / "steering_core_neurons.csv"
)

STEERING_EDGE_FILE = (
    OUTPUT_DIR
    / "steering_core_edges.csv"
)

DELTA7_FILE = (
    OUTPUT_DIR
    / "delta7_candidates.csv"
)

DELTA7_EDGE_FILE = (
    OUTPUT_DIR
    / "delta7_bridge_edges.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "delta7_sign_comparison.csv"
)


# =========================================================
# Load data
# =========================================================

steering_neurons = pd.read_csv(
    STEERING_NEURON_FILE
)

steering_edges = pd.read_csv(
    STEERING_EDGE_FILE
)

delta7_neurons = pd.read_csv(
    DELTA7_FILE
)

delta7_edges = pd.read_csv(
    DELTA7_EDGE_FILE
)


# =========================================================
# Normalize IDs / weights
# =========================================================

for df in [
    steering_neurons,
    delta7_neurons,
]:
    df["bodyId"] = (
        df["bodyId"]
        .astype("int64")
    )


for df in [
    steering_edges,
    delta7_edges,
]:

    df["source_bodyId"] = (
        df["source_bodyId"]
        .astype("int64")
    )

    df["target_bodyId"] = (
        df["target_bodyId"]
        .astype("int64")
    )

    df["weight"] = pd.to_numeric(
        df["weight"],
        errors="coerce",
    ).fillna(0.0)


# =========================================================
# Combine neuron populations
# =========================================================

NETWORK_TYPES = [
    "EPG",
    "Delta7",
    "FC2A",
    "FC2B",
    "FC2C",
    "PFL2",
    "PFL3",
    "DNa03",
    "DNa02",
]


neurons = pd.concat(
    [
        steering_neurons,
        delta7_neurons,
    ],
    ignore_index=True,
)


neurons = (
    neurons
    .drop_duplicates(
        subset=["bodyId"]
    )
)


neurons = (
    neurons[
        neurons["type"]
        .isin(NETWORK_TYPES)
    ]
    .copy()
    .reset_index(drop=True)
)


body_ids = set(
    neurons["bodyId"]
)


# =========================================================
# Combine edge tables
# =========================================================

edges = pd.concat(
    [
        steering_edges,
        delta7_edges,
    ],
    ignore_index=True,
)


edges = edges[
    edges["source_bodyId"]
    .isin(body_ids)
    &
    edges["target_bodyId"]
    .isin(body_ids)
].copy()


# Remove duplicate biological edges caused by
# combining the two query outputs.
edges = (
    edges
    .sort_values(
        "weight",
        ascending=False,
    )
    .drop_duplicates(
        subset=[
            "source_bodyId",
            "target_bodyId",
        ],
        keep="first",
    )
)


# =========================================================
# Functional navigation routes
#
# Heading:
# EPG -> Delta7 -> PFL
#
# Direct heading path:
# EPG -> PFL
#
# Goal:
# FC2 -> PFL
#
# Steering:
# PFL -> DNa
# DNa03 -> DNa02
# =========================================================

functional = (
    (
        (edges["source_type"] == "EPG")
        &
        (edges["target_type"] == "Delta7")
    )
    |
    (
        (edges["source_type"] == "EPG")
        &
        edges["target_type"].isin(
            ["PFL2", "PFL3"]
        )
    )
    |
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"].isin(
            ["PFL2", "PFL3"]
        )
    )
    |
    (
        edges["source_type"].isin(
            [
                "FC2A",
                "FC2B",
                "FC2C",
            ]
        )
        &
        edges["target_type"].isin(
            ["PFL2", "PFL3"]
        )
    )
    |
    (
        edges["source_type"].isin(
            ["PFL2", "PFL3"]
        )
        &
        edges["target_type"].isin(
            ["DNa03", "DNa02"]
        )
    )
    |
    (
        (edges["source_type"] == "DNa03")
        &
        (edges["target_type"] == "DNa02")
    )
)


edges = (
    edges[
        functional
    ]
    .copy()
)


print("=" * 95)
print("DELTA7 SIGN EXPERIMENT")
print("=" * 95)

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)


print()
print(
    neurons.groupby("type")
    .size()
    .to_string()
)


# =========================================================
# Helpers
# =========================================================

def wrap_angle(angle):

    while angle > math.pi:
        angle -= (
            2.0
            * math.pi
        )

    while angle < -math.pi:
        angle += (
            2.0
            * math.pi
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
# EPG compass phase
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
# FC2 goal phase
#
# Keep the previous best calibration FIXED.
#
# We are NOT recalibrating here because we want
# Delta7 sign to be the only experimental variable.
# =========================================================

FC2_DIRECTION = 1
FC2_OFFSET_DEG = 150.0


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


    normalized = (
        column
        - 1
    ) / 9.0


    phase = (
        FC2_DIRECTION
        * 2.0
        * math.pi
        * normalized
    )


    phase += math.radians(
        FC2_OFFSET_DEG
    )


    return wrap_angle(
        phase
    )


# =========================================================
# Metadata
# =========================================================

body_to_index = {
    int(body_id): index
    for index, body_id
    in enumerate(
        neurons["bodyId"]
    )
}


epg_phase = {}
fc2_phase = {}


for index, row in neurons.iterrows():

    if row["type"] == "EPG":

        phase = get_epg_phase(
            row["instance"]
        )

        if phase is not None:
            epg_phase[index] = phase


    elif row["type"] in [
        "FC2A",
        "FC2B",
        "FC2C",
    ]:

        phase = get_fc2_phase(
            row["instance"]
        )

        if phase is not None:
            fc2_phase[index] = phase


print()
print(
    f"EPG phase neurons: "
    f"{len(epg_phase)}"
)

print(
    f"FC2 phase neurons: "
    f"{len(fc2_phase)}"
)


# =========================================================
# Output neurons
# =========================================================

dna_left = []
dna_right = []


for index, row in neurons.iterrows():

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
# Divide synapses
#
# Positive population:
# all edges except Delta7 -> PFL
#
# Delta population:
# only Delta7 -> PFL
# =========================================================

positive_edges = edges[
    ~(
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"].isin(
            ["PFL2", "PFL3"]
        )
    )
].copy()


delta_edges = edges[
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"].isin(
            ["PFL2", "PFL3"]
        )
    )
].copy()


print()
print(
    f"Positive/default edges: "
    f"{len(positive_edges)}"
)

print(
    f"Delta7 -> PFL edges: "
    f"{len(delta_edges)}"
)


# =========================================================
# Prepare synapse data
# =========================================================

def prepare_edges(df):

    source_indices = []
    target_indices = []
    raw_weights = []


    for _, row in df.iterrows():

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
            body_to_index[source]
        )

        target_indices.append(
            body_to_index[target]
        )

        raw_weights.append(
            float(
                row["weight"]
            )
        )


    return (
        source_indices,
        target_indices,
        np.asarray(
            raw_weights,
            dtype=float,
        ),
    )


(
    positive_source,
    positive_target,
    positive_raw,
) = prepare_edges(
    positive_edges
)


(
    delta_source,
    delta_target,
    delta_raw,
) = prepare_edges(
    delta_edges
)


# =========================================================
# Use ONE common anatomical scaling
#
# This preserves relative edge strength across the
# entire connectome rather than scaling populations
# independently.
# =========================================================

all_raw = np.concatenate(
    [
        positive_raw,
        delta_raw,
    ]
)


MAX_RAW = max(
    all_raw.max(),
    1.0,
)


def scale_weights(raw):

    return (
        0.02
        +
        0.30
        * np.sqrt(
            raw
            / MAX_RAW
        )
    )


positive_weights = scale_weights(
    positive_raw
)

delta_weights = scale_weights(
    delta_raw
)


# =========================================================
# Run one condition
#
# delta_sign:
#
# +1 = excitatory hypothesis
# -1 = inhibitory hypothesis
# =========================================================

def simulate(
    heading_error_deg,
    delta_sign,
):

    N = len(
        neurons
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
    # EPG current-heading bump
    #
    # Current heading reference = 0°
    # -----------------------------------------------------

    EPG_GAIN = 1.30


    for index, phase in (
        epg_phase.items()
    ):

        activity = circular_tuning(
            neuron_phase=phase,
            stimulus_phase=0.0,
            sharpness=3.5,
        )


        G.drive[index] = (
            0.03
            +
            EPG_GAIN
            * activity
        )


    # -----------------------------------------------------
    # FC2 goal-direction bump
    # -----------------------------------------------------

    goal_phase = math.radians(
        heading_error_deg
    )


    FC2_GAIN = 1.35


    for index, phase in (
        fc2_phase.items()
    ):

        activity = circular_tuning(
            neuron_phase=phase,
            stimulus_phase=goal_phase,
            sharpness=3.0,
        )


        G.drive[index] = (
            0.03
            +
            FC2_GAIN
            * activity
        )


    # -----------------------------------------------------
    # Positive/default synapses
    # -----------------------------------------------------

    S_positive = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )


    S_positive.connect(
        i=positive_source,
        j=positive_target,
    )


    S_positive.w = (
        positive_weights
    )


    # -----------------------------------------------------
    # Delta7 -> PFL synapses
    # -----------------------------------------------------

    if delta_sign > 0:

        delta_rule = (
            "v_post += w"
        )

    else:

        delta_rule = (
            "v_post -= w"
        )


    S_delta = Synapses(
        G,
        G,
        model="w : 1",
        on_pre=delta_rule,
    )


    S_delta.connect(
        i=delta_source,
        j=delta_target,
    )


    S_delta.w = (
        delta_weights
    )


    # -----------------------------------------------------
    # Monitor
    # -----------------------------------------------------

    monitor = SpikeMonitor(
        G
    )


    network = Network(
        G,
        S_positive,
        S_delta,
        monitor,
    )


    network.run(
        400 * ms
    )


    # -----------------------------------------------------
    # DNa02 output
    # -----------------------------------------------------

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


    steering = float(
        np.clip(
            steering,
            -1.0,
            1.0,
        )
    )


    return {
        "left_spikes":
            left_spikes,

        "right_spikes":
            right_spikes,

        "left_v":
            left_v,

        "right_v":
            right_v,

        "steering":
            steering,
    }


# =========================================================
# Test grid
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


HYPOTHESES = {
    "EXCITATORY":
        1,

    "INHIBITORY":
        -1,
}


rows = []


for hypothesis, sign in (
    HYPOTHESES.items()
):

    print()
    print("=" * 95)
    print(
        f"TESTING DELTA7 {hypothesis}"
    )
    print("=" * 95)


    for error in TEST_ERRORS:

        print(
            f"{hypothesis:<10} | "
            f"error={error:+4d}°"
        )


        result = simulate(
            heading_error_deg=error,
            delta_sign=sign,
        )


        if error < 0:

            expected = "RIGHT"

        elif error > 0:

            expected = "LEFT"

        else:

            expected = "STRAIGHT"


        if result["steering"] > 0:

            predicted = "RIGHT"

        elif result["steering"] < 0:

            predicted = "LEFT"

        else:

            predicted = "STRAIGHT"


        target = (
            -math.sin(
                math.radians(
                    error
                )
            )
        )


        rows.append(
            {
                "hypothesis":
                    hypothesis,

                "heading_error_deg":
                    error,

                "target":
                    target,

                "DNa02_L_spikes":
                    result[
                        "left_spikes"
                    ],

                "DNa02_R_spikes":
                    result[
                        "right_spikes"
                    ],

                "DNa02_L_v":
                    result[
                        "left_v"
                    ],

                "DNa02_R_v":
                    result[
                        "right_v"
                    ],

                "steering":
                    result[
                        "steering"
                    ],

                "expected":
                    expected,

                "predicted":
                    predicted,

                "correct":
                    (
                        expected
                        == predicted
                    ),
            }
        )


results = pd.DataFrame(
    rows
)


# =========================================================
# Print full results
# =========================================================

print()
print("=" * 110)
print("FULL RESULTS")
print("=" * 110)
print()

print(
    results.to_string(
        index=False
    )
)


# =========================================================
# Summary
# =========================================================

summary_rows = []


for hypothesis in (
    HYPOTHESES.keys()
):

    subset = results[
        results["hypothesis"]
        == hypothesis
    ].copy()


    nonzero = subset[
        subset[
            "heading_error_deg"
        ] != 0
    ]


    direction_accuracy = float(
        nonzero[
            "correct"
        ].mean()
    )


    correlation = float(
        np.corrcoef(
            subset[
                "steering"
            ],
            subset[
                "target"
            ],
        )[0, 1]
    )


    zero_row = subset[
        subset[
            "heading_error_deg"
        ] == 0
    ].iloc[0]


    neutral_bias = abs(
        float(
            zero_row[
                "steering"
            ]
        )
    )


    mean_magnitude = float(
        nonzero[
            "steering"
        ]
        .abs()
        .mean()
    )


    summary_rows.append(
        {
            "hypothesis":
                hypothesis,

            "direction_accuracy":
                direction_accuracy,

            "steering_correlation":
                correlation,

            "neutral_bias":
                neutral_bias,

            "mean_nonzero_magnitude":
                mean_magnitude,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print()
print("=" * 110)
print("HYPOTHESIS SUMMARY")
print("=" * 110)
print()

print(
    summary.to_string(
        index=False
    )
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