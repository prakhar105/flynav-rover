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
    OUTPUT_DIR / "steering_core_neurons.csv"
)

STEERING_EDGE_FILE = (
    OUTPUT_DIR / "steering_core_edges.csv"
)

DELTA7_FILE = (
    OUTPUT_DIR / "delta7_candidates.csv"
)

DELTA7_EDGE_FILE = (
    OUTPUT_DIR / "delta7_bridge_edges.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "inhibitory_delta7_phase_calibration.csv"
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
# Normalize
# =========================================================

for df in [
    steering_neurons,
    delta7_neurons,
]:
    df["bodyId"] = (
        df["bodyId"].astype("int64")
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
# Neurons
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
# Edges
# =========================================================

edges = pd.concat(
    [
        steering_edges,
        delta7_edges,
    ],
    ignore_index=True,
)


edges = edges[
    edges["source_bodyId"].isin(body_ids)
    &
    edges["target_bodyId"].isin(body_ids)
].copy()


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
# Functional circuit
# =========================================================

functional = (

    # EPG -> Delta7
    (
        (edges["source_type"] == "EPG")
        &
        (edges["target_type"] == "Delta7")
    )

    |

    # Direct EPG -> PFL
    (
        (edges["source_type"] == "EPG")
        &
        edges["target_type"]
        .isin(["PFL2", "PFL3"])
    )

    |

    # Delta7 -> PFL
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"]
        .isin(["PFL2", "PFL3"])
    )

    |

    # FC2 -> PFL
    (
        edges["source_type"]
        .isin(
            [
                "FC2A",
                "FC2B",
                "FC2C",
            ]
        )
        &
        edges["target_type"]
        .isin(["PFL2", "PFL3"])
    )

    |

    # PFL -> descending neurons
    (
        edges["source_type"]
        .isin(["PFL2", "PFL3"])
        &
        edges["target_type"]
        .isin(["DNa03", "DNa02"])
    )

    |

    # DNa03 -> DNa02
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
print("INHIBITORY DELTA7 PHASE CALIBRATION")
print("=" * 95)

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)


# =========================================================
# Angle helpers
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
    sharpness,
):

    difference = wrap_angle(
        neuron_phase
        - stimulus_phase
    )

    value = math.exp(
        sharpness
        * math.cos(difference)
    )

    maximum = math.exp(
        sharpness
    )

    return value / maximum


# =========================================================
# EPG phase mapping
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
        index = number - 1
    else:
        index = 16 - number

    return (
        -math.pi
        +
        (
            2.0
            * math.pi
            * index
            / 16.0
        )
    )


# =========================================================
# FC2 phase mapping
# =========================================================

def get_fc2_phase(
    instance,
    direction,
    offset_deg,
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
        column - 1
    ) / 9.0

    phase = (
        direction
        * 2.0
        * math.pi
        * normalized
    )

    phase += math.radians(
        offset_deg
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


for index, row in neurons.iterrows():

    if row["type"] != "EPG":
        continue

    phase = get_epg_phase(
        row["instance"]
    )

    if phase is not None:
        epg_phase[index] = phase


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
# Separate inhibitory Delta7 edges
# =========================================================

delta_edges = edges[
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"]
        .isin(["PFL2", "PFL3"])
    )
].copy()


positive_edges = edges[
    ~(
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"]
        .isin(["PFL2", "PFL3"])
    )
].copy()


# =========================================================
# Prepare edges
# =========================================================

def prepare_edges(df):

    sources = []
    targets = []
    weights = []

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

        sources.append(
            body_to_index[source]
        )

        targets.append(
            body_to_index[target]
        )

        weights.append(
            float(row["weight"])
        )

    return (
        sources,
        targets,
        np.asarray(
            weights,
            dtype=float,
        ),
    )


(
    pos_source,
    pos_target,
    pos_raw,
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


all_raw = np.concatenate(
    [
        pos_raw,
        delta_raw,
    ]
)


MAX_RAW = max(
    float(all_raw.max()),
    1.0,
)


def scale_weights(raw):

    return (
        0.02
        +
        0.30
        * np.sqrt(
            raw / MAX_RAW
        )
    )


pos_weights = scale_weights(
    pos_raw
)

delta_weights = scale_weights(
    delta_raw
)


# =========================================================
# Simulation
# =========================================================

def simulate(
    heading_error_deg,
    fc2_direction,
    fc2_offset_deg,
):

    N = len(neurons)

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
    # EPG heading
    # -----------------------------------------------------

    EPG_GAIN = 1.30

    for index, phase in epg_phase.items():

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
    # FC2 goal
    # -----------------------------------------------------

    goal_phase = math.radians(
        heading_error_deg
    )

    FC2_GAIN = 1.35


    for index, row in neurons.iterrows():

        if row["type"] not in [
            "FC2A",
            "FC2B",
            "FC2C",
        ]:
            continue

        phase = get_fc2_phase(
            row["instance"],
            fc2_direction,
            fc2_offset_deg,
        )

        if phase is None:
            continue

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
    # Normal/excitatory synapses
    # -----------------------------------------------------

    S_positive = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )

    S_positive.connect(
        i=pos_source,
        j=pos_target,
    )

    S_positive.w = (
        pos_weights
    )


    # -----------------------------------------------------
    # Delta7 inhibitory synapses
    # -----------------------------------------------------

    S_delta = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post -= w",
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
    # DNa output
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


    total_spikes = (
        left_spikes
        +
        right_spikes
    )


    if total_spikes > 0:

        spike_signal = (
            right_spikes
            - left_spikes
        ) / total_spikes

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
        np.clip(
            steering,
            -1.0,
            1.0,
        )
    )


# =========================================================
# Target steering
# =========================================================

def target_steering(error_deg):

    return -math.sin(
        math.radians(
            error_deg
        )
    )


# =========================================================
# Calibration / validation split
# =========================================================

CALIBRATION_ERRORS = [
    -120,
    -60,
    0,
    60,
    120,
]


VALIDATION_ERRORS = [
    -150,
    -90,
    -30,
    30,
    90,
    150,
]


# =========================================================
# Evaluate configuration
# =========================================================

def evaluate(
    direction,
    offset,
    errors,
):

    predicted = []

    targets = []

    correct = 0

    directional_total = 0


    for error in errors:

        steering = simulate(
            heading_error_deg=error,
            fc2_direction=direction,
            fc2_offset_deg=offset,
        )

        target = target_steering(
            error
        )

        predicted.append(
            steering
        )

        targets.append(
            target
        )


        if error != 0:

            directional_total += 1

            expected_sign = (
                1
                if error < 0
                else -1
            )

            actual_sign = (
                1
                if steering > 0
                else (
                    -1
                    if steering < 0
                    else 0
                )
            )

            if (
                expected_sign
                == actual_sign
            ):
                correct += 1


    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )


    if (
        np.std(predicted)
        < 1e-10
    ):

        correlation = -1.0

    else:

        correlation = float(
            np.corrcoef(
                predicted,
                targets,
            )[0, 1]
        )


    direction_accuracy = (
        correct
        / directional_total
        if directional_total
        else 0.0
    )


    neutral_bias = 0.0

    if 0 in errors:

        zero_index = (
            errors.index(0)
        )

        neutral_bias = abs(
            float(
                predicted[
                    zero_index
                ]
            )
        )


    nonzero_magnitude = [
        abs(float(value))
        for error, value in zip(
            errors,
            predicted,
        )
        if error != 0
    ]


    mean_magnitude = (
        float(
            np.mean(
                nonzero_magnitude
            )
        )
        if nonzero_magnitude
        else 0.0
    )


    # Encourage:
    #
    # correct direction
    # smooth angular relationship
    # useful signal magnitude
    # low straight-ahead bias

    score = (
        correlation
        +
        0.50
        * direction_accuracy
        +
        0.15
        * mean_magnitude
        -
        0.50
        * neutral_bias
    )


    return {
        "score":
            score,

        "correlation":
            correlation,

        "direction_accuracy":
            direction_accuracy,

        "neutral_bias":
            neutral_bias,

        "mean_magnitude":
            mean_magnitude,
    }


# =========================================================
# Phase search
# =========================================================

rows = []


for direction in [
    -1,
    1,
]:

    for offset in range(
        0,
        360,
        15,
    ):

        print(
            f"Testing "
            f"direction={direction:+d} | "
            f"offset={offset:3d}°"
        )

        result = evaluate(
            direction=direction,
            offset=offset,
            errors=CALIBRATION_ERRORS,
        )


        rows.append(
            {
                "direction":
                    direction,

                "offset_deg":
                    offset,

                "score":
                    result["score"],

                "correlation":
                    result[
                        "correlation"
                    ],

                "direction_accuracy":
                    result[
                        "direction_accuracy"
                    ],

                "neutral_bias":
                    result[
                        "neutral_bias"
                    ],

                "mean_magnitude":
                    result[
                        "mean_magnitude"
                    ],
            }
        )


results = pd.DataFrame(
    rows
)


results = (
    results
    .sort_values(
        "score",
        ascending=False,
    )
    .reset_index(drop=True)
)


# =========================================================
# Best mapping
# =========================================================

best = results.iloc[0]


BEST_DIRECTION = int(
    best["direction"]
)

BEST_OFFSET = float(
    best["offset_deg"]
)


print()
print("=" * 110)
print("BEST INHIBITORY-DELTA7 CALIBRATION")
print("=" * 110)
print()

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
    f"{BEST_OFFSET:.1f}°"
)


# =========================================================
# Held-out validation
# =========================================================

validation_rows = []


for error in VALIDATION_ERRORS:

    steering = simulate(
        heading_error_deg=error,
        fc2_direction=BEST_DIRECTION,
        fc2_offset_deg=BEST_OFFSET,
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
                target_steering(
                    error
                ),

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
print("=" * 110)
print("HELD-OUT VALIDATION")
print("=" * 110)
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


mean_magnitude = float(
    validation[
        "steering"
    ]
    .abs()
    .mean()
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

print(
    f"Mean steering magnitude: "
    f"{mean_magnitude:.4f}"
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