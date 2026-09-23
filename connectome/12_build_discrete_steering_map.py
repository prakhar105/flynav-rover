import re
from pathlib import Path

import numpy as np
import pandas as pd

from brian2 import (
    Network,
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
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

DELTA7_NEURON_FILE = (
    OUTPUT_DIR
    / "delta7_candidates.csv"
)

DELTA7_EDGE_FILE = (
    OUTPUT_DIR
    / "delta7_bridge_edges.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "discrete_steering_map.csv"
)

MATRIX_FILE = (
    OUTPUT_DIR
    / "discrete_steering_matrix.csv"
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
    DELTA7_NEURON_FILE
)

delta7_edges = pd.read_csv(
    DELTA7_EDGE_FILE
)


# =========================================================
# Normalize neuron IDs
# =========================================================

for df in [
    steering_neurons,
    delta7_neurons,
]:
    df["bodyId"] = (
        df["bodyId"]
        .astype("int64")
    )


# =========================================================
# Normalize edge IDs / weights
# =========================================================

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


# =========================================================
# Remove duplicate biological edges
#
# The same edge can appear in both source CSV files.
# Keep one copy only.
# =========================================================

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
# Functional navigation circuit
#
# Heading:
#
# EPG
#  ↓
# Delta7
#  ↓
# PFL
#
# plus direct:
#
# EPG -> PFL
#
#
# Goal:
#
# FC2 -> PFL
#
#
# Steering:
#
# PFL -> DNa03 / DNa02
# DNa03 -> DNa02
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
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )

    |

    # Delta7 -> PFL
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
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
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )

    |

    # PFL -> descending neurons
    (
        edges["source_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
        &
        edges["target_type"]
        .isin(
            [
                "DNa03",
                "DNa02",
            ]
        )
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


# =========================================================
# Basic network information
# =========================================================

print("=" * 100)
print("DISCRETE MALECNS STEERING MAP")
print("=" * 100)

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
# Body ID -> Brian2 index
# =========================================================

body_to_index = {
    int(body_id): index
    for index, body_id
    in enumerate(
        neurons["bodyId"]
    )
}


# =========================================================
# Parse EPG anatomical PB column
#
# Examples:
#
# EPG(PB08)_L1
# EPG(PB08)_L2
# EPG(PB08)_R1
# ...
#
# No world angle is assigned here.
# =========================================================

def parse_epg_column(instance):

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

    return (
        f"{side}{number}"
    )


# =========================================================
# Parse FC2 anatomical column
#
# IMPORTANT FIX:
#
# The OLD regex:
#
#     r"C(\d+)"
#
# incorrectly matched the C2 inside:
#
#     FC2A
#     FC2B
#     FC2C
#
# Therefore ALL 92 FC2 neurons were incorrectly assigned
# to anatomical column C2.
#
#
# We now require the explicit:
#
#     _C1_
#     _C2_
#     ...
#     _C9_
#
# part of the INSTANCE name.
#
# The final column may also appear at the end of a string,
# so $ is allowed after the digit.
# =========================================================

def parse_fc2_column(instance):

    if not isinstance(
        instance,
        str,
    ):
        return None

    match = re.search(
        r"_C(\d+)(?:_|$)",
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

    return column


# =========================================================
# Group EPG neurons by actual anatomical PB column
# =========================================================

epg_groups = {}


for index, row in neurons.iterrows():

    if row["type"] != "EPG":
        continue

    column = parse_epg_column(
        row["instance"]
    )

    if column is None:
        continue

    epg_groups.setdefault(
        column,
        [],
    ).append(
        index
    )


# =========================================================
# Anatomical EPG ring order
#
# This is only an ordering of PB labels.
#
# We are NOT yet claiming that any specific column maps
# onto a specific world angle.
# =========================================================

EPG_ORDER = [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "R8",
    "R7",
    "R6",
    "R5",
    "R4",
    "R3",
    "R2",
    "R1",
]


# =========================================================
# Group FC2 neurons by anatomical C column
#
# FC2A / FC2B / FC2C belonging to the same C-column
# are grouped together.
# =========================================================

fc2_groups = {
    column: []
    for column in range(
        1,
        10,
    )
}


for index, row in neurons.iterrows():

    if row["type"] not in [
        "FC2A",
        "FC2B",
        "FC2C",
    ]:
        continue

    column = parse_fc2_column(
        row["instance"]
    )

    if column is None:
        continue

    fc2_groups[
        column
    ].append(
        index
    )


# =========================================================
# Validate EPG parsing
# =========================================================

expected_epg_count = int(
    (
        neurons["type"]
        == "EPG"
    ).sum()
)


grouped_epg_count = sum(
    len(indices)
    for indices in epg_groups.values()
)


print()
print("=" * 100)
print("EPG COLUMN COUNTS")
print("=" * 100)


for column in EPG_ORDER:

    print(
        f"{column:>3} : "
        f"{len(epg_groups.get(column, []))}"
    )


print()

print(
    f"Expected EPG neurons: "
    f"{expected_epg_count}"
)

print(
    f"Grouped EPG neurons: "
    f"{grouped_epg_count}"
)


if (
    grouped_epg_count
    != expected_epg_count
):

    raise RuntimeError(
        "EPG parsing failed: "
        f"expected {expected_epg_count} neurons "
        f"but grouped {grouped_epg_count}."
    )


# =========================================================
# Validate FC2 parsing
# =========================================================

expected_fc2_count = int(
    neurons["type"]
    .isin(
        [
            "FC2A",
            "FC2B",
            "FC2C",
        ]
    )
    .sum()
)


grouped_fc2_count = sum(
    len(indices)
    for indices in fc2_groups.values()
)


nonempty_fc2_columns = [
    column
    for column, indices
    in fc2_groups.items()
    if len(indices) > 0
]


print()
print("=" * 100)
print("FC2 COLUMN COUNTS")
print("=" * 100)


for column in range(
    1,
    10,
):

    print(
        f"C{column}: "
        f"{len(fc2_groups[column])}"
    )


print()

print(
    f"Expected FC2 neurons: "
    f"{expected_fc2_count}"
)

print(
    f"Grouped FC2 neurons: "
    f"{grouped_fc2_count}"
)

print(
    f"Non-empty FC2 columns: "
    f"{nonempty_fc2_columns}"
)


# =========================================================
# Critical parser safeguards
# =========================================================

if (
    grouped_fc2_count
    != expected_fc2_count
):

    raise RuntimeError(
        "FC2 parsing failed: "
        f"expected {expected_fc2_count} neurons "
        f"but grouped {grouped_fc2_count}."
    )


if (
    len(nonempty_fc2_columns)
    <= 1
):

    raise RuntimeError(
        "FC2 parsing failed: "
        "all FC2 neurons collapsed into one "
        "anatomical column."
    )


# =========================================================
# DNa02 output indices
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


print()

print(
    "DNa02 left indices:",
    dna_left,
)

print(
    "DNa02 right indices:",
    dna_right,
)


if not dna_left:

    raise RuntimeError(
        "No left DNa02 neuron found."
    )


if not dna_right:

    raise RuntimeError(
        "No right DNa02 neuron found."
    )


# =========================================================
# Separate Delta7 -> PFL edges
#
# Current hypothesis:
#
# Delta7 -> PFL is inhibitory.
#
# All other selected edges remain excitatory/default
# in this simplified Brian2 model.
# =========================================================

delta_edges = edges[
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )
].copy()


positive_edges = edges[
    ~(
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )
].copy()


print()

print(
    "Positive/default edges:",
    len(positive_edges),
)

print(
    "Delta7 inhibitory edges:",
    len(delta_edges),
)


# =========================================================
# Convert anatomical edges into Brian2 indices
# =========================================================

def prepare_edges(df):

    sources = []
    targets = []
    raw_weights = []


    for _, row in df.iterrows():

        source_body = int(
            row["source_bodyId"]
        )

        target_body = int(
            row["target_bodyId"]
        )

        if (
            source_body
            not in body_to_index
            or
            target_body
            not in body_to_index
        ):
            continue


        sources.append(
            body_to_index[
                source_body
            ]
        )

        targets.append(
            body_to_index[
                target_body
            ]
        )

        raw_weights.append(
            float(
                row["weight"]
            )
        )


    return (
        sources,
        targets,
        np.asarray(
            raw_weights,
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


# =========================================================
# Safety
# =========================================================

if len(pos_raw) == 0:

    raise RuntimeError(
        "No positive/default edges available."
    )


if len(delta_raw) == 0:

    raise RuntimeError(
        "No Delta7 -> PFL edges available."
    )


# =========================================================
# Common anatomical weight scaling
#
# IMPORTANT:
#
# Both positive and Delta7 populations use the same
# normalization reference.
#
# This preserves relative anatomical strength better
# than scaling each population separately.
# =========================================================

all_raw = np.concatenate(
    [
        pos_raw,
        delta_raw,
    ]
)


MAX_RAW = max(
    float(
        all_raw.max()
    ),
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


pos_weights = scale_weights(
    pos_raw
)


delta_weights = scale_weights(
    delta_raw
)


# =========================================================
# Run one anatomical condition
#
# heading_column:
#     actual EPG PB anatomical column
#
# goal_column:
#     actual FC2 C anatomical column
#
#
# IMPORTANT:
#
# There is NO assumed angle relationship between EPG
# and FC2 here.
#
# We are directly probing the connectome.
# =========================================================

def simulate_condition(
    heading_column,
    goal_column,
):

    N = len(
        neurons
    )


    # -----------------------------------------------------
    # LIF surrogate
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
    # Activate ONE anatomical EPG heading column
    # -----------------------------------------------------

    EPG_DRIVE = 1.35


    heading_indices = (
        epg_groups.get(
            heading_column,
            [],
        )
    )


    for index in heading_indices:

        G.drive[index] = (
            EPG_DRIVE
        )


    # -----------------------------------------------------
    # Activate ONE anatomical FC2 goal column
    # -----------------------------------------------------

    FC2_DRIVE = 1.35


    goal_indices = (
        fc2_groups[
            goal_column
        ]
    )


    for index in goal_indices:

        G.drive[index] = (
            FC2_DRIVE
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
        i=pos_source,
        j=pos_target,
    )


    S_positive.w = (
        pos_weights
    )


    # -----------------------------------------------------
    # Delta7 -> PFL inhibitory synapses
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
    # Record output
    # -----------------------------------------------------

    spike_monitor = SpikeMonitor(
        G
    )


    state_monitor = StateMonitor(
        G,
        "v",
        record=True,
        dt=5 * ms,
    )


    network = Network(
        G,
        S_positive,
        S_delta,
        spike_monitor,
        state_monitor,
    )


    network.run(
        300 * ms
    )


    # -----------------------------------------------------
    # DNa02 spike counts
    # -----------------------------------------------------

    counts = np.asarray(
        spike_monitor.count
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
    # DNa02 membrane activity
    #
    # Average final 50 ms instead of reading one
    # potentially noisy final timestep.
    #
    # StateMonitor dt = 5 ms
    # 10 samples = 50 ms
    # -----------------------------------------------------

    voltage = np.asarray(
        state_monitor.v
    )


    samples_to_average = 10


    left_v = float(
        voltage[
            dna_left,
            -samples_to_average:
        ].mean()
    )


    right_v = float(
        voltage[
            dna_right,
            -samples_to_average:
        ].mean()
    )


    membrane_difference = (
        right_v
        - left_v
    )


    # -----------------------------------------------------
    # Steering interpretation
    #
    # Positive:
    #     DNa02_R dominance
    #     RIGHT steering
    #
    # Negative:
    #     DNa02_L dominance
    #     LEFT steering
    # -----------------------------------------------------

    spike_total = (
        left_spikes
        +
        right_spikes
    )


    if spike_total > 0:

        spike_difference = (
            right_spikes
            - left_spikes
        ) / spike_total

    else:

        spike_difference = 0.0


    # -----------------------------------------------------
    # Output fusion
    #
    # When spikes exist:
    # mostly spike signal
    #
    # When no spikes exist:
    # use subthreshold membrane asymmetry
    # -----------------------------------------------------

    if spike_total > 0:

        steering = (
            0.80
            * spike_difference
            +
            0.20
            * membrane_difference
        )

    else:

        steering = (
            membrane_difference
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

        "membrane_difference":
            membrane_difference,

        "steering":
            float(steering),
    }


# =========================================================
# Build full anatomical steering map
# =========================================================

rows = []


total_conditions = (
    len(EPG_ORDER)
    * len(fc2_groups)
)


condition_number = 0


print()
print("=" * 100)
print("BUILDING CONNECTOME STEERING MAP")
print("=" * 100)
print()


for heading_column in EPG_ORDER:

    if (
        heading_column
        not in epg_groups
    ):

        print(
            f"WARNING: "
            f"No EPG neurons for "
            f"{heading_column}"
        )

        continue


    for goal_column in range(
        1,
        10,
    ):

        condition_number += 1


        print(
            f"[{condition_number:3d}/"
            f"{total_conditions}] "
            f"heading={heading_column:>2} | "
            f"goal=C{goal_column} | "
            f"FC2 neurons="
            f"{len(fc2_groups[goal_column])}"
        )


        result = simulate_condition(
            heading_column=heading_column,
            goal_column=goal_column,
        )


        rows.append(
            {
                "heading_column":
                    heading_column,

                "goal_column":
                    goal_column,

                "epg_neuron_count":
                    len(
                        epg_groups[
                            heading_column
                        ]
                    ),

                "fc2_neuron_count":
                    len(
                        fc2_groups[
                            goal_column
                        ]
                    ),

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

                "membrane_difference":
                    result[
                        "membrane_difference"
                    ],

                "steering":
                    result[
                        "steering"
                    ],
            }
        )


# =========================================================
# Results DataFrame
# =========================================================

results = pd.DataFrame(
    rows
)


results.to_csv(
    RESULT_FILE,
    index=False,
)


# =========================================================
# Numeric steering matrix
# =========================================================

matrix = (
    results
    .pivot(
        index="heading_column",
        columns="goal_column",
        values="steering",
    )
    .reindex(
        EPG_ORDER
    )
)


matrix.columns = [
    f"C{x}"
    for x in matrix.columns
]


matrix.to_csv(
    MATRIX_FILE
)


print()
print("=" * 120)
print("RAW CONNECTOME STEERING MATRIX")
print("=" * 120)
print()


print(
    matrix.to_string(
        float_format=lambda x:
        f"{x:+.4f}"
    )
)


# =========================================================
# Direction-only view
#
# R = DNa02_R dominance
# L = DNa02_L dominance
# . = near-neutral
# =========================================================

NEUTRAL_THRESHOLD = 0.002


def steering_symbol(value):

    if value > NEUTRAL_THRESHOLD:

        return "R"

    if value < -NEUTRAL_THRESHOLD:

        return "L"

    return "."


# =========================================================
# Pandas-compatible direction conversion
# =========================================================

symbol_matrix = (
    matrix.copy()
)


for column in (
    symbol_matrix.columns
):

    symbol_matrix[
        column
    ] = (
        symbol_matrix[
            column
        ].map(
            steering_symbol
        )
    )


print()
print("=" * 120)
print("DIRECTION MAP")
print("=" * 120)
print()


print(
    symbol_matrix.to_string()
)


# =========================================================
# Signal magnitude summary
# =========================================================

print()
print("=" * 120)
print("SIGNAL SUMMARY")
print("=" * 120)
print()


absolute_values = (
    results[
        "steering"
    ].abs()
)


print(
    f"Minimum magnitude: "
    f"{absolute_values.min():.6f}"
)

print(
    f"Median magnitude: "
    f"{absolute_values.median():.6f}"
)

print(
    f"Mean magnitude: "
    f"{absolute_values.mean():.6f}"
)

print(
    f"Maximum magnitude: "
    f"{absolute_values.max():.6f}"
)


# =========================================================
# Positive / negative / neutral counts
# =========================================================

right_conditions = int(
    (
        results["steering"]
        > NEUTRAL_THRESHOLD
    ).sum()
)


left_conditions = int(
    (
        results["steering"]
        < -NEUTRAL_THRESHOLD
    ).sum()
)


neutral_conditions = int(
    len(results)
    -
    right_conditions
    -
    left_conditions
)


print()

print(
    f"RIGHT conditions: "
    f"{right_conditions}"
)

print(
    f"LEFT conditions: "
    f"{left_conditions}"
)

print(
    f"Neutral conditions: "
    f"{neutral_conditions}"
)


# =========================================================
# DNa02 spike summary
# =========================================================

total_left_spikes = int(
    results[
        "DNa02_L_spikes"
    ].sum()
)


total_right_spikes = int(
    results[
        "DNa02_R_spikes"
    ].sum()
)


conditions_with_spikes = int(
    (
        (
            results[
                "DNa02_L_spikes"
            ]
            +
            results[
                "DNa02_R_spikes"
            ]
        )
        > 0
    ).sum()
)


print()

print(
    f"Total DNa02_L spikes: "
    f"{total_left_spikes}"
)

print(
    f"Total DNa02_R spikes: "
    f"{total_right_spikes}"
)

print(
    f"Conditions with DNa02 spikes: "
    f"{conditions_with_spikes}"
    f"/{len(results)}"
)


# =========================================================
# Strongest RIGHT outputs
# =========================================================

print()
print("=" * 120)
print("STRONGEST RIGHT STEERING CONDITIONS")
print("=" * 120)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",
            "steering",
            "DNa02_L_spikes",
            "DNa02_R_spikes",
        ]
    ]
    .sort_values(
        "steering",
        ascending=False,
    )
    .head(15)
    .to_string(
        index=False
    )
)


# =========================================================
# Strongest LEFT outputs
# =========================================================

print()
print("=" * 120)
print("STRONGEST LEFT STEERING CONDITIONS")
print("=" * 120)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",
            "steering",
            "DNa02_L_spikes",
            "DNa02_R_spikes",
        ]
    ]
    .sort_values(
        "steering",
        ascending=True,
    )
    .head(15)
    .to_string(
        index=False
    )
)


# =========================================================
# Files
# =========================================================

print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)


print(
    RESULT_FILE
)


print(
    MATRIX_FILE
)