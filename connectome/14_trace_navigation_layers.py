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

DELTA7_NEURON_FILE = (
    OUTPUT_DIR
    / "delta7_candidates.csv"
)

DELTA7_EDGE_FILE = (
    OUTPUT_DIR
    / "delta7_bridge_edges.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "navigation_layer_activity.csv"
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
# Combine neurons
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
# Combine edges
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

    # direct EPG -> PFL
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

    # PFL -> DNa
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


print("=" * 100)
print("NAVIGATION LAYER ACTIVITY TRACE")
print("=" * 100)

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)


# =========================================================
# Lookup
# =========================================================

body_to_index = {
    int(body_id): index
    for index, body_id
    in enumerate(
        neurons["bodyId"]
    )
}


# =========================================================
# EPG parser
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

    return f"{side}{number}"


# =========================================================
# FC2 parser
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
# Build EPG groups
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
# Build FC2 groups
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
# Validate groups
# =========================================================

epg_total = sum(
    len(x)
    for x in epg_groups.values()
)


fc2_total = sum(
    len(x)
    for x in fc2_groups.values()
)


print()

print(
    f"Grouped EPG neurons: "
    f"{epg_total}"
)

print(
    f"Grouped FC2 neurons: "
    f"{fc2_total}"
)


if epg_total != 46:

    raise RuntimeError(
        f"Expected 46 EPG neurons, "
        f"got {epg_total}"
    )


if fc2_total != 92:

    raise RuntimeError(
        f"Expected 92 FC2 neurons, "
        f"got {fc2_total}"
    )


# =========================================================
# Separate inhibitory Delta7 edges
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


# =========================================================
# Convert edges to Brian2 indices
# =========================================================

def prepare_edges(df):

    source_indices = []
    target_indices = []
    raw_weights = []


    for _, row in df.iterrows():

        source_body = int(
            row["source_bodyId"]
        )

        target_body = int(
            row["target_bodyId"]
        )

        if (
            source_body not in body_to_index
            or
            target_body not in body_to_index
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
# Common anatomical weight scaling
# =========================================================

all_raw = np.concatenate(
    [
        positive_raw,
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


positive_weights = scale_weights(
    positive_raw
)


delta_weights = scale_weights(
    delta_raw
)


# =========================================================
# Type index lookup
# =========================================================

TYPE_NAMES = [
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


type_indices = {}


for neuron_type in TYPE_NAMES:

    type_indices[
        neuron_type
    ] = np.asarray(
        [
            index
            for index, row
            in neurons.iterrows()
            if row["type"]
            == neuron_type
        ],
        dtype=int,
    )


# =========================================================
# Side indices for important populations
# =========================================================

def indices_for_type_side(
    neuron_type,
    side,
):

    return np.asarray(
        [
            index
            for index, row
            in neurons.iterrows()
            if (
                row["type"]
                == neuron_type
                and
                row.get(
                    "somaSide",
                    None,
                )
                == side
            )
        ],
        dtype=int,
    )


pfl2_left = indices_for_type_side(
    "PFL2",
    "L",
)

pfl2_right = indices_for_type_side(
    "PFL2",
    "R",
)

pfl3_left = indices_for_type_side(
    "PFL3",
    "L",
)

pfl3_right = indices_for_type_side(
    "PFL3",
    "R",
)

dna03_left = indices_for_type_side(
    "DNa03",
    "L",
)

dna03_right = indices_for_type_side(
    "DNa03",
    "R",
)

dna02_left = indices_for_type_side(
    "DNa02",
    "L",
)

dna02_right = indices_for_type_side(
    "DNa02",
    "R",
)


# =========================================================
# Utility
# =========================================================

def safe_sum(
    values,
    indices,
):

    if len(indices) == 0:
        return 0

    return int(
        values[
            indices
        ].sum()
    )


def safe_mean(
    values,
    indices,
):

    if len(indices) == 0:
        return 0.0

    return float(
        values[
            indices
        ].mean()
    )


# =========================================================
# Run one diagnostic condition
# =========================================================

def run_condition(
    heading_column,
    goal_column,
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
    # Same drives as previous experiment.
    #
    # Do NOT tune them here.
    # -----------------------------------------------------

    EPG_DRIVE = 1.35
    FC2_DRIVE = 1.35


    for index in epg_groups[
        heading_column
    ]:

        G.drive[
            index
        ] = EPG_DRIVE


    for index in fc2_groups[
        goal_column
    ]:

        G.drive[
            index
        ] = FC2_DRIVE


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
        300 * ms
    )


    counts = np.asarray(
        monitor.count
    )


    voltage = np.asarray(
        G.v
    )


    result = {
        "heading_column":
            heading_column,

        "goal_column":
            goal_column,
    }


    # -----------------------------------------------------
    # Activity by neuron population
    # -----------------------------------------------------

    for neuron_type in TYPE_NAMES:

        indices = type_indices[
            neuron_type
        ]

        result[
            f"{neuron_type}_spikes"
        ] = safe_sum(
            counts,
            indices,
        )

        result[
            f"{neuron_type}_spiking_neurons"
        ] = int(
            (
                counts[
                    indices
                ] > 0
            ).sum()
        )

        result[
            f"{neuron_type}_mean_v"
        ] = safe_mean(
            voltage,
            indices,
        )


    # -----------------------------------------------------
    # Left/right PFL detail
    # -----------------------------------------------------

    result[
        "PFL2_L_spikes"
    ] = safe_sum(
        counts,
        pfl2_left,
    )

    result[
        "PFL2_R_spikes"
    ] = safe_sum(
        counts,
        pfl2_right,
    )

    result[
        "PFL3_L_spikes"
    ] = safe_sum(
        counts,
        pfl3_left,
    )

    result[
        "PFL3_R_spikes"
    ] = safe_sum(
        counts,
        pfl3_right,
    )


    # -----------------------------------------------------
    # Descending neurons
    # -----------------------------------------------------

    result[
        "DNa03_L_spikes"
    ] = safe_sum(
        counts,
        dna03_left,
    )

    result[
        "DNa03_R_spikes"
    ] = safe_sum(
        counts,
        dna03_right,
    )

    result[
        "DNa02_L_spikes"
    ] = safe_sum(
        counts,
        dna02_left,
    )

    result[
        "DNa02_R_spikes"
    ] = safe_sum(
        counts,
        dna02_right,
    )


    result[
        "DNa02_L_v"
    ] = safe_mean(
        voltage,
        dna02_left,
    )

    result[
        "DNa02_R_v"
    ] = safe_mean(
        voltage,
        dna02_right,
    )


    result[
        "DNa02_difference"
    ] = (
        result[
            "DNa02_R_v"
        ]
        -
        result[
            "DNa02_L_v"
        ]
    )


    # -----------------------------------------------------
    # Strongest Delta7 neurons
    # -----------------------------------------------------

    delta_indices = (
        type_indices[
            "Delta7"
        ]
    )


    delta_counts = counts[
        delta_indices
    ]


    result[
        "Delta7_max_spikes_single_neuron"
    ] = (
        int(
            delta_counts.max()
        )
        if len(delta_counts)
        else 0
    )


    return result


# =========================================================
# Representative conditions
#
# We do NOT run 144 conditions.
#
# Three goal columns:
#
# C1 = strongly R-biased anatomical FC2 input
# C5 = transition/balanced region
# C9 = strongly L-biased anatomical FC2 input
#
# Four headings around the PB ring.
# =========================================================

TEST_HEADINGS = [
    "L1",
    "L5",
    "R5",
    "R1",
]


TEST_GOALS = [
    1,
    5,
    9,
]


rows = []


print()
print("=" * 100)
print("RUNNING REPRESENTATIVE CONDITIONS")
print("=" * 100)
print()


for heading in TEST_HEADINGS:

    for goal in TEST_GOALS:

        print(
            f"heading={heading:>2} | "
            f"goal=C{goal}"
        )


        result = run_condition(
            heading_column=heading,
            goal_column=goal,
        )


        rows.append(
            result
        )


# =========================================================
# Results
# =========================================================

results = pd.DataFrame(
    rows
)


results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# Layer propagation table
# =========================================================

layer_columns = [
    "heading_column",
    "goal_column",

    "EPG_spikes",
    "EPG_spiking_neurons",

    "Delta7_spikes",
    "Delta7_spiking_neurons",
    "Delta7_max_spikes_single_neuron",

    "FC2A_spikes",
    "FC2B_spikes",
    "FC2C_spikes",

    "PFL2_spikes",
    "PFL2_spiking_neurons",

    "PFL3_spikes",
    "PFL3_spiking_neurons",

    "DNa03_spikes",
    "DNa03_spiking_neurons",

    "DNa02_spikes",
    "DNa02_spiking_neurons",
]


print()
print("=" * 150)
print("LAYER PROPAGATION")
print("=" * 150)
print()


print(
    results[
        layer_columns
    ].to_string(
        index=False
    )
)


# =========================================================
# PFL left/right activity
# =========================================================

print()
print("=" * 120)
print("PFL LEFT / RIGHT ACTIVITY")
print("=" * 120)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",

            "PFL2_L_spikes",
            "PFL2_R_spikes",

            "PFL3_L_spikes",
            "PFL3_R_spikes",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# DNa activity
# =========================================================

print()
print("=" * 120)
print("DESCENDING OUTPUT")
print("=" * 120)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",

            "DNa03_L_spikes",
            "DNa03_R_spikes",

            "DNa02_L_spikes",
            "DNa02_R_spikes",

            "DNa02_L_v",
            "DNa02_R_v",

            "DNa02_difference",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.5f}",
    )
)


# =========================================================
# Mean membrane state through layers
# =========================================================

print()
print("=" * 140)
print("MEAN FINAL MEMBRANE STATE")
print("=" * 140)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",

            "EPG_mean_v",
            "Delta7_mean_v",

            "FC2A_mean_v",
            "FC2B_mean_v",
            "FC2C_mean_v",

            "PFL2_mean_v",
            "PFL3_mean_v",

            "DNa03_mean_v",
            "DNa02_mean_v",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.5f}",
    )
)


# =========================================================
# Save
# =========================================================

print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_FILE
)