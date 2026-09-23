import re
from itertools import combinations
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

FROZEN_MAP_FILE = (
    OUTPUT_DIR
    / "frozen_fc2_population_map.csv"
)

STEP28_SUMMARY_FILE = (
    OUTPUT_DIR
    / "continuous_fc2_blending_summary.csv"
)

STEP29_SUMMARY_FILE = (
    OUTPUT_DIR
    / "temporal_fc2_blending_summary.csv"
)

STEP30_SUMMARY_FILE = (
    OUTPUT_DIR
    / "fc2_population_recruitment_summary.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment_summary.csv"
)

SUBSET_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_subsets.csv"
)


# =========================================================
# Frozen surrogate parameters
#
# DO NOT tune these in Step 31.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0

PFL_TO_DNA_GAIN = 3.0


# =========================================================
# Frozen DNa decoder
# =========================================================

SPIKE_SCALE = 5.0

ACTIVITY_SCALE = 1.10599


# =========================================================
# Frozen neural input
# =========================================================

BASELINE_DRIVE = 0.03

EPG_DRIVE = 1.35

FC2_DRIVE = 1.35

SIMULATION_TIME = 300 * ms


# =========================================================
# Experiment
# =========================================================

BLEND_VALUES = np.linspace(
    0.0,
    1.0,
    11,
)


REPRESENTATIVE_HEADINGS = [
    "L1",
    "L5",
    "R5",
    "R1",
]


FC2_PAIRS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 1),
]


# =========================================================
# Load connectome-derived data
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
# Normalize IDs and weights
# =========================================================

for dataframe in [
    steering_neurons,
    delta7_neurons,
]:

    dataframe["bodyId"] = (
        dataframe["bodyId"]
        .astype("int64")
    )


for dataframe in [
    steering_edges,
    delta7_edges,
]:

    dataframe["source_bodyId"] = (
        dataframe["source_bodyId"]
        .astype("int64")
    )

    dataframe["target_bodyId"] = (
        dataframe["target_bodyId"]
        .astype("int64")
    )

    dataframe["weight"] = pd.to_numeric(
        dataframe["weight"],
        errors="coerce",
    ).fillna(0.0)


# =========================================================
# Network populations
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
        subset=[
            "bodyId",
        ]
    )
)


neurons = (
    neurons[
        neurons["type"]
        .isin(
            NETWORK_TYPES
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


body_ids = set(
    neurons[
        "bodyId"
    ]
)


# =========================================================
# Combine anatomical edge tables
# =========================================================

edges = pd.concat(
    [
        steering_edges,
        delta7_edges,
    ],
    ignore_index=True,
)


edges = edges[
    edges[
        "source_bodyId"
    ].isin(
        body_ids
    )
    &
    edges[
        "target_bodyId"
    ].isin(
        body_ids
    )
].copy()


# =========================================================
# Same duplicate handling as frozen Step 27
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
# Frozen functional topology
# =========================================================

functional_mask = (

    (
        (edges["source_type"] == "EPG")
        &
        (
            (edges["target_type"] == "Delta7")

            |

            edges["target_type"].isin(
                [
                    "PFL2",
                    "PFL3",
                ]
            )
        )
    )

    |

    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"].isin(
            [
                "PFL2",
                "PFL3",
            ]
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
            [
                "PFL2",
                "PFL3",
            ]
        )
    )

    |

    (
        edges["source_type"].isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
        &
        edges["target_type"].isin(
            [
                "DNa02",
                "DNa03",
            ]
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
        functional_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# =========================================================
# Frozen topology validation
# =========================================================

if len(
    neurons
) != 220:

    raise RuntimeError(
        f"Expected 220 neurons, "
        f"got {len(neurons)}."
    )


if len(
    edges
) != 3231:

    raise RuntimeError(
        f"Expected 3231 functional edges, "
        f"got {len(edges)}."
    )


print("=" * 120)
print(
    "ANATOMICALLY BALANCED FC2 POPULATION RECRUITMENT"
)
print("=" * 120)

print()

print(
    f"Neurons: "
    f"{len(neurons)}"
)

print(
    f"Functional edges: "
    f"{len(edges)}"
)

print(
    f"EPG -> Delta7 gain: "
    f"{EPG_TO_DELTA7_GAIN:.1f}x"
)

print(
    f"PFL -> DNa gain: "
    f"{PFL_TO_DNA_GAIN:.1f}x"
)

print(
    f"FC2 active drive: "
    f"{FC2_DRIVE:.2f}"
)

print(
    f"Simulation time: "
    f"{float(SIMULATION_TIME / ms):.1f} ms"
)


# =========================================================
# Body ID -> Brian2 network index
# =========================================================

body_to_index = {

    int(body_id):
        index

    for index, body_id
    in enumerate(
        neurons[
            "bodyId"
        ]
    )
}


# =========================================================
# EPG / FC2 parsers
# =========================================================

def parse_epg_column(
    instance,
):

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


    side = match.group(
        1
    )

    number = int(
        match.group(
            2
        )
    )


    if not (
        1 <= number <= 8
    ):

        return None


    return (
        f"{side}{number}"
    )


def parse_fc2_column(
    instance,
):

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
        match.group(
            1
        )
    )


    if not (
        1 <= column <= 9
    ):

        return None


    return column


# =========================================================
# EPG anatomical groups
# =========================================================

epg_groups = {}


for index, row in neurons.iterrows():

    if row[
        "type"
    ] != "EPG":

        continue


    column = parse_epg_column(
        row[
            "instance"
        ]
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
# FC2 anatomical groups
#
# Store real network indices.
#
# Ordering will NOT be used as biological interpolation.
# =========================================================

fc2_groups = {

    column: []

    for column
    in range(
        1,
        10,
    )
}


for index, row in neurons.iterrows():

    if row[
        "type"
    ] not in [
        "FC2A",
        "FC2B",
        "FC2C",
    ]:

        continue


    column = parse_fc2_column(
        row[
            "instance"
        ]
    )


    if column is None:

        continue


    fc2_groups[
        column
    ].append(
        index
    )


# =========================================================
# Validate EPG / FC2 populations
# =========================================================

epg_count = sum(

    len(
        indices
    )

    for indices
    in epg_groups.values()
)


fc2_count = sum(

    len(
        indices
    )

    for indices
    in fc2_groups.values()
)


if epg_count != 46:

    raise RuntimeError(
        f"Expected 46 EPG neurons, "
        f"got {epg_count}."
    )


if fc2_count != 92:

    raise RuntimeError(
        f"Expected 92 FC2 neurons, "
        f"got {fc2_count}."
    )


print()

print(
    f"EPG neurons grouped: "
    f"{epg_count}"
)

print(
    f"FC2 neurons grouped: "
    f"{fc2_count}"
)


print()

print(
    "FC2 column sizes:"
)


for column in range(
    1,
    10,
):

    print(
        f"C{column}: "
        f"{len(fc2_groups[column])}"
    )


# =========================================================
# Generic population lookup
# =========================================================

def get_indices(
    neuron_type,
    side=None,
):

    output = []


    for index, row in neurons.iterrows():

        if row[
            "type"
        ] != neuron_type:

            continue


        if side is not None:

            if row.get(
                "somaSide",
                None,
            ) != side:

                continue


        output.append(
            index
        )


    return np.asarray(
        output,
        dtype=int,
    )


delta7_indices = get_indices(
    "Delta7"
)

pfl2_indices = get_indices(
    "PFL2"
)

pfl3_indices = get_indices(
    "PFL3"
)

dna03_indices = get_indices(
    "DNa03"
)

dna02_indices = get_indices(
    "DNa02"
)


dna03_left = get_indices(
    "DNa03",
    "L",
)

dna03_right = get_indices(
    "DNa03",
    "R",
)


dna02_left = get_indices(
    "DNa02",
    "L",
)

dna02_right = get_indices(
    "DNa02",
    "R",
)


# =========================================================
# Fresh pathway masks
#
# These are made AFTER reset_index().
# =========================================================

mask_epg_delta = (
    (edges["source_type"] == "EPG")
    &
    (edges["target_type"] == "Delta7")
)


mask_delta_pfl = (
    (edges["source_type"] == "Delta7")
    &
    edges["target_type"].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
)


mask_pfl3_dna02 = (
    (edges["source_type"] == "PFL3")
    &
    (edges["target_type"] == "DNa02")
)


mask_pfl3_dna03 = (
    (edges["source_type"] == "PFL3")
    &
    (edges["target_type"] == "DNa03")
)


mask_pfl2_dna03 = (
    (edges["source_type"] == "PFL2")
    &
    (edges["target_type"] == "DNa03")
)


mask_pfl2_dna02 = (
    (edges["source_type"] == "PFL2")
    &
    (edges["target_type"] == "DNa02")
)


mask_dna03_dna02 = (
    (edges["source_type"] == "DNa03")
    &
    (edges["target_type"] == "DNa02")
)


mask_other_upstream = (

    (
        (edges["source_type"] == "EPG")
        &
        edges["target_type"].isin(
            [
                "PFL2",
                "PFL3",
            ]
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
            [
                "PFL2",
                "PFL3",
            ]
        )
    )
)


# =========================================================
# Split functional pathways
# =========================================================

epg_delta_edges = edges.loc[
    mask_epg_delta
].copy()


delta_pfl_edges = edges.loc[
    mask_delta_pfl
].copy()


pfl3_dna02_edges = edges.loc[
    mask_pfl3_dna02
].copy()


pfl3_dna03_edges = edges.loc[
    mask_pfl3_dna03
].copy()


pfl2_dna03_edges = edges.loc[
    mask_pfl2_dna03
].copy()


pfl2_dna02_edges = edges.loc[
    mask_pfl2_dna02
].copy()


dna03_dna02_edges = edges.loc[
    mask_dna03_dna02
].copy()


other_edges = edges.loc[
    mask_other_upstream
].copy()


# =========================================================
# Exact edge accounting
# =========================================================

accounted_edges = (

    len(
        epg_delta_edges
    )

    +

    len(
        delta_pfl_edges
    )

    +

    len(
        pfl3_dna02_edges
    )

    +

    len(
        pfl3_dna03_edges
    )

    +

    len(
        pfl2_dna03_edges
    )

    +

    len(
        pfl2_dna02_edges
    )

    +

    len(
        dna03_dna02_edges
    )

    +

    len(
        other_edges
    )
)


if accounted_edges != len(
    edges
):

    raise RuntimeError(
        "Functional edge accounting mismatch: "
        f"{accounted_edges} accounted, "
        f"{len(edges)} expected."
    )


print()

print("=" * 120)
print(
    "FROZEN PATHWAY INVENTORY"
)
print("=" * 120)

print(
    f"EPG -> Delta7: "
    f"{len(epg_delta_edges)}"
)

print(
    f"Delta7 -> PFL: "
    f"{len(delta_pfl_edges)}"
)

print(
    f"PFL3 -> DNa02: "
    f"{len(pfl3_dna02_edges)}"
)

print(
    f"PFL3 -> DNa03: "
    f"{len(pfl3_dna03_edges)}"
)

print(
    f"PFL2 -> DNa03: "
    f"{len(pfl2_dna03_edges)}"
)

print(
    f"PFL2 -> DNa02: "
    f"{len(pfl2_dna02_edges)}"
)

print(
    f"DNa03 -> DNa02: "
    f"{len(dna03_dna02_edges)}"
)

print(
    f"Other upstream: "
    f"{len(other_edges)}"
)


# =========================================================
# FC2 -> individual PFL anatomical projection vectors
#
# Important:
#
# We use RAW MaleCNS FC2 -> PFL weights here only to choose
# a representative subset of neurons.
#
# No actual synapse weights in the Brian2 circuit are
# modified.
# =========================================================

pfl_rows = neurons[
    neurons[
        "type"
    ].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
].copy()


pfl_body_ids = [

    int(
        body_id
    )

    for body_id
    in pfl_rows[
        "bodyId"
    ]
]


pfl_body_to_projection_index = {

    body_id:
        index

    for index, body_id
    in enumerate(
        pfl_body_ids
    )
}


N_PFL_TARGETS = len(
    pfl_body_ids
)


fc2_projection_vectors = {}


for column in range(
    1,
    10,
):

    for network_index in fc2_groups[
        column
    ]:

        body_id = int(
            neurons.loc[
                network_index,
                "bodyId",
            ]
        )


        vector = np.zeros(
            N_PFL_TARGETS,
            dtype=float,
        )


        outgoing = edges[
            (
                edges[
                    "source_bodyId"
                ]
                ==
                body_id
            )
            &
            edges[
                "target_type"
            ].isin(
                [
                    "PFL2",
                    "PFL3",
                ]
            )
        ]


        for _, edge_row in outgoing.iterrows():

            target_body_id = int(
                edge_row[
                    "target_bodyId"
                ]
            )


            if (
                target_body_id
                not in
                pfl_body_to_projection_index
            ):

                continue


            target_position = (
                pfl_body_to_projection_index[
                    target_body_id
                ]
            )


            vector[
                target_position
            ] += float(
                edge_row[
                    "weight"
                ]
            )


        fc2_projection_vectors[
            network_index
        ] = vector


# =========================================================
# Build anatomically representative subsets
#
# For every FC2 column and every possible subset size k:
#
# target projection =
#
#       (k / total_neurons)
#       *
#       full-column projection
#
# Exhaustively inspect all combinations of k neurons.
#
# Largest search is tiny here:
# C(11,5) = 462.
# =========================================================

balanced_subsets = {}

subset_rows = []


print()

print("=" * 120)
print(
    "BUILDING ANATOMICALLY BALANCED FC2 SUBSETS"
)
print("=" * 120)

print()


for column in range(
    1,
    10,
):

    network_indices = list(
        fc2_groups[
            column
        ]
    )


    # deterministic combination ordering
    network_indices = sorted(
        network_indices,
        key=lambda index:
            int(
                neurons.loc[
                    index,
                    "bodyId",
                ]
            ),
    )


    projection_matrix = np.stack(
        [
            fc2_projection_vectors[
                network_index
            ]

            for network_index
            in network_indices
        ]
    )


    full_projection = (
        projection_matrix.sum(
            axis=0
        )
    )


    n_total = len(
        network_indices
    )


    balanced_subsets[
        column
    ] = {}


    # =====================================================
    # k = 0
    # =====================================================

    balanced_subsets[
        column
    ][
        0
    ] = []


    subset_rows.append(
        {
            "column":
                column,

            "subset_size":
                0,

            "total_size":
                n_total,

            "relative_projection_error":
                0.0,

            "body_ids":
                "",
        }
    )


    # =====================================================
    # Intermediate subset sizes
    # =====================================================

    for k in range(
        1,
        n_total,
    ):

        target_projection = (

            float(
                k
            )
            /
            float(
                n_total
            )

        ) * full_projection


        # -------------------------------------------------
        # Normalize error to the full-column projection,
        # not the target projection.
        #
        # This makes errors comparable across different k.
        # -------------------------------------------------

        normalization = max(
            float(
                np.linalg.norm(
                    full_projection
                )
            ),
            1e-12,
        )


        best_combo = None

        best_error = np.inf


        for combo in combinations(
            range(
                n_total
            ),
            k,
        ):

            combo_positions = list(
                combo
            )


            subset_projection = (
                projection_matrix[
                    combo_positions
                ]
                .sum(
                    axis=0
                )
            )


            error = (

                float(
                    np.linalg.norm(
                        subset_projection
                        -
                        target_projection
                    )
                )
                /
                normalization
            )


            if error < best_error:

                best_error = error

                best_combo = combo


        selected_network_indices = [

            network_indices[
                position
            ]

            for position
            in best_combo
        ]


        balanced_subsets[
            column
        ][
            k
        ] = selected_network_indices


        selected_body_ids = [

            int(
                neurons.loc[
                    network_index,
                    "bodyId",
                ]
            )

            for network_index
            in selected_network_indices
        ]


        subset_rows.append(
            {
                "column":
                    column,

                "subset_size":
                    k,

                "total_size":
                    n_total,

                "relative_projection_error":
                    float(
                        best_error
                    ),

                "body_ids":
                    ",".join(
                        str(
                            body_id
                        )

                        for body_id
                        in selected_body_ids
                    ),
            }
        )


        print(
            f"C{column} "
            f"k={k:2d}/{n_total:2d} "
            f"| normalized projection error="
            f"{best_error:.4f}"
        )


    # =====================================================
    # k = N
    # =====================================================

    balanced_subsets[
        column
    ][
        n_total
    ] = list(
        network_indices
    )


    all_body_ids = [

        int(
            neurons.loc[
                network_index,
                "bodyId",
            ]
        )

        for network_index
        in network_indices
    ]


    subset_rows.append(
        {
            "column":
                column,

            "subset_size":
                n_total,

            "total_size":
                n_total,

            "relative_projection_error":
                0.0,

            "body_ids":
                ",".join(
                    str(
                        body_id
                    )

                    for body_id
                    in all_body_ids
                ),
        }
    )


subset_dataframe = pd.DataFrame(
    subset_rows
)


subset_dataframe.to_csv(
    SUBSET_FILE,
    index=False,
)


print()

print(
    "Mean intermediate subset projection error: "
    f"{subset_dataframe[
        (
            subset_dataframe['subset_size'] > 0
        )
        &
        (
            subset_dataframe['subset_size']
            <
            subset_dataframe['total_size']
        )
    ]['relative_projection_error'].mean():.4f}"
)


print(
    "Worst intermediate subset projection error: "
    f"{subset_dataframe[
        (
            subset_dataframe['subset_size'] > 0
        )
        &
        (
            subset_dataframe['subset_size']
            <
            subset_dataframe['total_size']
        )
    ]['relative_projection_error'].max():.4f}"
)


# =========================================================
# Prepare functional edges for Brian2
# =========================================================

def prepare_edges(
    dataframe,
):

    sources = []
    targets = []
    raw_weights = []


    for _, row in dataframe.iterrows():

        source_id = int(
            row[
                "source_bodyId"
            ]
        )

        target_id = int(
            row[
                "target_bodyId"
            ]
        )


        if (
            source_id
            not in body_to_index
        ):

            continue


        if (
            target_id
            not in body_to_index
        ):

            continue


        sources.append(
            body_to_index[
                source_id
            ]
        )


        targets.append(
            body_to_index[
                target_id
            ]
        )


        raw_weights.append(
            float(
                row[
                    "weight"
                ]
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
    epg_delta_source,
    epg_delta_target,
    epg_delta_raw,
) = prepare_edges(
    epg_delta_edges
)


(
    delta_pfl_source,
    delta_pfl_target,
    delta_pfl_raw,
) = prepare_edges(
    delta_pfl_edges
)


(
    pfl3_dna02_source,
    pfl3_dna02_target,
    pfl3_dna02_raw,
) = prepare_edges(
    pfl3_dna02_edges
)


(
    pfl3_dna03_source,
    pfl3_dna03_target,
    pfl3_dna03_raw,
) = prepare_edges(
    pfl3_dna03_edges
)


(
    pfl2_dna03_source,
    pfl2_dna03_target,
    pfl2_dna03_raw,
) = prepare_edges(
    pfl2_dna03_edges
)


(
    pfl2_dna02_source,
    pfl2_dna02_target,
    pfl2_dna02_raw,
) = prepare_edges(
    pfl2_dna02_edges
)


(
    dna03_dna02_source,
    dna03_dna02_target,
    dna03_dna02_raw,
) = prepare_edges(
    dna03_dna02_edges
)


(
    other_source,
    other_target,
    other_raw,
) = prepare_edges(
    other_edges
)


# =========================================================
# Frozen anatomical synapse scaling
# =========================================================

raw_arrays = [
    epg_delta_raw,
    delta_pfl_raw,
    pfl3_dna02_raw,
    pfl3_dna03_raw,
    pfl2_dna03_raw,
    pfl2_dna02_raw,
    dna03_dna02_raw,
    other_raw,
]


MAX_RAW = float(
    np.concatenate(
        [
            array

            for array
            in raw_arrays

            if len(
                array
            ) > 0
        ]
    ).max()
)


def scale_weights(
    raw_weights,
):

    return (
        0.02
        +
        0.30
        * np.sqrt(
            raw_weights
            /
            MAX_RAW
        )
    )


epg_delta_weights = scale_weights(
    epg_delta_raw
)

delta_pfl_weights = scale_weights(
    delta_pfl_raw
)

pfl3_dna02_weights = scale_weights(
    pfl3_dna02_raw
)

pfl3_dna03_weights = scale_weights(
    pfl3_dna03_raw
)

pfl2_dna03_weights = scale_weights(
    pfl2_dna03_raw
)

pfl2_dna02_weights = scale_weights(
    pfl2_dna02_raw
)

dna03_dna02_weights = scale_weights(
    dna03_dna02_raw
)

other_weights = scale_weights(
    other_raw
)


# =========================================================
# Helpers
# =========================================================

def safe_sum(
    values,
    indices,
):

    if len(
        indices
    ) == 0:

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

    if len(
        indices
    ) == 0:

        return 0.0


    return float(
        values[
            indices
        ].mean()
    )


def create_positive_synapses(
    group,
    source,
    target,
    weights,
    gain=1.0,
):

    if len(
        source
    ) == 0:

        return None


    synapses = Synapses(
        group,
        group,
        model="w : 1",
        on_pre="v_post += w",
    )


    synapses.connect(
        i=source,
        j=target,
    )


    synapses.w = (
        weights
        *
        gain
    )


    return synapses


# =========================================================
# Anatomically balanced recruitment
#
# Total recruited population is controlled.
#
# We do NOT independently round both populations like
# Step 30.
#
# Example:
#
# C2 has 11 neurons
# C3 has 11 neurons
#
# alpha=0.5:
#
# target total = 11
#
# not 6 + 6 = 12.
# =========================================================

def get_recruited_neurons(
    column_a,
    column_b,
    alpha,
):

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )


    n_a_total = len(
        fc2_groups[
            column_a
        ]
    )


    n_b_total = len(
        fc2_groups[
            column_b
        ]
    )


    # =====================================================
    # Interpolate desired total population size
    # =====================================================

    target_total = int(
        round(
            (
                1.0
                -
                alpha
            )
            *
            n_a_total

            +

            alpha
            *
            n_b_total
        )
    )


    target_total = int(
        np.clip(
            target_total,
            1,
            max(
                n_a_total,
                n_b_total,
            ),
        )
    )


    # =====================================================
    # Determine B share from alpha
    # =====================================================

    n_b = int(
        round(
            alpha
            *
            target_total
        )
    )


    n_b = int(
        np.clip(
            n_b,
            0,
            n_b_total,
        )
    )


    # =====================================================
    # A gets the remainder
    # =====================================================

    n_a = (
        target_total
        -
        n_b
    )


    # =====================================================
    # Correct any capacity overflow
    # =====================================================

    if n_a > n_a_total:

        overflow = (
            n_a
            -
            n_a_total
        )


        n_a = (
            n_a_total
        )


        n_b = min(
            n_b_total,
            n_b
            +
            overflow,
        )


    if n_b > n_b_total:

        overflow = (
            n_b
            -
            n_b_total
        )


        n_b = (
            n_b_total
        )


        n_a = min(
            n_a_total,
            n_a
            +
            overflow,
        )


    # =====================================================
    # Retrieve anatomically balanced subsets
    # =====================================================

    recruited_a = list(
        balanced_subsets[
            column_a
        ][
            n_a
        ]
    )


    recruited_b = list(
        balanced_subsets[
            column_b
        ][
            n_b
        ]
    )


    return (
        recruited_a,
        recruited_b,
        n_a,
        n_b,
        n_a_total,
        n_b_total,
        target_total,
    )


# =========================================================
# Run one neural condition
# =========================================================

def run_condition(
    heading_column,
    column_a,
    column_b,
    alpha,
):

    equations = """
    dv/dt = (-v + drive) / (20*ms) : 1
    drive : 1
    """


    group = NeuronGroup(
        len(
            neurons
        ),
        equations,
        threshold="v > 1.0",
        reset="v = 0.0",
        refractory=3 * ms,
        method="euler",
    )


    group.v = 0.0

    group.drive = (
        BASELINE_DRIVE
    )


    # =====================================================
    # EPG heading
    # =====================================================

    for index in epg_groups[
        heading_column
    ]:

        group.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # Anatomy-balanced FC2 recruitment
    # =====================================================

    (
        recruited_a,
        recruited_b,
        n_a,
        n_b,
        n_a_total,
        n_b_total,
        target_total,
    ) = get_recruited_neurons(
        column_a,
        column_b,
        alpha,
    )


    # =====================================================
    # Every recruited neuron receives FULL drive
    # =====================================================

    for index in recruited_a:

        group.drive[
            index
        ] = FC2_DRIVE


    for index in recruited_b:

        group.drive[
            index
        ] = FC2_DRIVE


    objects = [
        group
    ]


    # =====================================================
    # EPG / FC2 -> PFL
    # =====================================================

    synapses = create_positive_synapses(
        group,
        other_source,
        other_target,
        other_weights,
        gain=1.0,
    )


    if synapses is not None:

        objects.append(
            synapses
        )


    # =====================================================
    # EPG -> Delta7
    # =====================================================

    synapses = create_positive_synapses(
        group,
        epg_delta_source,
        epg_delta_target,
        epg_delta_weights,
        gain=EPG_TO_DELTA7_GAIN,
    )


    if synapses is not None:

        objects.append(
            synapses
        )


    # =====================================================
    # Delta7 -> PFL
    #
    # Frozen inhibitory surrogate hypothesis.
    # =====================================================

    if len(
        delta_pfl_source
    ) > 0:

        delta_synapses = Synapses(
            group,
            group,
            model="w : 1",
            on_pre="v_post -= w",
        )


        delta_synapses.connect(
            i=delta_pfl_source,
            j=delta_pfl_target,
        )


        delta_synapses.w = (
            delta_pfl_weights
        )


        objects.append(
            delta_synapses
        )


    # =====================================================
    # Complete frozen PFL -> DNa topology
    # =====================================================

    motor_routes = [

        (
            pfl3_dna02_source,
            pfl3_dna02_target,
            pfl3_dna02_weights,
        ),

        (
            pfl3_dna03_source,
            pfl3_dna03_target,
            pfl3_dna03_weights,
        ),

        (
            pfl2_dna03_source,
            pfl2_dna03_target,
            pfl2_dna03_weights,
        ),

        (
            pfl2_dna02_source,
            pfl2_dna02_target,
            pfl2_dna02_weights,
        ),

    ]


    for (
        source,
        target,
        weights,
    ) in motor_routes:

        synapses = create_positive_synapses(
            group,
            source,
            target,
            weights,
            gain=PFL_TO_DNA_GAIN,
        )


        if synapses is not None:

            objects.append(
                synapses
            )


    # =====================================================
    # DNa03 -> DNa02 relay
    # =====================================================

    synapses = create_positive_synapses(
        group,
        dna03_dna02_source,
        dna03_dna02_target,
        dna03_dna02_weights,
        gain=1.0,
    )


    if synapses is not None:

        objects.append(
            synapses
        )


    # =====================================================
    # Monitor
    # =====================================================

    monitor = SpikeMonitor(
        group
    )


    objects.append(
        monitor
    )


    # =====================================================
    # Run
    # =====================================================

    network = Network(
        *objects
    )


    network.run(
        SIMULATION_TIME
    )


    counts = np.asarray(
        monitor.count
    )


    voltage = np.asarray(
        group.v
    )


    # =====================================================
    # Neural diagnostics
    # =====================================================

    delta7_spikes = safe_sum(
        counts,
        delta7_indices,
    )


    pfl2_spikes = safe_sum(
        counts,
        pfl2_indices,
    )


    pfl3_spikes = safe_sum(
        counts,
        pfl3_indices,
    )


    dna03_spikes = safe_sum(
        counts,
        dna03_indices,
    )


    dna02_spikes = safe_sum(
        counts,
        dna02_indices,
    )


    dna03_l_spikes = safe_sum(
        counts,
        dna03_left,
    )


    dna03_r_spikes = safe_sum(
        counts,
        dna03_right,
    )


    dna02_l_spikes = safe_sum(
        counts,
        dna02_left,
    )


    dna02_r_spikes = safe_sum(
        counts,
        dna02_right,
    )


    dna02_l_v = safe_mean(
        voltage,
        dna02_left,
    )


    dna02_r_v = safe_mean(
        voltage,
        dna02_right,
    )


    # =====================================================
    # Frozen continuous DNa decoder
    # =====================================================

    left_activity = (

        dna02_l_v

        +

        dna02_l_spikes
        /
        SPIKE_SCALE
    )


    right_activity = (

        dna02_r_v

        +

        dna02_r_spikes
        /
        SPIKE_SCALE
    )


    activity_difference = (
        right_activity
        -
        left_activity
    )


    steering = float(
        np.tanh(
            activity_difference
            /
            (
                2.0
                *
                ACTIVITY_SCALE
            )
        )
    )


    # =====================================================
    # Save body IDs for debugging
    # =====================================================

    recruited_a_body_ids = [

        int(
            neurons.loc[
                index,
                "bodyId",
            ]
        )

        for index
        in recruited_a
    ]


    recruited_b_body_ids = [

        int(
            neurons.loc[
                index,
                "bodyId",
            ]
        )

        for index
        in recruited_b
    ]


    return {

        "heading_column":
            heading_column,

        "column_a":
            int(
                column_a
            ),

        "column_b":
            int(
                column_b
            ),

        "alpha":
            float(
                alpha
            ),

        "target_total_recruited":
            int(
                target_total
            ),

        "a_recruited":
            int(
                n_a
            ),

        "a_total":
            int(
                n_a_total
            ),

        "b_recruited":
            int(
                n_b
            ),

        "b_total":
            int(
                n_b_total
            ),

        "total_recruited":
            int(
                n_a
                +
                n_b
            ),

        "a_body_ids":
            ",".join(
                str(
                    body_id
                )

                for body_id
                in recruited_a_body_ids
            ),

        "b_body_ids":
            ",".join(
                str(
                    body_id
                )

                for body_id
                in recruited_b_body_ids
            ),

        "Delta7_spikes":
            delta7_spikes,

        "PFL2_spikes":
            pfl2_spikes,

        "PFL3_spikes":
            pfl3_spikes,

        "DNa03_spikes":
            dna03_spikes,

        "DNa03_L_spikes":
            dna03_l_spikes,

        "DNa03_R_spikes":
            dna03_r_spikes,

        "DNa02_spikes":
            dna02_spikes,

        "DNa02_L_spikes":
            dna02_l_spikes,

        "DNa02_R_spikes":
            dna02_r_spikes,

        "DNa02_L_v":
            dna02_l_v,

        "DNa02_R_v":
            dna02_r_v,

        "left_activity":
            left_activity,

        "right_activity":
            right_activity,

        "activity_difference":
            activity_difference,

        "steering":
            steering,
    }


# =========================================================
# Run full experiment
# =========================================================

rows = []


total_conditions = (

    len(
        REPRESENTATIVE_HEADINGS
    )

    *

    len(
        FC2_PAIRS
    )

    *

    len(
        BLEND_VALUES
    )
)


counter = 0


print()

print("=" * 120)
print(
    "RUNNING ANATOMICALLY BALANCED FC2 RECRUITMENT"
)
print("=" * 120)

print()

print(
    f"Conditions: "
    f"{total_conditions}"
)


for heading in REPRESENTATIVE_HEADINGS:

    for (
        column_a,
        column_b,
    ) in FC2_PAIRS:

        for alpha in BLEND_VALUES:

            counter += 1


            print(
                f"[{counter:3d}/"
                f"{total_conditions}] "
                f"heading={heading:>2} | "
                f"C{column_a}->C{column_b} | "
                f"alpha={alpha:.1f}"
            )


            result = run_condition(
                heading_column=heading,
                column_a=column_a,
                column_b=column_b,
                alpha=float(
                    alpha
                ),
            )


            rows.append(
                result
            )


results = pd.DataFrame(
    rows
)


results.to_csv(
    RESULT_FILE,
    index=False,
)


# =========================================================
# Endpoint validation against frozen Step 27
# =========================================================

print()

print("=" * 120)
print(
    "ENDPOINT VALIDATION AGAINST FROZEN MAP"
)
print("=" * 120)

print()


endpoint_errors = []


if FROZEN_MAP_FILE.exists():

    frozen = pd.read_csv(
        FROZEN_MAP_FILE
    )


    frozen_lookup = {

        (
            str(
                row[
                    "heading_column"
                ]
            ),
            int(
                row[
                    "goal_column"
                ]
            ),
        ):
            float(
                row[
                    "steering"
                ]
            )

        for _, row
        in frozen.iterrows()
    }


    for _, row in results.iterrows():

        alpha = float(
            row[
                "alpha"
            ]
        )


        if np.isclose(
            alpha,
            0.0,
        ):

            expected_column = int(
                row[
                    "column_a"
                ]
            )


        elif np.isclose(
            alpha,
            1.0,
        ):

            expected_column = int(
                row[
                    "column_b"
                ]
            )


        else:

            continue


        key = (
            str(
                row[
                    "heading_column"
                ]
            ),
            expected_column,
        )


        if key not in frozen_lookup:

            continue


        expected = frozen_lookup[
            key
        ]


        actual = float(
            row[
                "steering"
            ]
        )


        endpoint_errors.append(
            abs(
                actual
                -
                expected
            )
        )


    print(
        f"Endpoint comparisons: "
        f"{len(endpoint_errors)}"
    )


    print(
        f"Mean endpoint error: "
        f"{np.mean(endpoint_errors):.10f}"
    )


    print(
        f"Max endpoint error: "
        f"{np.max(endpoint_errors):.10f}"
    )


    if np.max(
        endpoint_errors
    ) <= 1e-8:

        print(
            "PASS: anatomy-balanced endpoints "
            "reproduce frozen Step 27."
        )


    else:

        print(
            "WARNING: endpoint mismatch detected."
        )


else:

    print(
        "Frozen Step 27 map not found."
    )


# =========================================================
# Steering sign helper
# =========================================================

def steering_sign(
    value,
    threshold=0.05,
):

    if value > threshold:

        return 1


    if value < -threshold:

        return -1


    return 0


# =========================================================
# Build trajectory diagnostics
# =========================================================

summary_rows = []


for (
    heading,
    column_a,
    column_b,
), group in results.groupby(
    [
        "heading_column",
        "column_a",
        "column_b",
    ]
):

    group = (
        group
        .sort_values(
            "alpha"
        )
    )


    steering = group[
        "steering"
    ].to_numpy(
        dtype=float,
    )


    alpha = group[
        "alpha"
    ].to_numpy(
        dtype=float,
    )


    steps = np.abs(
        np.diff(
            steering
        )
    )


    start = float(
        steering[
            0
        ]
    )


    end = float(
        steering[
            -1
        ]
    )


    net_change = (
        end
        -
        start
    )


    total_variation = float(
        steps.sum()
    )


    directness_ratio = (

        abs(
            net_change
        )
        /
        total_variation

        if total_variation
        > 1e-12

        else 1.0
    )


    endpoint_min = min(
        start,
        end,
    )


    endpoint_max = max(
        start,
        end,
    )


    lower_overshoot = max(
        0.0,
        endpoint_min
        -
        float(
            steering.min()
        ),
    )


    upper_overshoot = max(
        0.0,
        float(
            steering.max()
        )
        -
        endpoint_max,
    )


    overshoot = max(
        lower_overshoot,
        upper_overshoot,
    )


    signs = [

        steering_sign(
            value
        )

        for value
        in steering
    ]


    nonzero_signs = [

        sign

        for sign
        in signs

        if sign != 0
    ]


    sign_transitions = 0


    for index in range(
        1,
        len(
            nonzero_signs
        ),
    ):

        if (
            nonzero_signs[
                index
            ]
            !=
            nonzero_signs[
                index - 1
            ]
        ):

            sign_transitions += 1


    intermediate_values = steering[
        1:-1
    ]


    near_zero_intermediate = int(
        (
            np.abs(
                intermediate_values
            )
            <
            0.01
        ).sum()
    )


    if len(
        steps
    ) > 0:

        max_step_index = int(
            np.argmax(
                steps
            )
        )


        max_abs_step = float(
            steps[
                max_step_index
            ]
        )


        largest_step_alpha_from = float(
            alpha[
                max_step_index
            ]
        )


        largest_step_alpha_to = float(
            alpha[
                max_step_index
                +
                1
            ]
        )


    else:

        max_abs_step = 0.0

        largest_step_alpha_from = float(
            alpha[
                0
            ]
        )

        largest_step_alpha_to = float(
            alpha[
                0
            ]
        )


    # =====================================================
    # Recruitment-count diagnostics
    # =====================================================

    total_recruited = group[
        "total_recruited"
    ].to_numpy(
        dtype=int,
    )


    recruitment_steps = np.abs(
        np.diff(
            total_recruited
        )
    )


    max_population_count_jump = (

        int(
            recruitment_steps.max()
        )

        if len(
            recruitment_steps
        ) > 0

        else 0
    )


    summary_rows.append(
        {

            "heading_column":
                heading,

            "column_a":
                int(
                    column_a
                ),

            "column_b":
                int(
                    column_b
                ),

            "start_steering":
                start,

            "end_steering":
                end,

            "net_change":
                net_change,

            "mean_abs_step":
                float(
                    steps.mean()
                ),

            "max_abs_step":
                max_abs_step,

            "largest_step_alpha_from":
                largest_step_alpha_from,

            "largest_step_alpha_to":
                largest_step_alpha_to,

            "total_variation":
                total_variation,

            "directness_ratio":
                directness_ratio,

            "overshoot":
                overshoot,

            "sign_transitions":
                int(
                    sign_transitions
                ),

            "near_zero_intermediate":
                near_zero_intermediate,

            "max_population_count_jump":
                max_population_count_jump,

            "max_abs_steering":
                float(
                    np.abs(
                        steering
                    ).max()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =========================================================
# Compact steering table
# =========================================================

table_rows = []


for (
    heading,
    column_a,
    column_b,
), group in results.groupby(
    [
        "heading_column",
        "column_a",
        "column_b",
    ]
):

    group = (
        group
        .sort_values(
            "alpha"
        )
    )


    record = {

        "heading":
            heading,

        "pair":
            f"C{int(column_a)}->C{int(column_b)}",
    }


    for _, row in group.iterrows():

        alpha = float(
            row[
                "alpha"
            ]
        )


        record[
            f"a{alpha:.1f}"
        ] = float(
            row[
                "steering"
            ]
        )


    table_rows.append(
        record
    )


steering_table = pd.DataFrame(
    table_rows
)


print()

print("=" * 180)
print(
    "ANATOMY-BALANCED STEERING TABLE"
)
print("=" * 180)

print()


print(
    steering_table.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.3f}",
    )
)


# =========================================================
# Global diagnostic
# =========================================================

print()

print("=" * 120)
print(
    "GLOBAL ANATOMY-BALANCED RECRUITMENT DIAGNOSTIC"
)
print("=" * 120)

print()


print(
    f"Blend trajectories: "
    f"{len(summary)}"
)


print(
    f"Mean trajectory mean-step: "
    f"{summary['mean_abs_step'].mean():.4f}"
)


print(
    f"Median trajectory mean-step: "
    f"{summary['mean_abs_step'].median():.4f}"
)


print(
    f"Mean maximum step: "
    f"{summary['max_abs_step'].mean():.4f}"
)


print(
    f"Worst single 0.1-alpha step: "
    f"{summary['max_abs_step'].max():.4f}"
)


print(
    f"Mean directness ratio: "
    f"{summary['directness_ratio'].mean():.4f}"
)


print(
    f"Median directness ratio: "
    f"{summary['directness_ratio'].median():.4f}"
)


print(
    f"Maximum interpolation overshoot: "
    f"{summary['overshoot'].max():.4f}"
)


print(
    f"Trajectories with sign transition: "
    f"{int(
        (
            summary[
                'sign_transitions'
            ]
            >
            0
        ).sum()
    )}/"
    f"{len(summary)}"
)


print(
    f"Trajectories with >1 sign transition: "
    f"{int(
        (
            summary[
                'sign_transitions'
            ]
            >
            1
        ).sum()
    )}/"
    f"{len(summary)}"
)


print(
    f"Steps larger than 0.10: "
    f"{int(
        (
            summary[
                'max_abs_step'
            ]
            >
            0.10
        ).sum()
    )}/"
    f"{len(summary)}"
)


print(
    f"Steps larger than 0.20: "
    f"{int(
        (
            summary[
                'max_abs_step'
            ]
            >
            0.20
        ).sum()
    )}/"
    f"{len(summary)}"
)


print(
    f"Trajectories with near-zero "
    f"intermediate state: "
    f"{int(
        (
            summary[
                'near_zero_intermediate'
            ]
            >
            0
        ).sum()
    )}/"
    f"{len(summary)}"
)


print(
    f"Worst recruited-population count jump: "
    f"{int(
        summary[
            'max_population_count_jump'
        ].max()
    )}"
)


# =========================================================
# Compare Steps 28, 29, 30 and 31
# =========================================================

print()

print("=" * 120)
print(
    "INTERPOLATION METHOD COMPARISON"
)
print("=" * 120)

print()


comparison_rows = []


def add_previous_summary(
    name,
    path,
):

    if not path.exists():

        return


    dataframe = pd.read_csv(
        path
    )


    comparison_rows.append(
        {
            "method":
                name,

            "trajectories":
                len(
                    dataframe
                ),

            "mean_max_step":
                float(
                    dataframe[
                        "max_abs_step"
                    ].mean()
                ),

            "worst_step":
                float(
                    dataframe[
                        "max_abs_step"
                    ].max()
                ),

            "mean_directness":
                float(
                    dataframe[
                        "directness_ratio"
                    ].mean()
                ),

            "multi_sign_flip":
                int(
                    (
                        dataframe[
                            "sign_transitions"
                        ]
                        >
                        1
                    ).sum()
                ),
        }
    )


add_previous_summary(
    "Step28 amplitude",
    STEP28_SUMMARY_FILE,
)

add_previous_summary(
    "Step29 temporal",
    STEP29_SUMMARY_FILE,
)

add_previous_summary(
    "Step30 naive recruitment",
    STEP30_SUMMARY_FILE,
)


comparison_rows.append(
    {
        "method":
            "Step31 anatomy-balanced",

        "trajectories":
            len(
                summary
            ),

        "mean_max_step":
            float(
                summary[
                    "max_abs_step"
                ].mean()
            ),

        "worst_step":
            float(
                summary[
                    "max_abs_step"
                ].max()
            ),

        "mean_directness":
            float(
                summary[
                    "directness_ratio"
                ].mean()
            ),

        "multi_sign_flip":
            int(
                (
                    summary[
                        "sign_transitions"
                    ]
                    >
                    1
                ).sum()
            ),
    }
)


comparison_dataframe = pd.DataFrame(
    comparison_rows
)


print(
    comparison_dataframe.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =========================================================
# Worst trajectories
# =========================================================

print()

print("=" * 170)
print(
    "WORST ANATOMY-BALANCED RECRUITMENT TRAJECTORIES"
)
print("=" * 170)

print()


print(
    summary
    .sort_values(
        "max_abs_step",
        ascending=False,
    )
    .head(
        15
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =========================================================
# Detailed worst trajectory
# =========================================================

worst = (
    summary
    .sort_values(
        "max_abs_step",
        ascending=False,
    )
    .iloc[
        0
    ]
)


worst_heading = str(
    worst[
        "heading_column"
    ]
)


worst_a = int(
    worst[
        "column_a"
    ]
)


worst_b = int(
    worst[
        "column_b"
    ]
)


worst_detail = results[
    (
        results[
            "heading_column"
        ]
        ==
        worst_heading
    )
    &
    (
        results[
            "column_a"
        ]
        ==
        worst_a
    )
    &
    (
        results[
            "column_b"
        ]
        ==
        worst_b
    )
].sort_values(
    "alpha"
)


print()

print("=" * 190)
print(
    "DETAIL OF WORST ANATOMY-BALANCED TRAJECTORY"
)
print("=" * 190)

print()


print(
    worst_detail[
        [
            "heading_column",
            "column_a",
            "column_b",
            "alpha",
            "target_total_recruited",
            "a_recruited",
            "a_total",
            "b_recruited",
            "b_total",
            "total_recruited",
            "DNa03_L_spikes",
            "DNa03_R_spikes",
            "DNa02_L_spikes",
            "DNa02_R_spikes",
            "DNa02_L_v",
            "DNa02_R_v",
            "activity_difference",
            "steering",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.5f}",
    )
)


# =========================================================
# Files
# =========================================================

print()

print("=" * 120)
print(
    "FILES CREATED"
)
print("=" * 120)

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)

print(
    SUBSET_FILE
)