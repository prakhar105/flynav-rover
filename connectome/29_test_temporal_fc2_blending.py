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

FROZEN_MAP_FILE = (
    OUTPUT_DIR
    / "frozen_fc2_population_map.csv"
)

AMPLITUDE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "continuous_fc2_blending_summary.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "temporal_fc2_blending.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "temporal_fc2_blending_summary.csv"
)


# =========================================================
# Frozen surrogate parameters
#
# DO NOT tune any of these in this experiment.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0

PFL_TO_DNA_GAIN = 3.0


# =========================================================
# Frozen DNa decoder
# =========================================================

SPIKE_SCALE = 5.0

ACTIVITY_SCALE = 1.10599


# =========================================================
# Frozen neural input levels
# =========================================================

BASELINE_DRIVE = 0.03

EPG_DRIVE = 1.35

FC2_DRIVE = 1.35


# =========================================================
# Temporal encoding
#
# Total neural run:
#
# 30 slots × 10 ms = 300 ms
#
# Every ACTIVE FC2 population always receives 1.35.
#
# Continuous mixture is represented by how many slots
# belong to C_a versus C_b.
#
# Example:
#
# alpha = 0.3
#
# C_a active for 21 / 30 slots
# C_b active for  9 / 30 slots
#
# This is an engineering interface approximation.
# It is NOT a claim about literal biological encoding.
# =========================================================

N_TIME_SLOTS = 30

SLOT_DURATION = 10 * ms

TOTAL_SIMULATION_TIME = (
    N_TIME_SLOTS
    * SLOT_DURATION
)


BLEND_VALUES = np.linspace(
    0.0,
    1.0,
    11,
)


# =========================================================
# Representative EPG headings
# =========================================================

REPRESENTATIVE_HEADINGS = [
    "L1",
    "L5",
    "R5",
    "R1",
]


# =========================================================
# Test all adjacent FC2 transitions
#
# Include C9 -> C1 wrap.
# =========================================================

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
# Neural populations
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
# Combine anatomical edges
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
# Functional network
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
# Frozen network validation
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
print("TEMPORAL FC2 POPULATION BLENDING TEST")
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
    f"Time slots: "
    f"{N_TIME_SLOTS}"
)

print(
    f"Slot duration: "
    f"{float(SLOT_DURATION / ms):.1f} ms"
)

print(
    f"Total neural time: "
    f"{float(TOTAL_SIMULATION_TIME / ms):.1f} ms"
)

print(
    f"Representative headings: "
    f"{REPRESENTATIVE_HEADINGS}"
)

print(
    f"FC2 transitions: "
    f"{FC2_PAIRS}"
)


# =========================================================
# Brian2 index lookup
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
# Population parsers
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
# EPG groups
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
# FC2 groups
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
# Validate EPG / FC2 extraction
# =========================================================

epg_count = sum(
    len(indices)

    for indices
    in epg_groups.values()
)


fc2_count = sum(
    len(indices)

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
# Split pathways
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
# Edge accounting
# =========================================================

accounted_edges = (

    len(epg_delta_edges)

    +
    len(delta_pfl_edges)

    +
    len(pfl3_dna02_edges)

    +
    len(pfl3_dna03_edges)

    +
    len(pfl2_dna03_edges)

    +
    len(pfl2_dna02_edges)

    +
    len(dna03_dna02_edges)

    +
    len(other_edges)
)


if accounted_edges != len(
    edges
):

    raise RuntimeError(
        "Functional edge accounting failed: "
        f"{accounted_edges} accounted, "
        f"{len(edges)} expected."
    )


print()

print("=" * 120)
print("FROZEN PATHWAY INVENTORY")
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
# Edge preparation
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


        if source_id not in body_to_index:

            continue


        if target_id not in body_to_index:

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
# Frozen anatomical scaling
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

            if len(array) > 0
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
            / MAX_RAW
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


def create_positive_synapses(
    group,
    source,
    target,
    weights,
    gain=1.0,
):

    if len(source) == 0:

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
        * gain
    )


    return synapses


# =========================================================
# Temporal slot scheduling
#
# Deterministic error-accumulator / Bresenham-like schedule.
#
# It spreads C_b slots throughout the 300 ms window rather
# than putting all C_a first and all C_b afterwards.
#
# For alpha = 0.3:
#
# 9 of 30 slots become B.
#
# The remaining 21 become A.
# =========================================================

def build_temporal_schedule(
    alpha,
):

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )


    n_b = int(
        round(
            alpha
            * N_TIME_SLOTS
        )
    )


    n_b = int(
        np.clip(
            n_b,
            0,
            N_TIME_SLOTS,
        )
    )


    n_a = (
        N_TIME_SLOTS
        -
        n_b
    )


    # -----------------------------------------------------
    # Pure endpoints
    # -----------------------------------------------------

    if n_b == 0:

        schedule = [
            "A"
        ] * N_TIME_SLOTS


    elif n_a == 0:

        schedule = [
            "B"
        ] * N_TIME_SLOTS


    else:

        schedule = []

        accumulator = 0


        for _ in range(
            N_TIME_SLOTS
        ):

            accumulator += (
                n_b
            )


            if accumulator >= N_TIME_SLOTS:

                schedule.append(
                    "B"
                )

                accumulator -= (
                    N_TIME_SLOTS
                )


            else:

                schedule.append(
                    "A"
                )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    actual_a = schedule.count(
        "A"
    )

    actual_b = schedule.count(
        "B"
    )


    if actual_a != n_a:

        raise RuntimeError(
            "Temporal schedule A count mismatch."
        )


    if actual_b != n_b:

        raise RuntimeError(
            "Temporal schedule B count mismatch."
        )


    return (
        schedule,
        n_a,
        n_b,
    )


# =========================================================
# FC2 slot activation
#
# All FC2 neurons remain at baseline unless their C-column
# is selected for the current time slot.
#
# Selected population receives the full 1.35 drive.
# =========================================================

def activate_fc2_column(
    group,
    column_a,
    column_b,
    active_label,
):

    # -----------------------------------------------------
    # Reset both candidate columns to baseline.
    # -----------------------------------------------------

    for index in fc2_groups[
        column_a
    ]:

        group.drive[
            index
        ] = BASELINE_DRIVE


    for index in fc2_groups[
        column_b
    ]:

        group.drive[
            index
        ] = BASELINE_DRIVE


    # -----------------------------------------------------
    # Activate selected population at FULL drive.
    # -----------------------------------------------------

    if active_label == "A":

        active_column = (
            column_a
        )


    elif active_label == "B":

        active_column = (
            column_b
        )


    else:

        raise ValueError(
            f"Unknown slot label: {active_label}"
        )


    for index in fc2_groups[
        active_column
    ]:

        group.drive[
            index
        ] = FC2_DRIVE


# =========================================================
# Run one temporal-blending neural condition
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
        len(neurons),
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
    # Fixed EPG heading for entire 300 ms simulation
    # =====================================================

    for index in epg_groups[
        heading_column
    ]:

        group.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # Build temporal FC2 schedule
    # =====================================================

    (
        slot_schedule,
        n_a_slots,
        n_b_slots,
    ) = build_temporal_schedule(
        alpha
    )


    objects = [
        group
    ]


    # =====================================================
    # EPG + FC2 -> PFL
    # =====================================================

    syn = create_positive_synapses(
        group,
        other_source,
        other_target,
        other_weights,
        gain=1.0,
    )


    if syn is not None:

        objects.append(
            syn
        )


    # =====================================================
    # EPG -> Delta7
    # =====================================================

    syn = create_positive_synapses(
        group,
        epg_delta_source,
        epg_delta_target,
        epg_delta_weights,
        gain=EPG_TO_DELTA7_GAIN,
    )


    if syn is not None:

        objects.append(
            syn
        )


    # =====================================================
    # Delta7 -> PFL inhibitory model hypothesis
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

    for (
        source,
        target,
        weights,
    ) in [

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

    ]:

        syn = create_positive_synapses(
            group,
            source,
            target,
            weights,
            gain=PFL_TO_DNA_GAIN,
        )


        if syn is not None:

            objects.append(
                syn
            )


    # =====================================================
    # DNa03 -> DNa02 anatomical relay
    # =====================================================

    syn = create_positive_synapses(
        group,
        dna03_dna02_source,
        dna03_dna02_target,
        dna03_dna02_weights,
        gain=1.0,
    )


    if syn is not None:

        objects.append(
            syn
        )


    # =====================================================
    # Spike monitor accumulates across all time slots.
    # =====================================================

    monitor = SpikeMonitor(
        group
    )


    objects.append(
        monitor
    )


    network = Network(
        *objects
    )


    # =====================================================
    # Temporal FC2 encoding
    #
    # Network state is NOT reset between slots.
    #
    # Only the sensory drive switches.
    # =====================================================

    for slot_label in slot_schedule:

        activate_fc2_column(
            group,
            column_a,
            column_b,
            slot_label,
        )


        network.run(
            SLOT_DURATION
        )


    # =====================================================
    # Read neural state after complete 300 ms
    # =====================================================

    counts = np.asarray(
        monitor.count
    )


    voltage = np.asarray(
        group.v
    )


    # =====================================================
    # Population diagnostics
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


    dna03_l = safe_sum(
        counts,
        dna03_left,
    )


    dna03_r = safe_sum(
        counts,
        dna03_right,
    )


    dna02_l = safe_sum(
        counts,
        dna02_left,
    )


    dna02_r = safe_sum(
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
    # Frozen Step 27 continuous decoder
    # =====================================================

    left_activity = (
        dna02_l_v
        +
        dna02_l
        / SPIKE_SCALE
    )


    right_activity = (
        dna02_r_v
        +
        dna02_r
        / SPIKE_SCALE
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
                * ACTIVITY_SCALE
            )
        )
    )


    schedule_string = "".join(
        slot_schedule
    )


    switch_count = int(
        sum(
            1

            for index in range(
                1,
                len(
                    slot_schedule
                ),
            )

            if (
                slot_schedule[
                    index
                ]
                !=
                slot_schedule[
                    index - 1
                ]
            )
        )
    )


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

        "a_slots":
            int(
                n_a_slots
            ),

        "b_slots":
            int(
                n_b_slots
            ),

        "switch_count":
            switch_count,

        "slot_schedule":
            schedule_string,

        "Delta7_spikes":
            delta7_spikes,

        "PFL2_spikes":
            pfl2_spikes,

        "PFL3_spikes":
            pfl3_spikes,

        "DNa03_spikes":
            dna03_spikes,

        "DNa03_L_spikes":
            dna03_l,

        "DNa03_R_spikes":
            dna03_r,

        "DNa02_spikes":
            dna02_spikes,

        "DNa02_L_spikes":
            dna02_l,

        "DNa02_R_spikes":
            dna02_r,

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
# Show temporal schedules once
# =========================================================

print()

print("=" * 120)
print("TEMPORAL SLOT SCHEDULES")
print("=" * 120)

print()


for alpha in BLEND_VALUES:

    (
        schedule,
        n_a,
        n_b,
    ) = build_temporal_schedule(
        alpha
    )


    print(
        f"alpha={alpha:.1f} "
        f"| A={n_a:2d} "
        f"| B={n_b:2d} "
        f"| {''.join(schedule)}"
    )


# =========================================================
# Run full temporal-blending experiment
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
print("RUNNING TEMPORAL FC2 BLENDS")
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


            rows.append(
                run_condition(
                    heading_column=heading,
                    column_a=column_a,
                    column_b=column_b,
                    alpha=float(
                        alpha
                    ),
                )
            )


results = pd.DataFrame(
    rows
)


results.to_csv(
    RESULT_FILE,
    index=False,
)


# =========================================================
# Validate temporal endpoints against frozen Step 27
#
# alpha = 0:
#   A active for all 300 ms
#
# alpha = 1:
#   B active for all 300 ms
#
# Therefore these should reproduce Step 27.
# =========================================================

print()

print("=" * 120)
print("ENDPOINT VALIDATION AGAINST FROZEN MAP")
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


    if endpoint_errors:

        mean_endpoint_error = float(
            np.mean(
                endpoint_errors
            )
        )


        max_endpoint_error = float(
            np.max(
                endpoint_errors
            )
        )


        print(
            f"Endpoint comparisons: "
            f"{len(endpoint_errors)}"
        )


        print(
            f"Mean endpoint error: "
            f"{mean_endpoint_error:.10f}"
        )


        print(
            f"Max endpoint error: "
            f"{max_endpoint_error:.10f}"
        )


        if max_endpoint_error <= 1e-8:

            print(
                "PASS: temporal endpoints reproduce "
                "the frozen discrete controller."
            )


        else:

            print(
                "WARNING: temporal endpoint differs "
                "from the frozen baseline."
            )


else:

    print(
        "Frozen map not found; "
        "endpoint validation skipped."
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
# Per-trajectory diagnostics
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


    abs_steps = np.abs(
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
        abs_steps.sum()
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


    low_overshoot = max(
        0.0,
        endpoint_min
        -
        float(
            steering.min()
        ),
    )


    high_overshoot = max(
        0.0,
        float(
            steering.max()
        )
        -
        endpoint_max,
    )


    overshoot = max(
        low_overshoot,
        high_overshoot,
    )


    signs = [
        steering_sign(
            value
        )

        for value
        in steering
    ]


    nonzero_signs = [
        value

        for value
        in signs

        if value != 0
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


    max_step_index = int(
        np.argmax(
            abs_steps
        )
    )


    # -----------------------------------------------------
    # Near-zero intermediate states
    # -----------------------------------------------------

    intermediate_mask = (
        (alpha > 0.0)
        &
        (alpha < 1.0)
    )


    intermediate_values = steering[
        intermediate_mask
    ]


    near_zero_intermediate = int(
        (
            np.abs(
                intermediate_values
            )
            < 0.01
        ).sum()
    )


    # -----------------------------------------------------
    # Longest consecutive near-zero run among intermediates
    # -----------------------------------------------------

    longest_zero_run = 0
    current_zero_run = 0


    for value in intermediate_values:

        if abs(
            value
        ) < 0.01:

            current_zero_run += 1


            longest_zero_run = max(
                longest_zero_run,
                current_zero_run,
            )


        else:

            current_zero_run = 0


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
                    abs_steps.mean()
                ),

            "max_abs_step":
                float(
                    abs_steps.max()
                ),

            "largest_step_alpha_from":
                float(
                    alpha[
                        max_step_index
                    ]
                ),

            "largest_step_alpha_to":
                float(
                    alpha[
                        max_step_index
                        + 1
                    ]
                ),

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

            "longest_zero_run":
                int(
                    longest_zero_run
                ),

            "max_abs_steering":
                float(
                    np.max(
                        np.abs(
                            steering
                        )
                    )
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
# Compact blend table
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


blend_table = pd.DataFrame(
    table_rows
)


print()

print("=" * 180)
print("TEMPORAL FC2 BLEND STEERING TABLE")
print("=" * 180)

print()


print(
    blend_table.to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.3f}",
    )
)


# =========================================================
# Summary table
# =========================================================

print()

print("=" * 170)
print("TEMPORAL BLEND SMOOTHNESS SUMMARY")
print("=" * 170)

print()


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Global temporal-blending diagnostic
# =========================================================

print()

print("=" * 120)
print("GLOBAL TEMPORAL BLENDING DIAGNOSTIC")
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
    f"{int((summary['sign_transitions'] > 0).sum())}/"
    f"{len(summary)}"
)


print(
    f"Trajectories with >1 sign transition: "
    f"{int((summary['sign_transitions'] > 1).sum())}/"
    f"{len(summary)}"
)


print(
    f"Steps larger than 0.10: "
    f"{int((summary['max_abs_step'] > 0.10).sum())}/"
    f"{len(summary)} trajectories"
)


print(
    f"Steps larger than 0.20: "
    f"{int((summary['max_abs_step'] > 0.20).sum())}/"
    f"{len(summary)} trajectories"
)


print(
    f"Trajectories with any near-zero "
    f"intermediate state: "
    f"{int((summary['near_zero_intermediate'] > 0).sum())}/"
    f"{len(summary)}"
)


print(
    f"Worst consecutive near-zero run: "
    f"{int(summary['longest_zero_run'].max())} "
    f"alpha steps"
)


# =========================================================
# Compare with Step 28 amplitude blending
# =========================================================

print()

print("=" * 120)
print("AMPLITUDE VS TEMPORAL BLENDING")
print("=" * 120)

print()


temporal_worst_step = float(
    summary[
        "max_abs_step"
    ].max()
)


temporal_mean_max_step = float(
    summary[
        "max_abs_step"
    ].mean()
)


if AMPLITUDE_SUMMARY_FILE.exists():

    amplitude_summary = pd.read_csv(
        AMPLITUDE_SUMMARY_FILE
    )


    amplitude_worst_step = float(
        amplitude_summary[
            "max_abs_step"
        ].max()
    )


    amplitude_mean_max_step = float(
        amplitude_summary[
            "max_abs_step"
        ].mean()
    )


    improvement = (
        amplitude_worst_step
        -
        temporal_worst_step
    )


    percent_improvement = (

        100.0
        *
        improvement
        /
        amplitude_worst_step

        if amplitude_worst_step
        > 1e-12

        else 0.0
    )


    print(
        f"Amplitude worst step: "
        f"{amplitude_worst_step:.4f}"
    )


    print(
        f"Temporal worst step:  "
        f"{temporal_worst_step:.4f}"
    )


    print(
        f"Worst-step reduction: "
        f"{improvement:.4f} "
        f"({percent_improvement:.1f}%)"
    )


    print()


    print(
        f"Amplitude mean max-step: "
        f"{amplitude_mean_max_step:.4f}"
    )


    print(
        f"Temporal mean max-step:  "
        f"{temporal_mean_max_step:.4f}"
    )


else:

    print(
        "Amplitude blending summary not found."
    )


# =========================================================
# Worst trajectories
# =========================================================

print()

print("=" * 160)
print("WORST TEMPORAL BLEND TRAJECTORIES")
print("=" * 160)

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
# Detail of worst trajectory
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
        == worst_heading
    )
    &
    (
        results[
            "column_a"
        ]
        == worst_a
    )
    &
    (
        results[
            "column_b"
        ]
        == worst_b
    )
].sort_values(
    "alpha"
)


print()

print("=" * 175)
print("DETAIL OF WORST TEMPORAL BLEND")
print("=" * 175)

print()


print(
    worst_detail[
        [
            "heading_column",
            "column_a",
            "column_b",
            "alpha",
            "a_slots",
            "b_slots",
            "switch_count",
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
print("FILES CREATED")
print("=" * 120)

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)