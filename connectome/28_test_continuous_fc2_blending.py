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

RESULT_FILE = (
    OUTPUT_DIR
    / "continuous_fc2_blending.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "continuous_fc2_blending_summary.csv"
)


# =========================================================
# Frozen surrogate calibration
#
# DO NOT tune these in this experiment.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0

PFL_TO_DNA_GAIN = 3.0


# =========================================================
# Frozen DNa decoder
# =========================================================

SPIKE_SCALE = 5.0

ACTIVITY_SCALE = 1.10599


# =========================================================
# Neural simulation
# =========================================================

BASELINE_DRIVE = 0.03

EPG_DRIVE = 1.35

FC2_DRIVE = 1.35

SIMULATION_TIME = 300 * ms


# =========================================================
# Experiment
#
# alpha = 0.0
#     pure C_a
#
# alpha = 1.0
#     pure C_b
#
# Intermediate alpha values continuously blend the
# external drives of the two FC2 populations.
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
# Combine connectome edges
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
# Functional connectivity
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
# Frozen topology sanity check
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
print("CONTINUOUS FC2 POPULATION BLENDING TEST")
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
    f"Representative headings: "
    f"{REPRESENTATIVE_HEADINGS}"
)

print(
    f"FC2 adjacent pairs: "
    f"{len(FC2_PAIRS)}"
)

print(
    f"Blend points per pair: "
    f"{len(BLEND_VALUES)}"
)


# =========================================================
# Body ID -> Brian2 index
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
#
# Full 9-dimensional population representation.
# =========================================================

fc2_groups = {
    column: []

    for column in range(
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
# Validate populations
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
# General population helper
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
# Fresh masks against current reset-index edge dataframe
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
# Prepare edge arrays
# =========================================================

def prepare_edges(
    dataframe,
):

    sources = []
    targets = []
    weights = []


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


        weights.append(
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
            weights,
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
# Frozen anatomical weight scaling
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
# FC2 continuous blending
#
# Important:
#
# We interpolate the EXTERNAL DRIVE, not the connectome
# weights.
#
# alpha = 0:
#
#   C_a receives FC2_DRIVE
#   C_b receives BASELINE_DRIVE
#
# alpha = 1:
#
#   C_a receives BASELINE_DRIVE
#   C_b receives FC2_DRIVE
#
# Intermediate values continuously redistribute drive.
# =========================================================

def apply_fc2_blend(
    group,
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


    active_range = (
        FC2_DRIVE
        -
        BASELINE_DRIVE
    )


    drive_a = (
        BASELINE_DRIVE
        +
        (
            1.0
            -
            alpha
        )
        * active_range
    )


    drive_b = (
        BASELINE_DRIVE
        +
        alpha
        * active_range
    )


    for index in fc2_groups[
        column_a
    ]:

        group.drive[
            index
        ] = drive_a


    for index in fc2_groups[
        column_b
    ]:

        group.drive[
            index
        ] = drive_b


    return (
        drive_a,
        drive_b,
    )


# =========================================================
# One simulation condition
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
    # EPG heading
    # =====================================================

    for index in epg_groups[
        heading_column
    ]:

        group.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # Continuous FC2 mixture
    # =====================================================

    (
        drive_a,
        drive_b,
    ) = apply_fc2_blend(
        group,
        column_a,
        column_b,
        alpha,
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
    # Delta7 -> PFL inhibitory hypothesis
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
    # Complete PFL -> DNa topology
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
    # DNa03 -> DNa02 relay
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
    # Monitor
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
    # Frozen decoder
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


    return {

        "heading_column":
            heading_column,

        "column_a":
            column_a,

        "column_b":
            column_b,

        "alpha":
            float(alpha),

        "drive_a":
            float(drive_a),

        "drive_b":
            float(drive_b),

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
# Run experiment
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
print("RUNNING CONTINUOUS FC2 BLENDS")
print("=" * 120)

print()


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
# Validate alpha endpoints against frozen Step 27
# =========================================================

print()

print("=" * 120)
print("ENDPOINT VALIDATION AGAINST FROZEN MAP")
print("=" * 120)

print()


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


    endpoint_errors = []


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

        max_endpoint_error = float(
            np.max(
                endpoint_errors
            )
        )


        mean_endpoint_error = float(
            np.mean(
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


        if max_endpoint_error > 1e-8:

            print(
                "WARNING: endpoint result differs "
                "from frozen Step 27."
            )

        else:

            print(
                "PASS: blended endpoints reproduce "
                "the frozen discrete controller."
            )


else:

    print(
        "Frozen map file not found. "
        "Endpoint comparison skipped."
    )


# =========================================================
# Sign helper
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
# Per heading/pair interpolation diagnostics
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


    steps = np.diff(
        steering
    )


    abs_steps = np.abs(
        steps
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


    directness = (

        abs(
            net_change
        )
        /
        total_variation

        if total_variation > 1e-12

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


    below = max(
        0.0,
        endpoint_min
        -
        float(
            steering.min()
        ),
    )


    above = max(
        0.0,
        float(
            steering.max()
        )
        -
        endpoint_max,
    )


    overshoot = max(
        below,
        above,
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


    largest_step_index = int(
        np.argmax(
            abs_steps
        )
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
                    abs_steps.mean()
                ),

            "max_abs_step":
                float(
                    abs_steps.max()
                ),

            "largest_step_alpha_from":
                float(
                    alpha[
                        largest_step_index
                    ]
                ),

            "largest_step_alpha_to":
                float(
                    alpha[
                        largest_step_index
                        + 1
                    ]
                ),

            "total_variation":
                total_variation,

            "directness_ratio":
                directness,

            "overshoot":
                overshoot,

            "sign_transitions":
                int(
                    sign_transitions
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
# Steering table
#
# One row per heading + FC2 pair.
# Columns show alpha 0.0 ... 1.0.
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


# =========================================================
# Print blend table
# =========================================================

print()

print("=" * 180)
print("CONTINUOUS FC2 BLEND STEERING TABLE")
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
# Smoothness summary
# =========================================================

print()

print("=" * 160)
print("BLEND SMOOTHNESS SUMMARY")
print("=" * 160)

print()


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Global diagnostics
# =========================================================

print()

print("=" * 120)
print("GLOBAL BLENDING DIAGNOSTIC")
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


# =========================================================
# Worst blend trajectories
# =========================================================

print()

print("=" * 150)
print("WORST BLEND TRAJECTORIES")
print("=" * 150)

print()


print(
    summary
    .sort_values(
        "max_abs_step",
        ascending=False,
    )
    .head(
        12
    )
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Individual worst step
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

print("=" * 150)
print("DETAIL OF WORST BLEND TRAJECTORY")
print("=" * 150)

print()


print(
    worst_detail[
        [
            "heading_column",
            "column_a",
            "column_b",
            "alpha",
            "drive_a",
            "drive_b",
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