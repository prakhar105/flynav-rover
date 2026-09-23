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

STEERING_NEURON_FILE = OUTPUT_DIR / "steering_core_neurons.csv"
STEERING_EDGE_FILE = OUTPUT_DIR / "steering_core_edges.csv"

DELTA7_NEURON_FILE = OUTPUT_DIR / "delta7_candidates.csv"
DELTA7_EDGE_FILE = OUTPUT_DIR / "delta7_bridge_edges.csv"

RESULT_FILE = OUTPUT_DIR / "frozen_fc2_population_map.csv"

STEERING_MATRIX_FILE = (
    OUTPUT_DIR
    / "frozen_fc2_steering_matrix.csv"
)

MEMBRANE_MATRIX_FILE = (
    OUTPUT_DIR
    / "frozen_fc2_membrane_matrix.csv"
)

SPIKE_MATRIX_FILE = (
    OUTPUT_DIR
    / "frozen_fc2_spike_matrix.csv"
)


# =========================================================
# Frozen surrogate calibration
#
# Engineering calibration values for the Brian2 surrogate.
# These are NOT biological synaptic multipliers.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN = 3.0


# =========================================================
# Frozen continuous DNa02 decoder
#
# These came from the decoder comparison experiment.
# Do NOT recompute dynamically.
# =========================================================

SPIKE_SCALE = 5.0
ACTIVITY_SCALE = 1.10599


# =========================================================
# Simulation settings
# =========================================================

BASELINE_DRIVE = 0.03

EPG_DRIVE = 1.35
FC2_DRIVE = 1.35

SIMULATION_TIME = 300 * ms


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
        subset=["bodyId"]
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
    edges["source_bodyId"].isin(
        body_ids
    )
    &
    edges["target_bodyId"].isin(
        body_ids
    )
].copy()


# =========================================================
# Preserve previous duplicate handling
#
# If the same body pair appears more than once,
# retain the highest raw anatomical edge weight.
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
# Functional connectivity masks
# =========================================================

mask_epg_delta = (
    (edges["source_type"] == "EPG")
    &
    (edges["target_type"] == "Delta7")
)


mask_epg_pfl = (
    (edges["source_type"] == "EPG")
    &
    edges["target_type"].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
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


mask_fc2_pfl = (
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


# =========================================================
# COMPLETE MaleCNS PFL -> DNa topology
#
# IMPORTANT:
#
# We now preserve ALL extracted valid PFL -> DNa edges:
#
# PFL3 -> DNa02
# PFL3 -> DNa03
# PFL2 -> DNa03
# PFL2 -> DNa02
#
# Even though PFL2 -> DNa02 has tiny raw total weight,
# we do NOT manually prune valid anatomical connections.
# =========================================================

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


# =========================================================
# Final functional network
# =========================================================

functional_mask = (

    mask_epg_delta

    |

    mask_epg_pfl

    |

    mask_delta_pfl

    |

    mask_fc2_pfl

    |

    mask_pfl3_dna02

    |

    mask_pfl3_dna03

    |

    mask_pfl2_dna03

    |

    mask_pfl2_dna02

    |

    mask_dna03_dna02
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
# Startup summary
# =========================================================

print("=" * 110)
print(
    "FROZEN MALECNS FC2 POPULATION NAVIGATION MAP"
)
print("=" * 110)

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
    f"DNa spike scale: "
    f"{SPIKE_SCALE:.5f}"
)

print(
    f"DNa activity scale: "
    f"{ACTIVITY_SCALE:.5f}"
)


# =========================================================
# Index lookup
# =========================================================

body_to_index = {
    int(body_id): index

    for index, body_id
    in enumerate(
        neurons["bodyId"]
    )
}


# =========================================================
# Column parsers
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


    side = match.group(1)

    number = int(
        match.group(2)
    )


    if not (
        1 <= number <= 8
    ):

        return None


    return f"{side}{number}"


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
        match.group(1)
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
# FC2 groups
#
# Keep all 9 anatomical FC2 C-columns.
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
# Validate grouping
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


# =========================================================
# Population helper
# =========================================================

def get_indices(
    neuron_type,
    side=None,
):

    output = []


    for index, row in neurons.iterrows():

        if row["type"] != neuron_type:

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


# =========================================================
# Population indices
# =========================================================

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
# Split functional pathways
# =========================================================

epg_delta_edges = edges[
    (
        (edges["source_type"] == "EPG")
        &
        (edges["target_type"] == "Delta7")
    )
].copy()


delta_pfl_edges = edges[
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
].copy()


pfl3_dna02_edges = edges[
    (
        (edges["source_type"] == "PFL3")
        &
        (edges["target_type"] == "DNa02")
    )
].copy()


pfl3_dna03_edges = edges[
    (
        (edges["source_type"] == "PFL3")
        &
        (edges["target_type"] == "DNa03")
    )
].copy()


pfl2_dna03_edges = edges[
    (
        (edges["source_type"] == "PFL2")
        &
        (edges["target_type"] == "DNa03")
    )
].copy()


pfl2_dna02_edges = edges[
    (
        (edges["source_type"] == "PFL2")
        &
        (edges["target_type"] == "DNa02")
    )
].copy()


dna03_dna02_edges = edges[
    (
        (edges["source_type"] == "DNa03")
        &
        (edges["target_type"] == "DNa02")
    )
].copy()


# =========================================================
# Everything else:
#
# EPG -> PFL
# FC2 -> PFL
#
# All downstream routes are separated explicitly.
# =========================================================

# =========================================================
# Other upstream connectivity
#
# IMPORTANT:
#
# Build this mask directly against the CURRENT `edges`
# dataframe.
#
# Do not reuse Boolean masks created before
# edges.reset_index(drop=True), because their Pandas
# indices no longer necessarily align.
#
# The only connectivity belonging here is:
#
#   EPG -> PFL2 / PFL3
#   FC2 -> PFL2 / PFL3
# =========================================================

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


other_edges = edges.loc[
    mask_other_upstream
].copy()


# =========================================================
# Sanity check
#
# All 3231 functional edges should be accounted for exactly
# once by our functional pathway groups.
# =========================================================

accounted_edge_count = (

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


if accounted_edge_count != len(edges):

    raise RuntimeError(
        "Functional edge accounting mismatch: "
        f"{accounted_edge_count} accounted for "
        f"but network contains {len(edges)} edges."
    )

# =========================================================
# Pathway inventory
# =========================================================

print()

print("=" * 110)
print(
    "FROZEN PATHWAY INVENTORY"
)
print("=" * 110)


print(
    f"EPG -> Delta7 edges: "
    f"{len(epg_delta_edges)}"
)

print(
    f"Delta7 -> PFL edges: "
    f"{len(delta_pfl_edges)}"
)

print(
    f"PFL3 -> DNa02 edges: "
    f"{len(pfl3_dna02_edges)}"
)

print(
    f"PFL3 -> DNa03 edges: "
    f"{len(pfl3_dna03_edges)}"
)

print(
    f"PFL2 -> DNa03 edges: "
    f"{len(pfl2_dna03_edges)}"
)

print(
    f"PFL2 -> DNa02 edges: "
    f"{len(pfl2_dna02_edges)}"
)

print(
    f"DNa03 -> DNa02 edges: "
    f"{len(dna03_dna02_edges)}"
)

print(
    f"Other upstream edges: "
    f"{len(other_edges)}"
)


# =========================================================
# Important sanity checks
# =========================================================

pfl_dna_total = (
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
)


print()

print(
    f"Total PFL -> DNa edges: "
    f"{pfl_dna_total}"
)


if pfl_dna_total != 76:

    raise RuntimeError(
        f"Expected 76 PFL -> DNa edges, "
        f"got {pfl_dna_total}."
    )


# =========================================================
# Convert edge dataframes into Brian2 arrays
# =========================================================

def prepare_edges(
    dataframe,
):

    sources = []
    targets = []
    raw_weights = []


    for _, row in dataframe.iterrows():

        source_body_id = int(
            row["source_bodyId"]
        )

        target_body_id = int(
            row["target_bodyId"]
        )


        if (
            source_body_id
            not in body_to_index
        ):

            continue


        if (
            target_body_id
            not in body_to_index
        ):

            continue


        sources.append(
            body_to_index[
                source_body_id
            ]
        )


        targets.append(
            body_to_index[
                target_body_id
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
# Common anatomical weight scaling
#
# Preserve relative raw MaleCNS weights.
# =========================================================

weight_arrays = [
    epg_delta_raw,
    delta_pfl_raw,
    pfl3_dna02_raw,
    pfl3_dna03_raw,
    pfl2_dna03_raw,
    pfl2_dna02_raw,
    dna03_dna02_raw,
    other_raw,
]


nonempty_arrays = [
    values

    for values
    in weight_arrays

    if len(values) > 0
]


all_raw = np.concatenate(
    nonempty_arrays
)


MAX_RAW = max(
    float(
        all_raw.max()
    ),
    1.0,
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


# =========================================================
# Positive synapse helper
# =========================================================

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
# Run one EPG heading x FC2 column
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

    G.drive = (
        BASELINE_DRIVE
    )


    # =====================================================
    # EPG heading drive
    # =====================================================

    for index in epg_groups[
        heading_column
    ]:

        G.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # FC2 goal population drive
    #
    # Entire C-column population remains active.
    # =====================================================

    for index in fc2_groups[
        goal_column
    ]:

        G.drive[
            index
        ] = FC2_DRIVE


    network_objects = [
        G
    ]


    # =====================================================
    # EPG -> PFL and FC2 -> PFL
    # =====================================================

    S_other = create_positive_synapses(
        G,
        other_source,
        other_target,
        other_weights,
        gain=1.0,
    )


    if S_other is not None:

        network_objects.append(
            S_other
        )


    # =====================================================
    # EPG -> Delta7
    # =====================================================

    S_epg_delta = create_positive_synapses(
        G,
        epg_delta_source,
        epg_delta_target,
        epg_delta_weights,
        gain=EPG_TO_DELTA7_GAIN,
    )


    if S_epg_delta is not None:

        network_objects.append(
            S_epg_delta
        )


    # =====================================================
    # Delta7 -> PFL inhibitory
    # =====================================================

    if len(
        delta_pfl_source
    ) > 0:

        S_delta_pfl = Synapses(
            G,
            G,
            model="w : 1",
            on_pre="v_post -= w",
        )


        S_delta_pfl.connect(
            i=delta_pfl_source,
            j=delta_pfl_target,
        )


        S_delta_pfl.w = (
            delta_pfl_weights
        )


        network_objects.append(
            S_delta_pfl
        )


    # =====================================================
    # PFL3 -> DNa02
    # =====================================================

    S_pfl3_dna02 = create_positive_synapses(
        G,
        pfl3_dna02_source,
        pfl3_dna02_target,
        pfl3_dna02_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if S_pfl3_dna02 is not None:

        network_objects.append(
            S_pfl3_dna02
        )


    # =====================================================
    # PFL3 -> DNa03
    # =====================================================

    S_pfl3_dna03 = create_positive_synapses(
        G,
        pfl3_dna03_source,
        pfl3_dna03_target,
        pfl3_dna03_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if S_pfl3_dna03 is not None:

        network_objects.append(
            S_pfl3_dna03
        )


    # =====================================================
    # PFL2 -> DNa03
    # =====================================================

    S_pfl2_dna03 = create_positive_synapses(
        G,
        pfl2_dna03_source,
        pfl2_dna03_target,
        pfl2_dna03_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if S_pfl2_dna03 is not None:

        network_objects.append(
            S_pfl2_dna03
        )


    # =====================================================
    # PFL2 -> DNa02
    #
    # RESTORED.
    #
    # These four anatomical connections are preserved even
    # though their raw total weight is small.
    # =====================================================

    S_pfl2_dna02 = create_positive_synapses(
        G,
        pfl2_dna02_source,
        pfl2_dna02_target,
        pfl2_dna02_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if S_pfl2_dna02 is not None:

        network_objects.append(
            S_pfl2_dna02
        )


    # =====================================================
    # DNa03 -> DNa02
    #
    # Preserve anatomical relay at 1x.
    # =====================================================

    S_dna03_dna02 = create_positive_synapses(
        G,
        dna03_dna02_source,
        dna03_dna02_target,
        dna03_dna02_weights,
        gain=1.0,
    )


    if S_dna03_dna02 is not None:

        network_objects.append(
            S_dna03_dna02
        )


    # =====================================================
    # Spike monitor
    # =====================================================

    monitor = SpikeMonitor(
        G
    )


    network_objects.append(
        monitor
    )


    network = Network(
        *network_objects
    )


    network.run(
        SIMULATION_TIME
    )


    counts = np.asarray(
        monitor.count
    )


    voltage = np.asarray(
        G.v
    )


    # =====================================================
    # Layer diagnostics
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


    # =====================================================
    # DNa03 side activity
    # =====================================================

    dna03_l_spikes = safe_sum(
        counts,
        dna03_left,
    )

    dna03_r_spikes = safe_sum(
        counts,
        dna03_right,
    )


    # =====================================================
    # DNa02 side activity
    # =====================================================

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
    # Diagnostic asymmetries
    # =====================================================

    membrane_difference = (
        dna02_r_v
        -
        dna02_l_v
    )


    spike_count_difference = (
        dna02_r_spikes
        -
        dna02_l_spikes
    )


    spike_total = (
        dna02_l_spikes
        +
        dna02_r_spikes
    )


    if spike_total > 0:

        spike_difference = (
            spike_count_difference
            /
            spike_total
        )

    else:

        spike_difference = 0.0


    # =====================================================
    # Frozen continuous DNa decoder
    # =====================================================

    left_activity = (
        dna02_l_v
        +
        dna02_l_spikes
        / SPIKE_SCALE
    )


    right_activity = (
        dna02_r_v
        +
        dna02_r_spikes
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

        "goal_column":
            goal_column,

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

        "membrane_difference":
            membrane_difference,

        "spike_count_difference":
            spike_count_difference,

        "spike_difference":
            spike_difference,

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
# Full 16 x 9 experiment
# =========================================================

rows = []


total_conditions = (
    len(
        EPG_ORDER
    )
    * 9
)


condition_number = 0


print()

print("=" * 110)
print(
    "BUILDING FINAL 16 x 9 FC2 POPULATION MAP"
)
print("=" * 110)

print()


for heading_column in EPG_ORDER:

    for goal_column in range(
        1,
        10,
    ):

        condition_number += 1


        print(
            f"[{condition_number:3d}/"
            f"{total_conditions}] "
            f"heading="
            f"{heading_column:>2} | "
            f"FC2=C{goal_column}"
        )


        rows.append(
            run_condition(
                heading_column=heading_column,
                goal_column=goal_column,
            )
        )


# =========================================================
# Results
# =========================================================

results = pd.DataFrame(
    rows
)


results.to_csv(
    RESULT_FILE,
    index=False,
)


# =========================================================
# Matrix builder
# =========================================================

def build_matrix(
    value_column,
):

    output = (
        results
        .pivot(
            index="heading_column",
            columns="goal_column",
            values=value_column,
        )
        .reindex(
            EPG_ORDER
        )
    )


    output.columns = [
        f"C{column}"

        for column
        in output.columns
    ]


    return output


# =========================================================
# Build matrices
# =========================================================

steering_matrix = build_matrix(
    "steering"
)


membrane_matrix = build_matrix(
    "membrane_difference"
)


spike_matrix = build_matrix(
    "spike_difference"
)


steering_matrix.to_csv(
    STEERING_MATRIX_FILE
)


membrane_matrix.to_csv(
    MEMBRANE_MATRIX_FILE
)


spike_matrix.to_csv(
    SPIKE_MATRIX_FILE
)


# =========================================================
# Steering matrix
# =========================================================

print()

print("=" * 120)
print(
    "FINAL CONTINUOUS STEERING MATRIX"
)
print("=" * 120)

print()


print(
    steering_matrix.to_string(
        float_format=lambda value:
        f"{value:+.3f}"
    )
)


# =========================================================
# Membrane diagnostic
# =========================================================

print()

print("=" * 120)
print(
    "DNa02 MEMBRANE DIFFERENCE MATRIX"
)
print("=" * 120)

print()


print(
    membrane_matrix.to_string(
        float_format=lambda value:
        f"{value:+.3f}"
    )
)


# =========================================================
# Spike diagnostic
# =========================================================

print()

print("=" * 120)
print(
    "DNa02 SPIKE DIFFERENCE MATRIX"
)
print("=" * 120)

print()


print(
    spike_matrix.to_string(
        float_format=lambda value:
        f"{value:+.3f}"
    )
)


# =========================================================
# Direction map
# =========================================================

NEUTRAL_THRESHOLD = 0.05


def direction_symbol(
    value,
):

    if value > NEUTRAL_THRESHOLD:

        return "R"


    if value < -NEUTRAL_THRESHOLD:

        return "L"


    return "."


direction_matrix = (
    steering_matrix.copy()
)


for column in direction_matrix.columns:

    direction_matrix[
        column
    ] = direction_matrix[
        column
    ].map(
        direction_symbol
    )


print()

print("=" * 120)
print(
    "FINAL DIRECTION MAP"
)
print("=" * 120)

print()


print(
    direction_matrix.to_string()
)


# =========================================================
# Network activity summary
# =========================================================

print()

print("=" * 120)
print(
    "NETWORK ACTIVITY SUMMARY"
)
print("=" * 120)

print()


condition_count = len(
    results
)


delta7_active = int(
    (
        results[
            "Delta7_spikes"
        ]
        > 0
    ).sum()
)


dna03_active = int(
    (
        results[
            "DNa03_spikes"
        ]
        > 0
    ).sum()
)


dna02_active = int(
    (
        results[
            "DNa02_spikes"
        ]
        > 0
    ).sum()
)


saturated_conditions = int(
    (
        results[
            "steering"
        ].abs()
        > 0.75
    ).sum()
)


strong_conditions = int(
    (
        results[
            "steering"
        ].abs()
        > 0.50
    ).sum()
)


neutral_conditions = int(
    (
        results[
            "steering"
        ].abs()
        < NEUTRAL_THRESHOLD
    ).sum()
)


print(
    f"Conditions: "
    f"{condition_count}"
)


print(
    f"Conditions with Delta7 spikes: "
    f"{delta7_active}/"
    f"{condition_count}"
)


print(
    f"Conditions with DNa03 spikes: "
    f"{dna03_active}/"
    f"{condition_count}"
)


print(
    f"Conditions with DNa02 spikes: "
    f"{dna02_active}/"
    f"{condition_count}"
)


print(
    f"Mean |membrane difference|: "
    f"{results['membrane_difference'].abs().mean():.4f}"
)


print(
    f"Mean |spike difference|: "
    f"{results['spike_difference'].abs().mean():.4f}"
)


print(
    f"Mean |activity difference|: "
    f"{results['activity_difference'].abs().mean():.4f}"
)


print(
    f"Mean |steering|: "
    f"{results['steering'].abs().mean():.4f}"
)


print(
    f"Max |steering|: "
    f"{results['steering'].abs().max():.4f}"
)


print(
    f"Saturated conditions "
    f"(|steering| > 0.75): "
    f"{saturated_conditions}"
)


print(
    f"Strong conditions "
    f"(|steering| > 0.50): "
    f"{strong_conditions}"
)


print(
    f"Neutral conditions "
    f"(|steering| < 0.05): "
    f"{neutral_conditions}"
)


# =========================================================
# Heading smoothness
# =========================================================

heading_rank = {
    heading: index

    for index, heading
    in enumerate(
        EPG_ORDER
    )
}


smoothness_rows = []


for goal_column in range(
    1,
    10,
):

    subset = results[
        results[
            "goal_column"
        ]
        == goal_column
    ].copy()


    subset[
        "heading_rank"
    ] = subset[
        "heading_column"
    ].map(
        heading_rank
    )


    subset = (
        subset
        .sort_values(
            "heading_rank"
        )
    )


    values = subset[
        "steering"
    ].to_numpy(
        dtype=float
    )


    next_values = np.roll(
        values,
        -1,
    )


    jumps = np.abs(
        next_values
        -
        values
    )


    smoothness_rows.append(
        {
            "goal_column":
                goal_column,

            "mean_adjacent_jump":
                float(
                    jumps.mean()
                ),

            "max_adjacent_jump":
                float(
                    jumps.max()
                ),
        }
    )


smoothness = pd.DataFrame(
    smoothness_rows
)


print()

print("=" * 120)
print(
    "HEADING SMOOTHNESS BY FC2 COLUMN"
)
print("=" * 120)

print()


print(
    smoothness.to_string(
        index=False,
        float_format=lambda value:
        f"{value:.4f}",
    )
)


print()


print(
    f"Overall mean adjacent jump: "
    f"{smoothness['mean_adjacent_jump'].mean():.4f}"
)


print(
    f"Worst adjacent jump: "
    f"{smoothness['max_adjacent_jump'].max():.4f}"
)


# =========================================================
# Strongest outputs
# =========================================================

display_columns = [
    "heading_column",
    "goal_column",
    "DNa03_L_spikes",
    "DNa03_R_spikes",
    "DNa02_L_spikes",
    "DNa02_R_spikes",
    "DNa02_L_v",
    "DNa02_R_v",
    "activity_difference",
    "steering",
]


print()

print("=" * 140)
print(
    "STRONGEST RIGHT CONDITIONS"
)
print("=" * 140)

print()


print(
    results[
        display_columns
    ]
    .sort_values(
        "steering",
        ascending=False,
    )
    .head(
        15
    )
    .to_string(
        index=False,
        float_format=lambda value:
        f"{value:+.5f}",
    )
)


print()

print("=" * 140)
print(
    "STRONGEST LEFT CONDITIONS"
)
print("=" * 140)

print()


print(
    results[
        display_columns
    ]
    .sort_values(
        "steering",
        ascending=True,
    )
    .head(
        15
    )
    .to_string(
        index=False,
        float_format=lambda value:
        f"{value:+.5f}",
    )
)


# =========================================================
# Files created
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
    STEERING_MATRIX_FILE
)

print(
    MEMBRANE_MATRIX_FILE
)

print(
    SPIKE_MATRIX_FILE
)