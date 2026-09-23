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

STEP31_RESULT_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment.csv"
)

CONDITION_FILE = (
    OUTPUT_DIR
    / "shadow_dna_condition_metrics.csv"
)

TRANSITION_FILE = (
    OUTPUT_DIR
    / "shadow_dna_transition_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "shadow_dna_signal_summary.csv"
)


# =========================================================
# Frozen surrogate parameters
#
# DO NOT tune anything in Step 34.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN = 3.0

SPIKE_SCALE = 5.0
ACTIVITY_SCALE = 1.10599

BASELINE_DRIVE = 0.03
EPG_DRIVE = 1.35
FC2_DRIVE = 1.35

SIMULATION_TIME = 300 * ms
STATE_MONITOR_DT = 1 * ms


# =========================================================
# Shadow population layout
#
# 0 = shadow DNa03_L
# 1 = shadow DNa03_R
# 2 = shadow DNa02_L
# 3 = shadow DNa02_R
#
# These neurons:
#
# - receive copied incoming events
# - use the same 20 ms membrane leak
# - receive the same baseline drive
# - NEVER spike
# - NEVER reset
# - NEVER project back into the real network
#
# They are passive observers only.
# =========================================================

SHADOW_DNA03_L = 0
SHADOW_DNA03_R = 1
SHADOW_DNA02_L = 2
SHADOW_DNA02_R = 3

N_SHADOW_NEURONS = 4


# =========================================================
# Load Step 31 exact conditions
# =========================================================

if not STEP31_RESULT_FILE.exists():

    raise FileNotFoundError(
        "Step 31 result file not found:\n"
        f"{STEP31_RESULT_FILE}\n\n"
        "Run Step 31 first."
    )


step31 = pd.read_csv(
    STEP31_RESULT_FILE
)


REQUIRED_STEP31_COLUMNS = [
    "heading_column",
    "column_a",
    "column_b",
    "alpha",
    "a_body_ids",
    "b_body_ids",
    "steering",
]


missing_columns = [
    column
    for column in REQUIRED_STEP31_COLUMNS
    if column not in step31.columns
]


if missing_columns:

    raise RuntimeError(
        "Step 31 result file is missing columns: "
        f"{missing_columns}"
    )


if len(step31) != 396:

    raise RuntimeError(
        f"Expected 396 Step 31 conditions, "
        f"got {len(step31)}."
    )


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
# Normalize IDs / weights
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
# Frozen network populations
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
        neurons["type"].isin(
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
# Frozen duplicate handling
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
# Frozen network checks
# =========================================================

if len(neurons) != 220:

    raise RuntimeError(
        f"Expected 220 neurons, "
        f"got {len(neurons)}."
    )


if len(edges) != 3231:

    raise RuntimeError(
        f"Expected 3231 functional edges, "
        f"got {len(edges)}."
    )


print("=" * 120)
print(
    "STEP 34 - SHADOW DNa DYNAMICS TEST"
)
print("=" * 120)

print()

print(
    f"Real neurons: "
    f"{len(neurons)}"
)

print(
    f"Functional edges: "
    f"{len(edges)}"
)

print(
    f"Passive shadow DNa neurons: "
    f"{N_SHADOW_NEURONS}"
)

print(
    f"Step 31 conditions: "
    f"{len(step31)}"
)

print(
    f"Simulation time: "
    f"{float(SIMULATION_TIME / ms):.1f} ms"
)

print(
    f"State sampling: "
    f"{float(STATE_MONITOR_DT / ms):.1f} ms"
)


# =========================================================
# Main network index
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
# Helpers
# =========================================================

def normalize_side(
    value,
):

    if pd.isna(
        value
    ):

        return ""


    return str(
        value
    ).strip().upper()


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

            row_side = normalize_side(
                row.get(
                    "somaSide",
                    "",
                )
            )


            if (
                row_side
                !=
                str(
                    side
                ).upper()
            ):

                continue


        output.append(
            index
        )


    return np.asarray(
        output,
        dtype=int,
    )


def parse_body_ids(
    value,
):

    if pd.isna(
        value
    ):

        return []


    text = str(
        value
    ).strip()


    if not text:

        return []


    output = []


    for piece in text.split(
        ","
    ):

        piece = piece.strip()


        if not piece:

            continue


        output.append(
            int(
                piece
            )
        )


    return output


# =========================================================
# Identify motor neurons
# =========================================================

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


if len(
    dna03_indices
) != 2:

    raise RuntimeError(
        f"Expected 2 DNa03 neurons, "
        f"got {len(dna03_indices)}."
    )


if len(
    dna02_indices
) != 2:

    raise RuntimeError(
        f"Expected 2 DNa02 neurons, "
        f"got {len(dna02_indices)}."
    )


if (
    len(dna03_left) != 1
    or
    len(dna03_right) != 1
    or
    len(dna02_left) != 1
    or
    len(dna02_right) != 1
):

    raise RuntimeError(
        "Expected exactly one left/right "
        "DNa03 and DNa02 neuron."
    )


real_dna03_left_index = int(
    dna03_left[
        0
    ]
)

real_dna03_right_index = int(
    dna03_right[
        0
    ]
)

real_dna02_left_index = int(
    dna02_left[
        0
    ]
)

real_dna02_right_index = int(
    dna02_right[
        0
    ]
)


# =========================================================
# Real DNa target -> shadow DNa target
# =========================================================

real_target_to_shadow = {

    real_dna03_left_index:
        SHADOW_DNA03_L,

    real_dna03_right_index:
        SHADOW_DNA03_R,

    real_dna02_left_index:
        SHADOW_DNA02_L,

    real_dna02_right_index:
        SHADOW_DNA02_R,
}


print()

print("=" * 120)
print(
    "REAL / SHADOW MOTOR MAP"
)
print("=" * 120)

print()

print(
    f"Real DNa03_L index "
    f"{real_dna03_left_index} "
    f"-> shadow {SHADOW_DNA03_L}"
)

print(
    f"Real DNa03_R index "
    f"{real_dna03_right_index} "
    f"-> shadow {SHADOW_DNA03_R}"
)

print(
    f"Real DNa02_L index "
    f"{real_dna02_left_index} "
    f"-> shadow {SHADOW_DNA02_L}"
)

print(
    f"Real DNa02_R index "
    f"{real_dna02_right_index} "
    f"-> shadow {SHADOW_DNA02_R}"
)


# =========================================================
# EPG parser
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


if sum(
    len(
        indices
    )

    for indices
    in epg_groups.values()
) != 46:

    raise RuntimeError(
        "Expected 46 grouped EPG neurons."
    )


# =========================================================
# Functional pathway masks
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
        "Functional edge accounting failed: "
        f"{accounted_edges} "
        f"!= {len(edges)}"
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
# Convert edge tables to Brian2 arrays
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
# Frozen weight scaling
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
            values

            for values
            in raw_arrays

            if len(
                values
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
        *
        np.sqrt(
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
# Map real DNa targets to passive shadow targets
# =========================================================

def map_to_shadow_targets(
    real_target_indices,
):

    output = []


    for real_target in real_target_indices:

        real_target = int(
            real_target
        )


        if (
            real_target
            not in
            real_target_to_shadow
        ):

            raise RuntimeError(
                "Motor edge target does not map "
                f"to shadow DNa: {real_target}"
            )


        output.append(
            real_target_to_shadow[
                real_target
            ]
        )


    return output


shadow_pfl3_dna02_target = (
    map_to_shadow_targets(
        pfl3_dna02_target
    )
)

shadow_pfl3_dna03_target = (
    map_to_shadow_targets(
        pfl3_dna03_target
    )
)

shadow_pfl2_dna03_target = (
    map_to_shadow_targets(
        pfl2_dna03_target
    )
)

shadow_pfl2_dna02_target = (
    map_to_shadow_targets(
        pfl2_dna02_target
    )
)

shadow_dna03_dna02_target = (
    map_to_shadow_targets(
        dna03_dna02_target
    )
)


# =========================================================
# Main-network positive synapses
# =========================================================

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
# Passive shadow synapses
#
# input_sum accumulates every weighted event without leak.
#
# shadow v receives the same jump but then leaks with the
# same 20 ms membrane equation as the real DNa neurons.
#
# Critically: there is no threshold/reset.
# =========================================================

def create_shadow_synapses(
    real_group,
    shadow_group,
    source,
    shadow_target,
    weights,
    gain=1.0,
):

    if len(
        source
    ) == 0:

        return None


    synapses = Synapses(
        real_group,
        shadow_group,
        model="w : 1",
        on_pre="""
        v_post += w
        input_sum_post += w
        """,
    )


    synapses.connect(
        i=source,
        j=shadow_target,
    )


    synapses.w = (
        weights
        *
        gain
    )


    return synapses


# =========================================================
# Run one exact Step 31 condition
# =========================================================

def run_condition(
    heading_column,
    active_fc2_body_ids,
):

    # =====================================================
    # REAL frozen network
    # =====================================================

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
    group.drive = BASELINE_DRIVE


    # =====================================================
    # PASSIVE shadow motor neurons
    # =====================================================

    shadow_equations = """
    dv/dt = (-v + drive) / (20*ms) : 1
    drive : 1
    input_sum : 1
    """


    shadow = NeuronGroup(
        N_SHADOW_NEURONS,
        shadow_equations,
        method="euler",
    )


    shadow.v = 0.0
    shadow.drive = BASELINE_DRIVE
    shadow.input_sum = 0.0


    # =====================================================
    # EPG heading
    # =====================================================

    if heading_column not in epg_groups:

        raise RuntimeError(
            f"Unknown heading: "
            f"{heading_column}"
        )


    for index in epg_groups[
        heading_column
    ]:

        group.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # Exact Step 31 FC2 subset
    # =====================================================

    active_fc2_body_ids = sorted(
        set(
            active_fc2_body_ids
        )
    )


    for body_id in active_fc2_body_ids:

        if body_id not in body_to_index:

            raise RuntimeError(
                "Unknown FC2 body ID: "
                f"{body_id}"
            )


        network_index = body_to_index[
            body_id
        ]


        neuron_type = str(
            neurons.loc[
                network_index,
                "type",
            ]
        )


        if neuron_type not in [
            "FC2A",
            "FC2B",
            "FC2C",
        ]:

            raise RuntimeError(
                f"Body ID {body_id} "
                f"is {neuron_type}, not FC2."
            )


        group.drive[
            network_index
        ] = FC2_DRIVE


    # =====================================================
    # Build objects
    # =====================================================

    objects = [
        group,
        shadow,
    ]


    # =====================================================
    # REAL: EPG / FC2 -> PFL
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
    # REAL: EPG -> Delta7
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
    # REAL: Delta7 -> PFL inhibitory hypothesis
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
    # REAL: PFL -> DNa pathways
    # =====================================================

    real_motor_routes = [

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
    ) in real_motor_routes:

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
    # REAL: DNa03 -> DNa02
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
    # SHADOW DNa03
    #
    # Receives the SAME PFL3 / PFL2 spike events as
    # real DNa03 at the SAME weights and 3x gain.
    # =====================================================

    shadow_synapses = create_shadow_synapses(
        group,
        shadow,
        pfl3_dna03_source,
        shadow_pfl3_dna03_target,
        pfl3_dna03_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if shadow_synapses is not None:

        objects.append(
            shadow_synapses
        )


    shadow_synapses = create_shadow_synapses(
        group,
        shadow,
        pfl2_dna03_source,
        shadow_pfl2_dna03_target,
        pfl2_dna03_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if shadow_synapses is not None:

        objects.append(
            shadow_synapses
        )


    # =====================================================
    # SHADOW DNa02 direct PFL input
    #
    # Same direct PFL3/PFL2 events as real DNa02.
    # =====================================================

    shadow_synapses = create_shadow_synapses(
        group,
        shadow,
        pfl3_dna02_source,
        shadow_pfl3_dna02_target,
        pfl3_dna02_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if shadow_synapses is not None:

        objects.append(
            shadow_synapses
        )


    shadow_synapses = create_shadow_synapses(
        group,
        shadow,
        pfl2_dna02_source,
        shadow_pfl2_dna02_target,
        pfl2_dna02_weights,
        gain=PFL_TO_DNA_GAIN,
    )


    if shadow_synapses is not None:

        objects.append(
            shadow_synapses
        )


    # =====================================================
    # SHADOW DNa02 receives REAL DNa03 spikes
    #
    # This is intentional.
    #
    # We want the shadow DNa02 observer to see exactly the
    # same incoming DNa03 relay events as the real DNa02.
    #
    # We do NOT use shadow DNa03 as a presynaptic source,
    # because it does not spike.
    # =====================================================

    shadow_synapses = create_shadow_synapses(
        group,
        shadow,
        dna03_dna02_source,
        shadow_dna03_dna02_target,
        dna03_dna02_weights,
        gain=1.0,
    )


    if shadow_synapses is not None:

        objects.append(
            shadow_synapses
        )


    # =====================================================
    # Monitors
    # =====================================================

    spike_monitor = SpikeMonitor(
        group
    )


    main_state_monitor = StateMonitor(
        group,
        "v",
        record=[
            real_dna03_left_index,
            real_dna03_right_index,
            real_dna02_left_index,
            real_dna02_right_index,
        ],
        dt=STATE_MONITOR_DT,
    )


    shadow_state_monitor = StateMonitor(
        shadow,
        "v",
        record=True,
        dt=STATE_MONITOR_DT,
    )


    objects.extend(
        [
            spike_monitor,
            main_state_monitor,
            shadow_state_monitor,
        ]
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


    # =====================================================
    # Real network state
    # =====================================================

    counts = np.asarray(
        spike_monitor.count,
        dtype=float,
    )


    final_voltage = np.asarray(
        group.v,
        dtype=float,
    )


    real_v_trace = np.asarray(
        main_state_monitor.v,
        dtype=float,
    )


    shadow_v_trace = np.asarray(
        shadow_state_monitor.v,
        dtype=float,
    )


    duration_seconds = float(
        SIMULATION_TIME
        /
        (1000 * ms)
    )


    # =====================================================
    # Real DNa rates
    # =====================================================

    real_dna03_l_spikes = float(
        counts[
            real_dna03_left_index
        ]
    )

    real_dna03_r_spikes = float(
        counts[
            real_dna03_right_index
        ]
    )


    real_dna02_l_spikes = float(
        counts[
            real_dna02_left_index
        ]
    )

    real_dna02_r_spikes = float(
        counts[
            real_dna02_right_index
        ]
    )


    real_dna03_l_rate = (
        real_dna03_l_spikes
        /
        duration_seconds
    )

    real_dna03_r_rate = (
        real_dna03_r_spikes
        /
        duration_seconds
    )


    real_dna02_l_rate = (
        real_dna02_l_spikes
        /
        duration_seconds
    )

    real_dna02_r_rate = (
        real_dna02_r_spikes
        /
        duration_seconds
    )


    real_dna03_rate_diff = (
        real_dna03_r_rate
        -
        real_dna03_l_rate
    )


    real_dna02_rate_diff = (
        real_dna02_r_rate
        -
        real_dna02_l_rate
    )


    # =====================================================
    # Real DNa mean membrane over 300 ms
    #
    # StateMonitor order:
    #
    # 0 = DNa03_L
    # 1 = DNa03_R
    # 2 = DNa02_L
    # 3 = DNa02_R
    # =====================================================

    real_dna03_l_mean_v = float(
        real_v_trace[
            0
        ].mean()
    )

    real_dna03_r_mean_v = float(
        real_v_trace[
            1
        ].mean()
    )


    real_dna02_l_mean_v = float(
        real_v_trace[
            2
        ].mean()
    )

    real_dna02_r_mean_v = float(
        real_v_trace[
            3
        ].mean()
    )


    real_dna03_mean_v_diff = (
        real_dna03_r_mean_v
        -
        real_dna03_l_mean_v
    )


    real_dna02_mean_v_diff = (
        real_dna02_r_mean_v
        -
        real_dna02_l_mean_v
    )


    # =====================================================
    # Passive SHADOW DNa mean membrane
    # =====================================================

    shadow_dna03_l_mean_v = float(
        shadow_v_trace[
            SHADOW_DNA03_L
        ].mean()
    )

    shadow_dna03_r_mean_v = float(
        shadow_v_trace[
            SHADOW_DNA03_R
        ].mean()
    )


    shadow_dna02_l_mean_v = float(
        shadow_v_trace[
            SHADOW_DNA02_L
        ].mean()
    )

    shadow_dna02_r_mean_v = float(
        shadow_v_trace[
            SHADOW_DNA02_R
        ].mean()
    )


    shadow_dna03_mean_v_diff = (
        shadow_dna03_r_mean_v
        -
        shadow_dna03_l_mean_v
    )


    shadow_dna02_mean_v_diff = (
        shadow_dna02_r_mean_v
        -
        shadow_dna02_l_mean_v
    )


    # =====================================================
    # Passive SHADOW final voltage
    # =====================================================

    shadow_final_v = np.asarray(
        shadow.v,
        dtype=float,
    )


    shadow_dna03_final_v_diff = (
        float(
            shadow_final_v[
                SHADOW_DNA03_R
            ]
        )
        -
        float(
            shadow_final_v[
                SHADOW_DNA03_L
            ]
        )
    )


    shadow_dna02_final_v_diff = (
        float(
            shadow_final_v[
                SHADOW_DNA02_R
            ]
        )
        -
        float(
            shadow_final_v[
                SHADOW_DNA02_L
            ]
        )
    )


    # =====================================================
    # Cumulative incoming weighted event totals
    #
    # No leak / reset / threshold.
    #
    # This is the cleanest measurement of how much
    # synaptic event input arrived on each side.
    # =====================================================

    shadow_input_sum = np.asarray(
        shadow.input_sum,
        dtype=float,
    )


    shadow_dna03_input_l = float(
        shadow_input_sum[
            SHADOW_DNA03_L
        ]
    )

    shadow_dna03_input_r = float(
        shadow_input_sum[
            SHADOW_DNA03_R
        ]
    )


    shadow_dna02_input_l = float(
        shadow_input_sum[
            SHADOW_DNA02_L
        ]
    )

    shadow_dna02_input_r = float(
        shadow_input_sum[
            SHADOW_DNA02_R
        ]
    )


    shadow_dna03_input_diff = (
        shadow_dna03_input_r
        -
        shadow_dna03_input_l
    )


    shadow_dna02_input_diff = (
        shadow_dna02_input_r
        -
        shadow_dna02_input_l
    )


    # =====================================================
    # Existing frozen steering decoder
    # =====================================================

    left_final_v = float(
        final_voltage[
            real_dna02_left_index
        ]
    )


    right_final_v = float(
        final_voltage[
            real_dna02_right_index
        ]
    )


    left_activity = (

        left_final_v

        +

        real_dna02_l_spikes
        /
        SPIKE_SCALE
    )


    right_activity = (

        right_final_v

        +

        real_dna02_r_spikes
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


    return {

        "active_fc2_count":
            len(
                active_fc2_body_ids
            ),

        # ---------------------------------------------
        # Existing motor output
        # ---------------------------------------------

        "steering":
            steering,

        "activity_difference":
            activity_difference,

        # ---------------------------------------------
        # REAL DNa03
        # ---------------------------------------------

        "real_DNa03_L_spikes":
            real_dna03_l_spikes,

        "real_DNa03_R_spikes":
            real_dna03_r_spikes,

        "real_DNa03_rate_diff":
            real_dna03_rate_diff,

        "real_DNa03_mean_v_diff":
            real_dna03_mean_v_diff,

        # ---------------------------------------------
        # SHADOW DNa03
        # ---------------------------------------------

        "shadow_DNa03_mean_v_diff":
            shadow_dna03_mean_v_diff,

        "shadow_DNa03_final_v_diff":
            shadow_dna03_final_v_diff,

        "shadow_DNa03_input_L":
            shadow_dna03_input_l,

        "shadow_DNa03_input_R":
            shadow_dna03_input_r,

        "shadow_DNa03_input_diff":
            shadow_dna03_input_diff,

        # ---------------------------------------------
        # REAL DNa02
        # ---------------------------------------------

        "real_DNa02_L_spikes":
            real_dna02_l_spikes,

        "real_DNa02_R_spikes":
            real_dna02_r_spikes,

        "real_DNa02_rate_diff":
            real_dna02_rate_diff,

        "real_DNa02_mean_v_diff":
            real_dna02_mean_v_diff,

        # ---------------------------------------------
        # SHADOW DNa02
        # ---------------------------------------------

        "shadow_DNa02_mean_v_diff":
            shadow_dna02_mean_v_diff,

        "shadow_DNa02_final_v_diff":
            shadow_dna02_final_v_diff,

        "shadow_DNa02_input_L":
            shadow_dna02_input_l,

        "shadow_DNa02_input_R":
            shadow_dna02_input_r,

        "shadow_DNa02_input_diff":
            shadow_dna02_input_diff,
    }


# =========================================================
# Run exact Step 31 conditions
# =========================================================

rows = []


print()

print("=" * 120)
print(
    "RERUNNING EXACT STEP 31 CONDITIONS"
)
print("=" * 120)

print()


for condition_number, source_row in step31.iterrows():

    heading = str(
        source_row[
            "heading_column"
        ]
    )


    column_a = int(
        source_row[
            "column_a"
        ]
    )


    column_b = int(
        source_row[
            "column_b"
        ]
    )


    alpha = float(
        source_row[
            "alpha"
        ]
    )


    a_body_ids = parse_body_ids(
        source_row[
            "a_body_ids"
        ]
    )


    b_body_ids = parse_body_ids(
        source_row[
            "b_body_ids"
        ]
    )


    active_fc2_body_ids = (
        a_body_ids
        +
        b_body_ids
    )


    print(
        f"[{condition_number + 1:3d}/"
        f"{len(step31)}] "
        f"heading={heading:>2} | "
        f"C{column_a}->C{column_b} | "
        f"alpha={alpha:.1f} | "
        f"FC2={len(active_fc2_body_ids)}"
    )


    diagnostic = run_condition(
        heading_column=heading,
        active_fc2_body_ids=active_fc2_body_ids,
    )


    step31_steering = float(
        source_row[
            "steering"
        ]
    )


    reproduced_steering = float(
        diagnostic[
            "steering"
        ]
    )


    rows.append(
        {

            "condition_id":
                len(
                    rows
                ),

            "heading_column":
                heading,

            "column_a":
                column_a,

            "column_b":
                column_b,

            "alpha":
                alpha,

            "source_step31_steering":
                step31_steering,

            "reproduction_error":
                abs(
                    reproduced_steering
                    -
                    step31_steering
                ),

            **diagnostic,
        }
    )


conditions = pd.DataFrame(
    rows
)


conditions.to_csv(
    CONDITION_FILE,
    index=False,
)


# =========================================================
# Reproduction check
# =========================================================

print()

print("=" * 120)
print(
    "STEP 31 REPRODUCTION CHECK"
)
print("=" * 120)

print()


mean_error = float(
    conditions[
        "reproduction_error"
    ].mean()
)


max_error = float(
    conditions[
        "reproduction_error"
    ].max()
)


print(
    f"Conditions checked: "
    f"{len(conditions)}"
)

print(
    f"Mean steering error: "
    f"{mean_error:.12f}"
)

print(
    f"Max steering error: "
    f"{max_error:.12f}"
)


if max_error <= 1e-8:

    print(
        "PASS: passive shadow neurons do not "
        "alter frozen Step 31 dynamics."
    )


else:

    print(
        "WARNING: Step 34 does not reproduce "
        "Step 31 exactly."
    )


# =========================================================
# Signals to compare
# =========================================================

SIGNALS = {

    "real_DNa03_rate":
        {
            "column":
                "real_DNa03_rate_diff",

            "stage":
                "DNa03",

            "kind":
                "real_spike_rate",
        },

    "real_DNa03_meanV":
        {
            "column":
                "real_DNa03_mean_v_diff",

            "stage":
                "DNa03",

            "kind":
                "real_reset_membrane",
        },

    "shadow_DNa03_meanV":
        {
            "column":
                "shadow_DNa03_mean_v_diff",

            "stage":
                "DNa03",

            "kind":
                "shadow_no_threshold",
        },

    "shadow_DNa03_input":
        {
            "column":
                "shadow_DNa03_input_diff",

            "stage":
                "DNa03",

            "kind":
                "cumulative_input",
        },

    "real_DNa02_rate":
        {
            "column":
                "real_DNa02_rate_diff",

            "stage":
                "DNa02",

            "kind":
                "real_spike_rate",
        },

    "real_DNa02_meanV":
        {
            "column":
                "real_DNa02_mean_v_diff",

            "stage":
                "DNa02",

            "kind":
                "real_reset_membrane",
        },

    "shadow_DNa02_meanV":
        {
            "column":
                "shadow_DNa02_mean_v_diff",

            "stage":
                "DNa02",

            "kind":
                "shadow_no_threshold",
        },

    "shadow_DNa02_input":
        {
            "column":
                "shadow_DNa02_input_diff",

            "stage":
                "DNa02",

            "kind":
                "cumulative_input",
        },
}


# =========================================================
# Characteristic scales
#
# p95 absolute magnitude over all conditions.
#
# This only makes different signal units comparable.
# It does NOT alter the network.
# =========================================================

signal_scales = {}


print()

print("=" * 120)
print(
    "SIGNAL CHARACTERISTIC SCALES"
)
print("=" * 120)

print()


for signal_name, metadata in SIGNALS.items():

    column = metadata[
        "column"
    ]


    values = conditions[
        column
    ].to_numpy(
        dtype=float,
    )


    finite_values = values[
        np.isfinite(
            values
        )
    ]


    scale = float(
        np.percentile(
            np.abs(
                finite_values
            ),
            95.0,
        )
    )


    if scale < 1e-12:

        scale = float(
            np.max(
                np.abs(
                    finite_values
                )
            )
        )


    if scale < 1e-12:

        scale = 1.0


    signal_scales[
        signal_name
    ] = scale


    print(
        f"{signal_name:24s} "
        f"| scale={scale:.6f}"
    )


# =========================================================
# Adjacent-alpha transition metrics
# =========================================================

transition_rows = []


for (
    heading,
    column_a,
    column_b,
), group in conditions.groupby(
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
        .reset_index(
            drop=True
        )
    )


    for index in range(
        len(
            group
        ) - 1
    ):

        first = group.iloc[
            index
        ]

        second = group.iloc[
            index + 1
        ]


        steering_step = abs(
            float(
                second[
                    "steering"
                ]
            )
            -
            float(
                first[
                    "steering"
                ]
            )
        )


        record = {

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

            "alpha_from":
                float(
                    first[
                        "alpha"
                    ]
                ),

            "alpha_to":
                float(
                    second[
                        "alpha"
                    ]
                ),

            "steering_from":
                float(
                    first[
                        "steering"
                    ]
                ),

            "steering_to":
                float(
                    second[
                        "steering"
                    ]
                ),

            "steering_abs_step":
                steering_step,
        }


        for signal_name, metadata in SIGNALS.items():

            column = metadata[
                "column"
            ]


            first_value = float(
                first[
                    column
                ]
            )

            second_value = float(
                second[
                    column
                ]
            )


            raw_step = abs(
                second_value
                -
                first_value
            )


            normalized_step = (
                raw_step
                /
                signal_scales[
                    signal_name
                ]
            )


            sign_flip = int(
                (
                    abs(
                        first_value
                    ) > 1e-9
                )
                and
                (
                    abs(
                        second_value
                    ) > 1e-9
                )
                and
                (
                    np.sign(
                        first_value
                    )
                    !=
                    np.sign(
                        second_value
                    )
                )
            )


            record[
                f"{signal_name}_raw_step"
            ] = raw_step


            record[
                f"{signal_name}_norm_step"
            ] = normalized_step


            record[
                f"{signal_name}_sign_flip"
            ] = sign_flip


        transition_rows.append(
            record
        )


transitions = pd.DataFrame(
    transition_rows
)


transitions.to_csv(
    TRANSITION_FILE,
    index=False,
)


# =========================================================
# Signal summary
# =========================================================

summary_rows = []


steering_steps = transitions[
    "steering_abs_step"
].to_numpy(
    dtype=float,
)


for signal_name, metadata in SIGNALS.items():

    normalized_steps = transitions[
        f"{signal_name}_norm_step"
    ].to_numpy(
        dtype=float,
    )


    raw_steps = transitions[
        f"{signal_name}_raw_step"
    ].to_numpy(
        dtype=float,
    )


    sign_flips = int(
        transitions[
            f"{signal_name}_sign_flip"
        ].sum()
    )


    if (
        np.std(
            raw_steps
        ) > 1e-12
        and
        np.std(
            steering_steps
        ) > 1e-12
    ):

        correlation = float(
            np.corrcoef(
                raw_steps,
                steering_steps,
            )[
                0,
                1
            ]
        )


    else:

        correlation = np.nan


    summary_rows.append(
        {

            "signal":
                signal_name,

            "stage":
                metadata[
                    "stage"
                ],

            "kind":
                metadata[
                    "kind"
                ],

            "characteristic_scale":
                signal_scales[
                    signal_name
                ],

            "mean_normalized_step":
                float(
                    normalized_steps.mean()
                ),

            "median_normalized_step":
                float(
                    np.median(
                        normalized_steps
                    )
                ),

            "p95_normalized_step":
                float(
                    np.percentile(
                        normalized_steps,
                        95.0,
                    )
                ),

            "max_normalized_step":
                float(
                    normalized_steps.max()
                ),

            "adjacent_sign_flips":
                sign_flips,

            "steps_gt_0_25":
                int(
                    (
                        normalized_steps
                        >
                        0.25
                    ).sum()
                ),

            "steps_gt_0_50":
                int(
                    (
                        normalized_steps
                        >
                        0.50
                    ).sum()
                ),

            "correlation_with_steering_jump":
                correlation,
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
# Print summary
# =========================================================

print()

print("=" * 190)
print(
    "SHADOW VS REAL MOTOR-STAGE SUMMARY"
)
print("=" * 190)

print()


print(
    summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Identify current steering decoder worst transition
# =========================================================

worst_transition = (
    transitions
    .sort_values(
        "steering_abs_step",
        ascending=False,
    )
    .iloc[
        0
    ]
)


worst_heading = str(
    worst_transition[
        "heading_column"
    ]
)

worst_a = int(
    worst_transition[
        "column_a"
    ]
)

worst_b = int(
    worst_transition[
        "column_b"
    ]
)

worst_alpha_from = float(
    worst_transition[
        "alpha_from"
    ]
)

worst_alpha_to = float(
    worst_transition[
        "alpha_to"
    ]
)


# =========================================================
# Exact worst transition
# =========================================================

print()

print("=" * 150)
print(
    "CURRENT STEERING WORST TRANSITION"
)
print("=" * 150)

print()


print(
    f"Heading: "
    f"{worst_heading}"
)

print(
    f"FC2 pair: "
    f"C{worst_a}->C{worst_b}"
)

print(
    f"Alpha: "
    f"{worst_alpha_from:.1f} "
    f"-> "
    f"{worst_alpha_to:.1f}"
)

print(
    f"Steering: "
    f"{float(worst_transition['steering_from']):+.4f} "
    f"-> "
    f"{float(worst_transition['steering_to']):+.4f}"
)

print(
    f"|steering jump|: "
    f"{float(worst_transition['steering_abs_step']):.4f}"
)


# =========================================================
# Layer-by-layer jump on worst transition
# =========================================================

worst_rows = []


for signal_name, metadata in SIGNALS.items():

    worst_rows.append(
        {

            "signal":
                signal_name,

            "stage":
                metadata[
                    "stage"
                ],

            "kind":
                metadata[
                    "kind"
                ],

            "raw_step":
                float(
                    worst_transition[
                        f"{signal_name}_raw_step"
                    ]
                ),

            "normalized_step":
                float(
                    worst_transition[
                        f"{signal_name}_norm_step"
                    ]
                ),

            "sign_flip":
                int(
                    worst_transition[
                        f"{signal_name}_sign_flip"
                    ]
                ),
        }
    )


worst_signal_table = pd.DataFrame(
    worst_rows
)


print()

print("=" * 160)
print(
    "REAL VS SHADOW SIGNAL JUMPS ON WORST TRANSITION"
)
print("=" * 160)

print()


print(
    worst_signal_table.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Raw states either side of worst transition
# =========================================================

worst_conditions = conditions[
    (
        conditions[
            "heading_column"
        ]
        ==
        worst_heading
    )
    &
    (
        conditions[
            "column_a"
        ]
        ==
        worst_a
    )
    &
    (
        conditions[
            "column_b"
        ]
        ==
        worst_b
    )
    &
    (
        conditions[
            "alpha"
        ].isin(
            [
                worst_alpha_from,
                worst_alpha_to,
            ]
        )
    )
].sort_values(
    "alpha"
)


detail_columns = [
    "alpha",
    "active_fc2_count",

    "real_DNa03_L_spikes",
    "real_DNa03_R_spikes",
    "real_DNa03_rate_diff",
    "real_DNa03_mean_v_diff",

    "shadow_DNa03_mean_v_diff",
    "shadow_DNa03_final_v_diff",
    "shadow_DNa03_input_L",
    "shadow_DNa03_input_R",
    "shadow_DNa03_input_diff",

    "real_DNa02_L_spikes",
    "real_DNa02_R_spikes",
    "real_DNa02_rate_diff",
    "real_DNa02_mean_v_diff",

    "shadow_DNa02_mean_v_diff",
    "shadow_DNa02_final_v_diff",
    "shadow_DNa02_input_L",
    "shadow_DNa02_input_R",
    "shadow_DNa02_input_diff",

    "activity_difference",
    "steering",
]


print()

print("=" * 240)
print(
    "RAW REAL / SHADOW STATES AROUND WORST TRANSITION"
)
print("=" * 240)

print()


print(
    worst_conditions[
        detail_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.6f}",
    )
)


# =========================================================
# Full trajectory for current worst pair
# =========================================================

worst_trajectory = conditions[
    (
        conditions[
            "heading_column"
        ]
        ==
        worst_heading
    )
    &
    (
        conditions[
            "column_a"
        ]
        ==
        worst_a
    )
    &
    (
        conditions[
            "column_b"
        ]
        ==
        worst_b
    )
].sort_values(
    "alpha"
)


trajectory_columns = [
    "alpha",

    "real_DNa03_rate_diff",
    "shadow_DNa03_mean_v_diff",
    "shadow_DNa03_input_diff",

    "real_DNa02_rate_diff",
    "shadow_DNa02_mean_v_diff",
    "shadow_DNa02_input_diff",

    "steering",
]


print()

print("=" * 190)
print(
    "FULL WORST TRAJECTORY - REAL VS SHADOW"
)
print("=" * 190)

print()


print(
    worst_trajectory[
        trajectory_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.6f}",
    )
)


# =========================================================
# Worst transition for each signal
# =========================================================

signal_worst_rows = []


for signal_name, metadata in SIGNALS.items():

    signal_column = (
        f"{signal_name}_norm_step"
    )


    signal_worst = (
        transitions
        .sort_values(
            signal_column,
            ascending=False,
        )
        .iloc[
            0
        ]
    )


    signal_worst_rows.append(
        {

            "signal":
                signal_name,

            "stage":
                metadata[
                    "stage"
                ],

            "kind":
                metadata[
                    "kind"
                ],

            "heading":
                str(
                    signal_worst[
                        "heading_column"
                    ]
                ),

            "pair":
                (
                    f"C"
                    f"{int(signal_worst['column_a'])}"
                    f"->C"
                    f"{int(signal_worst['column_b'])}"
                ),

            "alpha_from":
                float(
                    signal_worst[
                        "alpha_from"
                    ]
                ),

            "alpha_to":
                float(
                    signal_worst[
                        "alpha_to"
                    ]
                ),

            "normalized_step":
                float(
                    signal_worst[
                        signal_column
                    ]
                ),

            "steering_jump":
                float(
                    signal_worst[
                        "steering_abs_step"
                    ]
                ),
        }
    )


signal_worst_table = pd.DataFrame(
    signal_worst_rows
)


print()

print("=" * 180)
print(
    "WORST TRANSITION PER REAL / SHADOW SIGNAL"
)
print("=" * 180)

print()


print(
    signal_worst_table.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
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
    CONDITION_FILE
)

print(
    TRANSITION_FILE
)

print(
    SUMMARY_FILE
)