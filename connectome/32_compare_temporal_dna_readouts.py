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


# Exact Step 31 conditions.
STEP31_RESULT_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment.csv"
)


RESULT_FILE = (
    OUTPUT_DIR
    / "dna_temporal_readout_conditions.csv"
)

TRAJECTORY_FILE = (
    OUTPUT_DIR
    / "dna_temporal_readout_trajectories.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "dna_temporal_readout_comparison.csv"
)


# =========================================================
# Frozen surrogate parameters
#
# DO NOT tune these in Step 32.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0

PFL_TO_DNA_GAIN = 3.0


# =========================================================
# Existing Step 27 / Step 31 decoder
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
# Temporal diagnostic settings
# =========================================================

# Record DNa02 state every 1 ms.
STATE_MONITOR_DT = 1 * ms

# Terminal window used for "recent activity".
TERMINAL_WINDOW_MS = 100.0

# Passive output trace.
#
# This variable receives +1 whenever a neuron spikes and
# decays exponentially.
#
# IMPORTANT:
# It does NOT feed back into v and therefore does not alter
# the frozen neural dynamics.
OUTPUT_TRACE_TAU = 50 * ms


# =========================================================
# Load exact Step 31 conditions
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


required_step31_columns = [
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

    for column
    in required_step31_columns

    if column
    not in step31.columns
]


if missing_columns:

    raise RuntimeError(
        "Step 31 result file is missing columns: "
        f"{missing_columns}"
    )


if len(
    step31
) != 396:

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
# Frozen neural populations
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
# Frozen topology checks
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
    "TEMPORAL DNa02 READOUT COMPARISON"
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

print(
    f"Terminal window: "
    f"{TERMINAL_WINDOW_MS:.1f} ms"
)

print(
    f"Output trace tau: "
    f"{float(OUTPUT_TRACE_TAU / ms):.1f} ms"
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


epg_count = sum(

    len(
        indices
    )

    for indices
    in epg_groups.values()
)


if epg_count != 46:

    raise RuntimeError(
        f"Expected 46 EPG neurons, "
        f"got {epg_count}."
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


dna02_indices = get_indices(
    "DNa02"
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
    dna02_indices
) != 2:

    raise RuntimeError(
        "Expected exactly two DNa02 neurons."
    )


if len(
    dna02_left
) != 1:

    raise RuntimeError(
        "Expected one left DNa02 neuron."
    )


if len(
    dna02_right
) != 1:

    raise RuntimeError(
        "Expected one right DNa02 neuron."
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
# Exact functional-edge accounting
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
        "Functional edge accounting failed: "
        f"{accounted_edges} != {len(edges)}"
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
# Convert edges to Brian2 arrays
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
# Helpers
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


def safe_mean(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    if values.size == 0:

        return 0.0


    return float(
        values.mean()
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
# DNa02 StateMonitor row mapping
# =========================================================

dna02_record_indices = [

    int(
        value
    )

    for value
    in dna02_indices
]


dna02_monitor_row = {

    global_index:
        monitor_row

    for monitor_row, global_index
    in enumerate(
        dna02_record_indices
    )
}


dna02_left_rows = [

    dna02_monitor_row[
        int(
            global_index
        )
    ]

    for global_index
    in dna02_left
]


dna02_right_rows = [

    dna02_monitor_row[
        int(
            global_index
        )
    ]

    for global_index
    in dna02_right
]


# =========================================================
# Run one EXACT Step 31 condition
# =========================================================

def run_condition(
    heading_column,
    active_fc2_body_ids,
):

    equations = f"""
    dv/dt = (-v + drive) / (20*ms) : 1
    dx/dt = -x / ({float(OUTPUT_TRACE_TAU / ms)}*ms) : 1
    drive : 1
    """


    reset_code = """
    v = 0.0
    x += 1.0
    """


    group = NeuronGroup(
        len(
            neurons
        ),
        equations,
        threshold="v > 1.0",
        reset=reset_code,
        refractory=3 * ms,
        method="euler",
    )


    group.v = 0.0

    group.x = 0.0

    group.drive = (
        BASELINE_DRIVE
    )


    # =====================================================
    # Frozen EPG heading
    # =====================================================

    if (
        heading_column
        not in epg_groups
    ):

        raise RuntimeError(
            f"Unknown EPG heading: "
            f"{heading_column}"
        )


    for index in epg_groups[
        heading_column
    ]:

        group.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # Exact Step 31 FC2 recruited body IDs
    # =====================================================

    for body_id in active_fc2_body_ids:

        if (
            body_id
            not in body_to_index
        ):

            raise RuntimeError(
                "Step 31 FC2 body ID not found "
                f"in network: {body_id}"
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
                f"Body ID {body_id} is "
                f"{neuron_type}, not FC2."
            )


        group.drive[
            network_index
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
    # Spike monitor
    # =====================================================

    spike_monitor = SpikeMonitor(
        group
    )


    objects.append(
        spike_monitor
    )


    # =====================================================
    # DNa02 temporal state monitor
    #
    # x is passive and does not feed into v.
    # =====================================================

    state_monitor = StateMonitor(
        group,
        [
            "v",
            "x",
        ],
        record=dna02_record_indices,
        dt=STATE_MONITOR_DT,
    )


    objects.append(
        state_monitor
    )


    # =====================================================
    # Run network
    # =====================================================

    network = Network(
        *objects
    )


    network.run(
        SIMULATION_TIME
    )


    # =====================================================
    # Existing Step 27/31 final-state decoder
    # =====================================================

    counts = np.asarray(
        spike_monitor.count
    )


    final_voltage = np.asarray(
        group.v
    )


    left_total_spikes = int(
        counts[
            dna02_left
        ].sum()
    )


    right_total_spikes = int(
        counts[
            dna02_right
        ].sum()
    )


    left_final_v = float(
        final_voltage[
            dna02_left
        ].mean()
    )


    right_final_v = float(
        final_voltage[
            dna02_right
        ].mean()
    )


    left_current_activity = (

        left_final_v

        +

        left_total_spikes
        /
        SPIKE_SCALE
    )


    right_current_activity = (

        right_final_v

        +

        right_total_spikes
        /
        SPIKE_SCALE
    )


    current_activity_difference = (
        right_current_activity
        -
        left_current_activity
    )


    current_steering = float(
        np.tanh(
            current_activity_difference
            /
            (
                2.0
                *
                ACTIVITY_SCALE
            )
        )
    )


    # =====================================================
    # StateMonitor arrays
    # =====================================================

    times_ms = np.asarray(
        state_monitor.t
        /
        ms,
        dtype=float,
    )


    v_trace = np.asarray(
        state_monitor.v,
        dtype=float,
    )


    x_trace = np.asarray(
        state_monitor.x,
        dtype=float,
    )


    terminal_start_ms = (

        float(
            SIMULATION_TIME
            /
            ms
        )

        -

        TERMINAL_WINDOW_MS
    )


    terminal_mask = (
        times_ms
        >=
        terminal_start_ms
    )


    if not np.any(
        terminal_mask
    ):

        raise RuntimeError(
            "No StateMonitor samples in "
            "terminal window."
        )


    # =====================================================
    # Mean membrane voltage — full 300 ms
    # =====================================================

    left_mean_v_full = safe_mean(
        v_trace[
            dna02_left_rows,
            :
        ]
    )


    right_mean_v_full = safe_mean(
        v_trace[
            dna02_right_rows,
            :
        ]
    )


    mean_v_full_difference = (
        right_mean_v_full
        -
        left_mean_v_full
    )


    # =====================================================
    # Mean membrane voltage — final 100 ms
    # =====================================================

    left_mean_v_terminal = safe_mean(
        v_trace[
            dna02_left_rows,
            :
        ][
            :,
            terminal_mask
        ]
    )


    right_mean_v_terminal = safe_mean(
        v_trace[
            dna02_right_rows,
            :
        ][
            :,
            terminal_mask
        ]
    )


    mean_v_terminal_difference = (
        right_mean_v_terminal
        -
        left_mean_v_terminal
    )


    # =====================================================
    # Passive low-pass output trace
    # =====================================================

    left_trace_full = safe_mean(
        x_trace[
            dna02_left_rows,
            :
        ]
    )


    right_trace_full = safe_mean(
        x_trace[
            dna02_right_rows,
            :
        ]
    )


    trace_full_difference = (
        right_trace_full
        -
        left_trace_full
    )


    left_trace_terminal = safe_mean(
        x_trace[
            dna02_left_rows,
            :
        ][
            :,
            terminal_mask
        ]
    )


    right_trace_terminal = safe_mean(
        x_trace[
            dna02_right_rows,
            :
        ][
            :,
            terminal_mask
        ]
    )


    trace_terminal_difference = (
        right_trace_terminal
        -
        left_trace_terminal
    )


    # =====================================================
    # Spike rates
    # =====================================================

    full_duration_seconds = float(
        SIMULATION_TIME
        /
        (1000 * ms)
    )


    left_rate_full_hz = (
        left_total_spikes
        /
        full_duration_seconds
    )


    right_rate_full_hz = (
        right_total_spikes
        /
        full_duration_seconds
    )


    rate_full_difference_hz = (
        right_rate_full_hz
        -
        left_rate_full_hz
    )


    # =====================================================
    # Final 100 ms spike rate
    # =====================================================

    spike_indices = np.asarray(
        spike_monitor.i,
        dtype=int,
    )


    spike_times_ms = np.asarray(
        spike_monitor.t
        /
        ms,
        dtype=float,
    )


    terminal_spike_mask = (
        spike_times_ms
        >=
        terminal_start_ms
    )


    terminal_spike_indices = spike_indices[
        terminal_spike_mask
    ]


    left_terminal_spikes = int(
        np.isin(
            terminal_spike_indices,
            dna02_left,
        ).sum()
    )


    right_terminal_spikes = int(
        np.isin(
            terminal_spike_indices,
            dna02_right,
        ).sum()
    )


    terminal_duration_seconds = (
        TERMINAL_WINDOW_MS
        /
        1000.0
    )


    left_rate_terminal_hz = (
        left_terminal_spikes
        /
        terminal_duration_seconds
    )


    right_rate_terminal_hz = (
        right_terminal_spikes
        /
        terminal_duration_seconds
    )


    rate_terminal_difference_hz = (
        right_rate_terminal_hz
        -
        left_rate_terminal_hz
    )


    return {

        "current_steering":
            current_steering,

        "current_activity_difference":
            current_activity_difference,

        "left_final_v":
            left_final_v,

        "right_final_v":
            right_final_v,

        "left_total_spikes":
            left_total_spikes,

        "right_total_spikes":
            right_total_spikes,

        "mean_v_full_difference":
            mean_v_full_difference,

        "mean_v_terminal_difference":
            mean_v_terminal_difference,

        "rate_full_difference_hz":
            rate_full_difference_hz,

        "rate_terminal_difference_hz":
            rate_terminal_difference_hz,

        "trace_full_difference":
            trace_full_difference,

        "trace_terminal_difference":
            trace_terminal_difference,
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


for condition_index, source_row in step31.iterrows():

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
        f"[{condition_index + 1:3d}/"
        f"{len(step31)}] "
        f"heading={heading:>2} | "
        f"C{column_a}->C{column_b} | "
        f"alpha={alpha:.1f} | "
        f"FC2 active={len(active_fc2_body_ids)}"
    )


    diagnostic = run_condition(
        heading_column=heading,
        active_fc2_body_ids=active_fc2_body_ids,
    )


    record = {

        "heading_column":
            heading,

        "column_a":
            column_a,

        "column_b":
            column_b,

        "alpha":
            alpha,

        "active_fc2_count":
            len(
                active_fc2_body_ids
            ),

        "source_step31_steering":
            float(
                source_row[
                    "steering"
                ]
            ),

        **diagnostic,
    }


    record[
        "current_reproduction_error"
    ] = abs(

        record[
            "current_steering"
        ]

        -

        record[
            "source_step31_steering"
        ]
    )


    rows.append(
        record
    )


results = pd.DataFrame(
    rows
)


# =========================================================
# Verify that adding passive monitors/trace did NOT alter
# frozen dynamics.
# =========================================================

print()

print("=" * 120)
print(
    "STEP 31 REPRODUCTION CHECK"
)
print("=" * 120)

print()


mean_reproduction_error = float(
    results[
        "current_reproduction_error"
    ].mean()
)


max_reproduction_error = float(
    results[
        "current_reproduction_error"
    ].max()
)


print(
    f"Conditions checked: "
    f"{len(results)}"
)


print(
    f"Mean steering error: "
    f"{mean_reproduction_error:.10f}"
)


print(
    f"Max steering error: "
    f"{max_reproduction_error:.10f}"
)


if max_reproduction_error <= 1e-8:

    print(
        "PASS: temporal monitoring does not "
        "alter Step 31 dynamics."
    )


else:

    print(
        "WARNING: rerun differs from Step 31."
    )


# =========================================================
# New readout methods
#
# For fair comparison, each NEW raw DNa signal is mapped
# into [-1, +1] with a scale estimated ONLY from the pure
# endpoint states alpha=0 or alpha=1.
#
# Intermediate blend states are NOT used for calibration.
#
# This is an engineering normalization for comparison,
# not a biological parameter.
# =========================================================

RAW_READOUTS = {

    "meanV_full":
        "mean_v_full_difference",

    "meanV_last100":
        "mean_v_terminal_difference",

    "rate_full":
        "rate_full_difference_hz",

    "rate_last100":
        "rate_terminal_difference_hz",

    "lowpass_full":
        "trace_full_difference",

    "lowpass_last100":
        "trace_terminal_difference",
}


endpoint_mask = (

    np.isclose(
        results[
            "alpha"
        ].to_numpy(
            dtype=float
        ),
        0.0,
    )

    |

    np.isclose(
        results[
            "alpha"
        ].to_numpy(
            dtype=float
        ),
        1.0,
    )
)


endpoint_results = results.loc[
    endpoint_mask
].copy()


readout_scales = {}


print()

print("=" * 120)
print(
    "ENDPOINT-ONLY READOUT NORMALIZATION"
)
print("=" * 120)

print()


for method_name, raw_column in RAW_READOUTS.items():

    endpoint_values = np.abs(
        endpoint_results[
            raw_column
        ].to_numpy(
            dtype=float
        )
    )


    scale = float(
        np.percentile(
            endpoint_values,
            95.0,
        )
    )


    if scale < 1e-9:

        scale = float(
            np.max(
                endpoint_values
            )
        )


    if scale < 1e-9:

        scale = 1.0


    readout_scales[
        method_name
    ] = scale


    output_column = (
        f"steering_{method_name}"
    )


    results[
        output_column
    ] = np.tanh(

        results[
            raw_column
        ].to_numpy(
            dtype=float
        )

        /

        (
            2.0
            *
            scale
        )
    )


    print(
        f"{method_name:20s} "
        f"| endpoint p95 scale = "
        f"{scale:.6f}"
    )


# Existing readout gets its own named column.
results[
    "steering_current"
] = results[
    "current_steering"
]


# =========================================================
# Save condition-level results
# =========================================================

results.to_csv(
    RESULT_FILE,
    index=False,
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
# Readout methods to compare
# =========================================================

STEERING_METHODS = {

    "current_finalV_plus_totalSpikes":
        "steering_current",

    "meanV_full":
        "steering_meanV_full",

    "meanV_last100":
        "steering_meanV_last100",

    "rate_full":
        "steering_rate_full",

    "rate_last100":
        "steering_rate_last100",

    "lowpass_full":
        "steering_lowpass_full",

    "lowpass_last100":
        "steering_lowpass_last100",
}


# =========================================================
# Per-trajectory analysis
# =========================================================

trajectory_rows = []


for method_name, steering_column in STEERING_METHODS.items():

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


        alpha = group[
            "alpha"
        ].to_numpy(
            dtype=float
        )


        steering = group[
            steering_column
        ].to_numpy(
            dtype=float
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

            worst_index = int(
                np.argmax(
                    steps
                )
            )


            max_abs_step = float(
                steps[
                    worst_index
                ]
            )


            largest_step_alpha_from = float(
                alpha[
                    worst_index
                ]
            )


            largest_step_alpha_to = float(
                alpha[
                    worst_index
                    +
                    1
                ]
            )


        else:

            max_abs_step = 0.0

            largest_step_alpha_from = 0.0

            largest_step_alpha_to = 0.0


        trajectory_rows.append(
            {

                "method":
                    method_name,

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
                    sign_transitions,

                "near_zero_intermediate":
                    near_zero_intermediate,

                "max_abs_steering":
                    float(
                        np.abs(
                            steering
                        ).max()
                    ),
            }
        )


trajectory_results = pd.DataFrame(
    trajectory_rows
)


trajectory_results.to_csv(
    TRAJECTORY_FILE,
    index=False,
)


# =========================================================
# Aggregate readout comparison
# =========================================================

comparison_rows = []


for method_name in STEERING_METHODS:

    method_data = trajectory_results[
        trajectory_results[
            "method"
        ]
        ==
        method_name
    ]


    comparison_rows.append(
        {

            "method":
                method_name,

            "trajectories":
                len(
                    method_data
                ),

            "mean_mean_step":
                float(
                    method_data[
                        "mean_abs_step"
                    ].mean()
                ),

            "mean_max_step":
                float(
                    method_data[
                        "max_abs_step"
                    ].mean()
                ),

            "worst_step":
                float(
                    method_data[
                        "max_abs_step"
                    ].max()
                ),

            "mean_directness":
                float(
                    method_data[
                        "directness_ratio"
                    ].mean()
                ),

            "median_directness":
                float(
                    method_data[
                        "directness_ratio"
                    ].median()
                ),

            "maximum_overshoot":
                float(
                    method_data[
                        "overshoot"
                    ].max()
                ),

            "sign_transition_trajectories":
                int(
                    (
                        method_data[
                            "sign_transitions"
                        ]
                        >
                        0
                    ).sum()
                ),

            "multi_sign_flip":
                int(
                    (
                        method_data[
                            "sign_transitions"
                        ]
                        >
                        1
                    ).sum()
                ),

            "step_gt_0_10":
                int(
                    (
                        method_data[
                            "max_abs_step"
                        ]
                        >
                        0.10
                    ).sum()
                ),

            "step_gt_0_20":
                int(
                    (
                        method_data[
                            "max_abs_step"
                        ]
                        >
                        0.20
                    ).sum()
                ),

            "near_zero_trajectory":
                int(
                    (
                        method_data[
                            "near_zero_intermediate"
                        ]
                        >
                        0
                    ).sum()
                ),
        }
    )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    COMPARISON_FILE,
    index=False,
)


# =========================================================
# Print comparison
# =========================================================

print()

print("=" * 180)
print(
    "DNa TEMPORAL READOUT COMPARISON"
)
print("=" * 180)

print()


print(
    comparison.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Compare each method against current decoder
# =========================================================

current_row = comparison[
    comparison[
        "method"
    ]
    ==
    "current_finalV_plus_totalSpikes"
].iloc[
    0
]


current_worst = float(
    current_row[
        "worst_step"
    ]
)


current_mean_max = float(
    current_row[
        "mean_max_step"
    ]
)


print()

print("=" * 150)
print(
    "IMPROVEMENT RELATIVE TO CURRENT DNa DECODER"
)
print("=" * 150)

print()


for _, row in comparison.iterrows():

    method = str(
        row[
            "method"
        ]
    )


    worst = float(
        row[
            "worst_step"
        ]
    )


    mean_max = float(
        row[
            "mean_max_step"
        ]
    )


    worst_reduction = (

        current_worst
        -
        worst
    )


    worst_percent = (

        100.0
        *
        worst_reduction
        /
        current_worst

        if current_worst
        > 1e-12

        else 0.0
    )


    mean_reduction = (

        current_mean_max
        -
        mean_max
    )


    mean_percent = (

        100.0
        *
        mean_reduction
        /
        current_mean_max

        if current_mean_max
        > 1e-12

        else 0.0
    )


    print(
        f"{method:32s} "
        f"| worst={worst:.4f} "
        f"| worst reduction="
        f"{worst_reduction:+.4f} "
        f"({worst_percent:+.1f}%) "
        f"| mean-max reduction="
        f"{mean_reduction:+.4f} "
        f"({mean_percent:+.1f}%)"
    )


# =========================================================
# Show worst trajectory for every readout
# =========================================================

print()

print("=" * 180)
print(
    "WORST TRAJECTORY PER DNa READOUT"
)
print("=" * 180)

print()


worst_method_rows = []


for method_name in STEERING_METHODS:

    method_data = trajectory_results[
        trajectory_results[
            "method"
        ]
        ==
        method_name
    ]


    worst = (
        method_data
        .sort_values(
            "max_abs_step",
            ascending=False,
        )
        .iloc[
            0
        ]
    )


    worst_method_rows.append(
        worst
    )


worst_method_dataframe = pd.DataFrame(
    worst_method_rows
)


print(
    worst_method_dataframe[
        [
            "method",
            "heading_column",
            "column_a",
            "column_b",
            "max_abs_step",
            "largest_step_alpha_from",
            "largest_step_alpha_to",
            "directness_ratio",
            "overshoot",
            "sign_transitions",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Current Step 31 worst trajectory
#
# Print every decoder side by side for the exact same
# condition so we can see whether the discontinuity exists
# in DNa state or only in the current decoder.
# =========================================================

current_trajectories = trajectory_results[
    trajectory_results[
        "method"
    ]
    ==
    "current_finalV_plus_totalSpikes"
]


current_worst_trajectory = (
    current_trajectories
    .sort_values(
        "max_abs_step",
        ascending=False,
    )
    .iloc[
        0
    ]
)


worst_heading = str(
    current_worst_trajectory[
        "heading_column"
    ]
)


worst_a = int(
    current_worst_trajectory[
        "column_a"
    ]
)


worst_b = int(
    current_worst_trajectory[
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

print("=" * 220)
print(
    "ALL READOUTS ON CURRENT DECODER'S WORST TRAJECTORY"
)
print("=" * 220)

print()


detail_columns = [
    "heading_column",
    "column_a",
    "column_b",
    "alpha",
    "left_total_spikes",
    "right_total_spikes",
    "left_final_v",
    "right_final_v",
    "steering_current",
    "steering_meanV_full",
    "steering_meanV_last100",
    "steering_rate_full",
    "steering_rate_last100",
    "steering_lowpass_full",
    "steering_lowpass_last100",
]


print(
    worst_detail[
        detail_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.5f}",
    )
)


# =========================================================
# Underlying raw DNa state for same worst trajectory
# =========================================================

print()

print("=" * 200)
print(
    "RAW DNa02 SIGNALS ON CURRENT WORST TRAJECTORY"
)
print("=" * 200)

print()


raw_columns = [
    "heading_column",
    "column_a",
    "column_b",
    "alpha",
    "current_activity_difference",
    "mean_v_full_difference",
    "mean_v_terminal_difference",
    "rate_full_difference_hz",
    "rate_terminal_difference_hz",
    "trace_full_difference",
    "trace_terminal_difference",
]


print(
    worst_detail[
        raw_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.6f}",
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
    TRAJECTORY_FILE
)

print(
    COMPARISON_FILE
)