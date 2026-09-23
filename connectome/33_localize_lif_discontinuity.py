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
    / "lif_discontinuity_condition_metrics.csv"
)

TRANSITION_FILE = (
    OUTPUT_DIR
    / "lif_discontinuity_transition_metrics.csv"
)

VECTOR_SUMMARY_FILE = (
    OUTPUT_DIR
    / "lif_discontinuity_vector_summary.csv"
)

LATERAL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "lif_discontinuity_lateral_summary.csv"
)


# =========================================================
# Frozen surrogate calibration
#
# DO NOT tune anything in Step 33.
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
# Load exact Step 31 conditions
# =========================================================

if not STEP31_RESULT_FILE.exists():

    raise FileNotFoundError(
        "Step 31 result file was not found:\n"
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
# Load connectome data
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
        subset=["bodyId"]
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
    edges["source_bodyId"].isin(
        body_ids
    )
    &
    edges["target_bodyId"].isin(
        body_ids
    )
].copy()


# =========================================================
# Same duplicate handling as Step 27
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
    "STEP 33 - LOCALIZE LIF DISCONTINUITY"
)
print("=" * 120)

print()

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)

print(
    f"Step 31 conditions: {len(step31)}"
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
# Network index
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


    if not 1 <= number <= 8:

        return None


    return f"{side}{number}"


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


if sum(
    len(indices)
    for indices
    in epg_groups.values()
) != 46:

    raise RuntimeError(
        "Expected 46 grouped EPG neurons."
    )


# =========================================================
# Population helpers
# =========================================================

def normalize_side(
    value,
):

    if pd.isna(value):

        return ""


    return str(
        value
    ).strip().upper()


def get_indices(
    neuron_types,
    side=None,
):

    if isinstance(
        neuron_types,
        str,
    ):

        neuron_types = [
            neuron_types
        ]


    allowed_types = set(
        neuron_types
    )


    output = []


    for index, row in neurons.iterrows():

        if row["type"] not in allowed_types:

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
                str(side).upper()
            ):

                continue


        output.append(
            index
        )


    return np.asarray(
        output,
        dtype=int,
    )


fc2_indices = get_indices(
    [
        "FC2A",
        "FC2B",
        "FC2C",
    ]
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


pfl2_left = get_indices(
    "PFL2",
    "L",
)

pfl2_right = get_indices(
    "PFL2",
    "R",
)

pfl3_left = get_indices(
    "PFL3",
    "L",
)

pfl3_right = get_indices(
    "PFL3",
    "R",
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


if len(fc2_indices) != 92:

    raise RuntimeError(
        f"Expected 92 FC2 neurons, "
        f"got {len(fc2_indices)}."
    )


if len(dna03_indices) != 2:

    raise RuntimeError(
        f"Expected 2 DNa03 neurons, "
        f"got {len(dna03_indices)}."
    )


if len(dna02_indices) != 2:

    raise RuntimeError(
        f"Expected 2 DNa02 neurons, "
        f"got {len(dna02_indices)}."
    )


print()

print("=" * 120)
print(
    "POPULATION INVENTORY"
)
print("=" * 120)

print(
    f"FC2:   {len(fc2_indices)}"
)

print(
    f"PFL2:  {len(pfl2_indices)} "
    f"(L={len(pfl2_left)}, "
    f"R={len(pfl2_right)})"
)

print(
    f"PFL3:  {len(pfl3_indices)} "
    f"(L={len(pfl3_left)}, "
    f"R={len(pfl3_right)})"
)

print(
    f"DNa03: {len(dna03_indices)} "
    f"(L={len(dna03_left)}, "
    f"R={len(dna03_right)})"
)

print(
    f"DNa02: {len(dna02_indices)} "
    f"(L={len(dna02_left)}, "
    f"R={len(dna02_right)})"
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


if accounted_edges != len(edges):

    raise RuntimeError(
        "Functional edge accounting mismatch: "
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
# Prepare edges
# =========================================================

def prepare_edges(
    dataframe,
):

    sources = []
    targets = []
    raw_weights = []


    for _, row in dataframe.iterrows():

        source_id = int(
            row["source_bodyId"]
        )

        target_id = int(
            row["target_bodyId"]
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
            values
            for values
            in raw_arrays
            if len(values) > 0
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
# Generic helpers
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
        *
        gain
    )


    return synapses


def parse_body_ids(
    value,
):

    if pd.isna(value):

        return []


    text = str(
        value
    ).strip()


    if not text:

        return []


    output = []


    for piece in text.split(","):

        piece = piece.strip()


        if not piece:

            continue


        output.append(
            int(piece)
        )


    return output


def safe_mean(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    if values.size == 0:

        return np.nan


    return float(
        values.mean()
    )


# =========================================================
# Monitor populations
# =========================================================

LAYER_INDICES = {

    "FC2":
        fc2_indices,

    "PFL2":
        pfl2_indices,

    "PFL3":
        pfl3_indices,

    "DNa03":
        dna03_indices,

    "DNa02":
        dna02_indices,
}


monitor_indices = sorted(
    set(
        int(index)
        for layer_indices
        in LAYER_INDICES.values()
        for index
        in layer_indices
    )
)


monitor_row_for_network_index = {

    network_index:
        monitor_row

    for monitor_row, network_index
    in enumerate(
        monitor_indices
    )
}


# =========================================================
# Run one exact Step 31 condition
# =========================================================

def run_condition(
    heading_column,
    active_fc2_body_ids,
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
    # EPG drive
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
    # Exact Step 31 FC2 population
    # =====================================================

    active_fc2_network_indices = []


    for body_id in sorted(
        set(
            active_fc2_body_ids
        )
    ):

        if body_id not in body_to_index:

            raise RuntimeError(
                f"Unknown FC2 body ID: "
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
                f"is not FC2."
            )


        group.drive[
            network_index
        ] = FC2_DRIVE


        active_fc2_network_indices.append(
            network_index
        )


    # =====================================================
    # Build network
    # =====================================================

    objects = [
        group
    ]


    # EPG + FC2 -> PFL
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


    # EPG -> Delta7
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


    # Delta7 -> PFL inhibitory surrogate
    if len(delta_pfl_source) > 0:

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


    # Complete frozen PFL -> DNa topology
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


    # DNa03 -> DNa02
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


    spike_monitor = SpikeMonitor(
        group
    )


    state_monitor = StateMonitor(
        group,
        "v",
        record=monitor_indices,
        dt=STATE_MONITOR_DT,
    )


    objects.append(
        spike_monitor
    )

    objects.append(
        state_monitor
    )


    network = Network(
        *objects
    )


    network.run(
        SIMULATION_TIME
    )


    # =====================================================
    # Raw simulation state
    # =====================================================

    counts = np.asarray(
        spike_monitor.count,
        dtype=float,
    )


    final_voltage = np.asarray(
        group.v,
        dtype=float,
    )


    v_trace = np.asarray(
        state_monitor.v,
        dtype=float,
    )


    duration_seconds = float(
        SIMULATION_TIME
        /
        (1000 * ms)
    )


    # =====================================================
    # Mean voltage and rate vectors by layer
    # =====================================================

    layer_rate_vectors = {}

    layer_voltage_vectors = {}


    for (
        layer_name,
        layer_indices,
    ) in LAYER_INDICES.items():

        layer_indices_list = [
            int(index)
            for index
            in layer_indices
        ]


        rows = [
            monitor_row_for_network_index[
                index
            ]

            for index
            in layer_indices_list
        ]


        layer_voltage_vectors[
            layer_name
        ] = np.mean(
            v_trace[
                rows,
                :
            ],
            axis=1,
        )


        layer_rate_vectors[
            layer_name
        ] = (
            counts[
                layer_indices_list
            ]
            /
            duration_seconds
        )


    # =====================================================
    # Binary FC2 sensory input vector
    # =====================================================

    fc2_input_vector = np.zeros(
        len(fc2_indices),
        dtype=float,
    )


    fc2_position = {

        int(network_index):
            position

        for position, network_index
        in enumerate(
            fc2_indices
        )
    }


    for network_index in (
        active_fc2_network_indices
    ):

        fc2_input_vector[
            fc2_position[
                int(network_index)
            ]
        ] = 1.0


    # =====================================================
    # Scalar population metrics
    # =====================================================

    def mean_rate(
        indices,
    ):

        indices = np.asarray(
            indices,
            dtype=int,
        )


        if len(indices) == 0:

            return np.nan


        return float(
            counts[
                indices
            ].sum()
            /
            len(indices)
            /
            duration_seconds
        )


    def mean_voltage(
        indices,
    ):

        indices = [
            int(index)
            for index
            in indices
        ]


        if len(indices) == 0:

            return np.nan


        rows = [
            monitor_row_for_network_index[
                index
            ]

            for index
            in indices
        ]


        return float(
            v_trace[
                rows,
                :
            ].mean()
        )


    scalar_metrics = {}


    for layer_name in [
        "FC2",
        "PFL2",
        "PFL3",
        "DNa03",
        "DNa02",
    ]:

        indices = LAYER_INDICES[
            layer_name
        ]


        scalar_metrics[
            f"{layer_name}_mean_rate_hz"
        ] = mean_rate(
            indices
        )


        scalar_metrics[
            f"{layer_name}_mean_v"
        ] = mean_voltage(
            indices
        )


    # =====================================================
    # Lateral differences
    #
    # right - left
    # =====================================================

    lateral_groups = {

        "PFL2":
            (
                pfl2_left,
                pfl2_right,
            ),

        "PFL3":
            (
                pfl3_left,
                pfl3_right,
            ),

        "DNa03":
            (
                dna03_left,
                dna03_right,
            ),

        "DNa02":
            (
                dna02_left,
                dna02_right,
            ),
    }


    for (
        layer_name,
        (
            left_indices,
            right_indices,
        ),
    ) in lateral_groups.items():

        if (
            len(left_indices) > 0
            and
            len(right_indices) > 0
        ):

            scalar_metrics[
                f"{layer_name}_rate_diff"
            ] = (
                mean_rate(
                    right_indices
                )
                -
                mean_rate(
                    left_indices
                )
            )


            scalar_metrics[
                f"{layer_name}_v_diff"
            ] = (
                mean_voltage(
                    right_indices
                )
                -
                mean_voltage(
                    left_indices
                )
            )


        else:

            scalar_metrics[
                f"{layer_name}_rate_diff"
            ] = np.nan


            scalar_metrics[
                f"{layer_name}_v_diff"
            ] = np.nan


    # =====================================================
    # Existing DNa02 decoder reproduction
    # =====================================================

    left_spikes = int(
        counts[
            dna02_left
        ].sum()
    )


    right_spikes = int(
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


    left_activity = (
        left_final_v
        +
        left_spikes
        /
        SPIKE_SCALE
    )


    right_activity = (
        right_final_v
        +
        right_spikes
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
                active_fc2_network_indices
            ),

        "steering":
            steering,

        "activity_difference":
            activity_difference,

        **scalar_metrics,

        "feature_vectors": {

            "FC2_input":
                fc2_input_vector,

            "FC2_rate":
                layer_rate_vectors[
                    "FC2"
                ],

            "PFL2_rate":
                layer_rate_vectors[
                    "PFL2"
                ],

            "PFL3_rate":
                layer_rate_vectors[
                    "PFL3"
                ],

            "DNa03_rate":
                layer_rate_vectors[
                    "DNa03"
                ],

            "DNa02_rate":
                layer_rate_vectors[
                    "DNa02"
                ],

            "FC2_voltage":
                layer_voltage_vectors[
                    "FC2"
                ],

            "PFL2_voltage":
                layer_voltage_vectors[
                    "PFL2"
                ],

            "PFL3_voltage":
                layer_voltage_vectors[
                    "PFL3"
                ],

            "DNa03_voltage":
                layer_voltage_vectors[
                    "DNa03"
                ],

            "DNa02_voltage":
                layer_voltage_vectors[
                    "DNa02"
                ],
        },
    }


# =========================================================
# Rerun exact Step 31 conditions
# =========================================================

condition_rows = []

feature_records = []


print()

print("=" * 120)
print(
    "RERUNNING EXACT STEP 31 CONDITIONS"
)
print("=" * 120)

print()


for source_index, source_row in step31.iterrows():

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


    active_body_ids = (

        parse_body_ids(
            source_row[
                "a_body_ids"
            ]
        )

        +

        parse_body_ids(
            source_row[
                "b_body_ids"
            ]
        )
    )


    condition_id = len(
        condition_rows
    )


    print(
        f"[{condition_id + 1:3d}/"
        f"{len(step31)}] "
        f"heading={heading:>2} | "
        f"C{column_a}->C{column_b} | "
        f"alpha={alpha:.1f} | "
        f"FC2={len(active_body_ids)}"
    )


    diagnostic = run_condition(
        heading_column=heading,
        active_fc2_body_ids=active_body_ids,
    )


    feature_vectors = diagnostic.pop(
        "feature_vectors"
    )


    reproduced_steering = float(
        diagnostic[
            "steering"
        ]
    )


    step31_steering = float(
        source_row[
            "steering"
        ]
    )


    condition_rows.append(
        {

            "condition_id":
                condition_id,

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


    feature_records.append(
        feature_vectors
    )


conditions = pd.DataFrame(
    condition_rows
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
    f"{mean_error:.10f}"
)

print(
    f"Max steering error: "
    f"{max_error:.10f}"
)


if max_error <= 1e-8:

    print(
        "PASS: layer monitoring does not "
        "alter Step 31 dynamics."
    )


else:

    print(
        "WARNING: Step 33 does not exactly "
        "reproduce Step 31."
    )


# =========================================================
# Characteristic vector scales
#
# scale = p95 of vector magnitude across all conditions.
#
# Used only for dimensionless diagnostic comparison.
# =========================================================

FEATURE_NAMES = [
    "FC2_input",
    "FC2_rate",
    "PFL2_rate",
    "PFL3_rate",
    "DNa03_rate",
    "DNa02_rate",
    "FC2_voltage",
    "PFL2_voltage",
    "PFL3_voltage",
    "DNa03_voltage",
    "DNa02_voltage",
]


feature_scales = {}


for feature_name in FEATURE_NAMES:

    matrix = np.stack(
        [
            record[
                feature_name
            ]
            for record
            in feature_records
        ]
    )


    vector_norms = np.linalg.norm(
        matrix,
        axis=1,
    )


    scale = float(
        np.percentile(
            vector_norms,
            95.0,
        )
    )


    if scale < 1e-12:

        scale = float(
            vector_norms.max()
        )


    if scale < 1e-12:

        scale = 1.0


    feature_scales[
        feature_name
    ] = scale


# =========================================================
# Build adjacent-alpha transition diagnostics
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


    for step_index in range(
        len(group) - 1
    ):

        row_from = group.iloc[
            step_index
        ]

        row_to = group.iloc[
            step_index + 1
        ]


        id_from = int(
            row_from[
                "condition_id"
            ]
        )

        id_to = int(
            row_to[
                "condition_id"
            ]
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
                    row_from[
                        "alpha"
                    ]
                ),

            "alpha_to":
                float(
                    row_to[
                        "alpha"
                    ]
                ),

            "steering_from":
                float(
                    row_from[
                        "steering"
                    ]
                ),

            "steering_to":
                float(
                    row_to[
                        "steering"
                    ]
                ),

            "steering_abs_step":
                abs(
                    float(
                        row_to[
                            "steering"
                        ]
                    )
                    -
                    float(
                        row_from[
                            "steering"
                        ]
                    )
                ),
        }


        # -------------------------------------------------
        # Vector-state jump at every layer
        # -------------------------------------------------

        for feature_name in FEATURE_NAMES:

            vector_from = (
                feature_records[
                    id_from
                ][
                    feature_name
                ]
            )


            vector_to = (
                feature_records[
                    id_to
                ][
                    feature_name
                ]
            )


            raw_step = float(
                np.linalg.norm(
                    vector_to
                    -
                    vector_from
                )
            )


            normalized_step = (
                raw_step
                /
                feature_scales[
                    feature_name
                ]
            )


            record[
                f"{feature_name}_raw_step"
            ] = raw_step


            record[
                f"{feature_name}_norm_step"
            ] = normalized_step


        # -------------------------------------------------
        # Amplification relative to FC2 input-set change
        # -------------------------------------------------

        input_step = record[
            "FC2_input_norm_step"
        ]


        for feature_name in FEATURE_NAMES:

            if feature_name == "FC2_input":

                continue


            if input_step > 1e-12:

                amplification = (
                    record[
                        f"{feature_name}_norm_step"
                    ]
                    /
                    input_step
                )


            else:

                amplification = np.nan


            record[
                f"{feature_name}_amp_vs_input"
            ] = amplification


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
# Vector discontinuity summary
# =========================================================

feature_metadata = {

    "FC2_input":
        (
            "FC2",
            "external_binary_input",
        ),

    "FC2_rate":
        (
            "FC2",
            "spike_rate",
        ),

    "PFL2_rate":
        (
            "PFL2",
            "spike_rate",
        ),

    "PFL3_rate":
        (
            "PFL3",
            "spike_rate",
        ),

    "DNa03_rate":
        (
            "DNa03",
            "spike_rate",
        ),

    "DNa02_rate":
        (
            "DNa02",
            "spike_rate",
        ),

    "FC2_voltage":
        (
            "FC2",
            "mean_membrane",
        ),

    "PFL2_voltage":
        (
            "PFL2",
            "mean_membrane",
        ),

    "PFL3_voltage":
        (
            "PFL3",
            "mean_membrane",
        ),

    "DNa03_voltage":
        (
            "DNa03",
            "mean_membrane",
        ),

    "DNa02_voltage":
        (
            "DNa02",
            "mean_membrane",
        ),
}


vector_summary_rows = []


for feature_name in FEATURE_NAMES:

    normalized_steps = transitions[
        f"{feature_name}_norm_step"
    ].to_numpy(
        dtype=float,
    )


    if feature_name != "FC2_input":

        amplification_values = transitions[
            f"{feature_name}_amp_vs_input"
        ].to_numpy(
            dtype=float,
        )


        amplification_values = (
            amplification_values[
                np.isfinite(
                    amplification_values
                )
            ]
        )


    else:

        amplification_values = np.asarray(
            [],
            dtype=float,
        )


    layer_name, modality = (
        feature_metadata[
            feature_name
        ]
    )


    vector_summary_rows.append(
        {

            "feature":
                feature_name,

            "layer":
                layer_name,

            "modality":
                modality,

            "vector_dimension":
                len(
                    feature_records[
                        0
                    ][
                        feature_name
                    ]
                ),

            "characteristic_scale":
                feature_scales[
                    feature_name
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

            "mean_amplification_vs_input":
                (
                    float(
                        amplification_values.mean()
                    )

                    if len(
                        amplification_values
                    ) > 0

                    else np.nan
                ),

            "p95_amplification_vs_input":
                (
                    float(
                        np.percentile(
                            amplification_values,
                            95.0,
                        )
                    )

                    if len(
                        amplification_values
                    ) > 0

                    else np.nan
                ),

            "max_amplification_vs_input":
                (
                    float(
                        amplification_values.max()
                    )

                    if len(
                        amplification_values
                    ) > 0

                    else np.nan
                ),
        }
    )


vector_summary = pd.DataFrame(
    vector_summary_rows
)


vector_summary.to_csv(
    VECTOR_SUMMARY_FILE,
    index=False,
)


# =========================================================
# Lateral-difference scalar analysis
#
# This is particularly important for steering because
# right-left asymmetry is what eventually drives turning.
# =========================================================

candidate_lateral_signals = [
    "PFL2_rate_diff",
    "PFL3_rate_diff",
    "DNa03_rate_diff",
    "DNa02_rate_diff",
    "PFL2_v_diff",
    "PFL3_v_diff",
    "DNa03_v_diff",
    "DNa02_v_diff",
]


lateral_summary_rows = []


for signal_name in candidate_lateral_signals:

    if signal_name not in conditions.columns:

        continue


    all_values = conditions[
        signal_name
    ].to_numpy(
        dtype=float,
    )


    finite_values = all_values[
        np.isfinite(
            all_values
        )
    ]


    if len(finite_values) == 0:

        continue


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


    scalar_steps = []

    normalized_scalar_steps = []

    sign_flips = 0


    for (
        _heading,
        _column_a,
        _column_b,
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
        )


        values = group[
            signal_name
        ].to_numpy(
            dtype=float,
        )


        for index in range(
            len(values) - 1
        ):

            first = values[
                index
            ]

            second = values[
                index + 1
            ]


            if (
                not np.isfinite(first)
                or
                not np.isfinite(second)
            ):

                continue


            step = abs(
                second
                -
                first
            )


            scalar_steps.append(
                step
            )


            normalized_scalar_steps.append(
                step
                /
                scale
            )


            if (
                abs(first) > 1e-9
                and
                abs(second) > 1e-9
                and
                np.sign(first)
                !=
                np.sign(second)
            ):

                sign_flips += 1


    scalar_steps = np.asarray(
        scalar_steps,
        dtype=float,
    )


    normalized_scalar_steps = np.asarray(
        normalized_scalar_steps,
        dtype=float,
    )


    lateral_summary_rows.append(
        {

            "signal":
                signal_name,

            "scale":
                scale,

            "mean_normalized_step":
                float(
                    normalized_scalar_steps.mean()
                ),

            "p95_normalized_step":
                float(
                    np.percentile(
                        normalized_scalar_steps,
                        95.0,
                    )
                ),

            "max_normalized_step":
                float(
                    normalized_scalar_steps.max()
                ),

            "adjacent_sign_flips":
                int(
                    sign_flips
                ),

            "steps_gt_0_25":
                int(
                    (
                        normalized_scalar_steps
                        >
                        0.25
                    ).sum()
                ),

            "steps_gt_0_50":
                int(
                    (
                        normalized_scalar_steps
                        >
                        0.50
                    ).sum()
                ),
        }
    )


lateral_summary = pd.DataFrame(
    lateral_summary_rows
)


lateral_summary.to_csv(
    LATERAL_SUMMARY_FILE,
    index=False,
)


# =========================================================
# Locate current decoder's single worst alpha transition
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
# Print global vector summary
# =========================================================

print()

print("=" * 180)
print(
    "LAYER VECTOR DISCONTINUITY SUMMARY"
)
print("=" * 180)

print()


print(
    vector_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Print lateral summary
# =========================================================

print()

print("=" * 160)
print(
    "LATERAL DIFFERENCE DISCONTINUITY SUMMARY"
)
print("=" * 160)

print()


if len(lateral_summary) > 0:

    print(
        lateral_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


else:

    print(
        "No usable bilateral population "
        "signals were available."
    )


# =========================================================
# Detailed current worst transition
# =========================================================

print()

print("=" * 170)
print(
    "CURRENT DECODER WORST ADJACENT TRANSITION"
)
print("=" * 170)

print()

print(
    f"Heading: "
    f"{worst_heading}"
)

print(
    f"FC2 transition: "
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
# Layer-by-layer localization for exact worst transition
# =========================================================

localization_rows = []


for feature_name in FEATURE_NAMES:

    layer_name, modality = (
        feature_metadata[
            feature_name
        ]
    )


    raw_step = float(
        worst_transition[
            f"{feature_name}_raw_step"
        ]
    )


    normalized_step = float(
        worst_transition[
            f"{feature_name}_norm_step"
        ]
    )


    if feature_name == "FC2_input":

        amplification = np.nan


    else:

        amplification = float(
            worst_transition[
                f"{feature_name}_amp_vs_input"
            ]
        )


    localization_rows.append(
        {

            "feature":
                feature_name,

            "layer":
                layer_name,

            "modality":
                modality,

            "raw_step":
                raw_step,

            "normalized_step":
                normalized_step,

            "amplification_vs_FC2_input":
                amplification,
        }
    )


localization = pd.DataFrame(
    localization_rows
)


print()

print("=" * 160)
print(
    "LAYER-BY-LAYER LOCALIZATION OF WORST TRANSITION"
)
print("=" * 160)

print()


print(
    localization.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Scalar states either side of worst transition
# =========================================================

worst_condition_rows = conditions[
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
    "FC2_mean_rate_hz",
    "FC2_mean_v",
    "PFL2_mean_rate_hz",
    "PFL2_mean_v",
    "PFL3_mean_rate_hz",
    "PFL3_mean_v",
    "DNa03_mean_rate_hz",
    "DNa03_mean_v",
    "DNa02_mean_rate_hz",
    "DNa02_mean_v",
    "PFL2_rate_diff",
    "PFL3_rate_diff",
    "DNa03_rate_diff",
    "DNa02_rate_diff",
    "PFL2_v_diff",
    "PFL3_v_diff",
    "DNa03_v_diff",
    "DNa02_v_diff",
    "activity_difference",
    "steering",
]


available_detail_columns = [
    column
    for column
    in detail_columns
    if column
    in worst_condition_rows.columns
]


print()

print("=" * 220)
print(
    "RAW LAYER STATES AROUND WORST TRANSITION"
)
print("=" * 220)

print()


print(
    worst_condition_rows[
        available_detail_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.5f}",
    )
)


# =========================================================
# Full trajectory around current worst case
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
    "active_fc2_count",
    "PFL2_rate_diff",
    "PFL3_rate_diff",
    "DNa03_rate_diff",
    "DNa02_rate_diff",
    "PFL2_v_diff",
    "PFL3_v_diff",
    "DNa03_v_diff",
    "DNa02_v_diff",
    "activity_difference",
    "steering",
]


available_trajectory_columns = [
    column
    for column
    in trajectory_columns
    if column
    in worst_trajectory.columns
]


print()

print("=" * 190)
print(
    "FULL WORST TRAJECTORY - LATERAL SIGNALS"
)
print("=" * 190)

print()


print(
    worst_trajectory[
        available_trajectory_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.5f}",
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
    VECTOR_SUMMARY_FILE
)

print(
    LATERAL_SUMMARY_FILE
)