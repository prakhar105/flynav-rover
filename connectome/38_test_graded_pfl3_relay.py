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


# ============================================================
# STEP 38 - GRADED PFL3 RELAY
#
# Why this exists
# ---------------
# Step 36 showed that the passive PFL3 input/membrane is much
# smoother than the real PFL3 spike-rate output.
#
# Step 37 then showed that simply lowering the hard PFL3 LIF
# threshold does not preserve the frozen endpoint steering map.
#
# This experiment changes ONE interface only:
#
#     PFL3 -> DNa
#
# The real PFL3 event-based output to DNa is replaced with a
# passive, graded PFL3 relay using the SAME connectome-derived
# PFL3->DNa edges and relative weights.
#
# Everything else remains frozen:
#   - 220 real neurons
#   - 3231 selected anatomical edges as source topology
#   - EPG->Delta7 gain = 3.0
#   - PFL->DNa anatomical gain = 3.0
#   - Delta7->PFL inhibitory model hypothesis
#   - Step 31 exact FC2 recruited subsets
#   - all real neurons use threshold 1.0
#   - tau=20 ms, reset=0, refractory=3 ms
#   - PFL2->DNa remains spike/event based
#   - DNa03->DNa02 remains spike/event based
#   - DNa steering decoder remains frozen
#
# Engineering approximation being tested:
#   a shadow PFL3 population receives the SAME upstream spike
#   events as real PFL3 but has no threshold/reset.
#
#   Its positive membrane excursion above baseline is then sent
#   continuously to DNa through the original PFL3->DNa anatomy.
#
# IMPORTANT:
#   The graded relay gain is calibrated ONLY on pure FC2-column
#   endpoints. Interpolated alpha states are not used to choose
#   the gain.
# ============================================================


prefs.codegen.target = "numpy"


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

STEERING_NEURON_FILE = OUTPUT_DIR / "steering_core_neurons.csv"
STEERING_EDGE_FILE = OUTPUT_DIR / "steering_core_edges.csv"
DELTA7_NEURON_FILE = OUTPUT_DIR / "delta7_candidates.csv"
DELTA7_EDGE_FILE = OUTPUT_DIR / "delta7_bridge_edges.csv"

STEP31_RESULT_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment.csv"
)

CALIBRATION_FILE = (
    OUTPUT_DIR
    / "graded_pfl3_gain_calibration.csv"
)

CONDITION_FILE = (
    OUTPUT_DIR
    / "graded_pfl3_conditions.csv"
)

TRAJECTORY_FILE = (
    OUTPUT_DIR
    / "graded_pfl3_trajectories.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "graded_pfl3_summary.csv"
)


# ============================================================
# Frozen parameters
# ============================================================

EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN = 3.0

SPIKE_SCALE = 5.0
ACTIVITY_SCALE = 1.10599

BASELINE_DRIVE = 0.03
EPG_DRIVE = 1.35
FC2_DRIVE = 1.35

SIMULATION_TIME = 300 * ms

SIGN_THRESHOLD = 0.05


# ============================================================
# Graded relay gain candidates
#
# This scale converts the passive shadow-PFL3 membrane state
# into an additional continuous drive at its DNa targets.
#
# It is an engineering calibration parameter, NOT a biological
# synaptic parameter.
# ============================================================

GRADED_GAIN_CANDIDATES = [
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.50,
    1.00,
    2.00,
    4.00,
    8.00,
]


PRIMARY_PAIRS = {
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
}


# ============================================================
# Load Step 31
# ============================================================

if not STEP31_RESULT_FILE.exists():
    raise FileNotFoundError(
        f"Step 31 result file not found:\n"
        f"{STEP31_RESULT_FILE}"
    )


step31_all = pd.read_csv(
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


missing = [
    column
    for column
    in required_step31_columns
    if column not in step31_all.columns
]


if missing:
    raise RuntimeError(
        "Step 31 result file is missing columns: "
        f"{missing}"
    )


if len(step31_all) != 396:
    raise RuntimeError(
        f"Expected 396 Step 31 conditions, "
        f"got {len(step31_all)}."
    )


primary_mask = step31_all.apply(
    lambda row:
        (
            int(row["column_a"]),
            int(row["column_b"]),
        )
        in PRIMARY_PAIRS,
    axis=1,
)


step31 = (
    step31_all.loc[
        primary_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(step31) != 352:
    raise RuntimeError(
        f"Expected 352 primary conditions, "
        f"got {len(step31)}."
    )


# ============================================================
# Load source anatomy
# ============================================================

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


# ============================================================
# Frozen 220-neuron table
# ============================================================

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


if len(neurons) != 220:
    raise RuntimeError(
        f"Expected 220 neurons, "
        f"got {len(neurons)}."
    )


body_ids = set(
    neurons[
        "bodyId"
    ]
)


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
    edges.loc[
        functional_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(edges) != 3231:
    raise RuntimeError(
        f"Expected 3231 functional edges, "
        f"got {len(edges)}."
    )


body_to_index = {
    int(body_id):
        index
    for index, body_id
    in enumerate(
        neurons["bodyId"]
    )
}


# ============================================================
# Helpers
# ============================================================

def normalize_side(value):

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def get_indices(
    neuron_type,
    side=None,
):

    output = []


    for index, row in neurons.iterrows():

        if row["type"] != neuron_type:
            continue


        if side is not None:

            if (
                normalize_side(
                    row.get(
                        "somaSide",
                        "",
                    )
                )
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


def parse_body_ids(value):

    if pd.isna(value):
        return []


    text = str(value).strip()


    if not text:
        return []


    return [
        int(piece.strip())
        for piece in text.split(",")
        if piece.strip()
    ]


def steering_sign(value):

    if value > SIGN_THRESHOLD:
        return 1

    if value < -SIGN_THRESHOLD:
        return -1

    return 0


def safe_corr(x, y):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )


    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    x = x[mask]
    y = y[mask]


    if len(x) < 3:
        return np.nan


    if (
        np.std(x) < 1e-12
        or
        np.std(y) < 1e-12
    ):
        return np.nan


    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


# ============================================================
# EPG groups
# ============================================================

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


    if not 1 <= number <= 8:
        return None


    return (
        f"{side}{number}"
    )


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
    len(values)
    for values
    in epg_groups.values()
) != 46:
    raise RuntimeError(
        "Expected 46 grouped EPG neurons."
    )


# ============================================================
# Populations
# ============================================================

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

dna02_left = get_indices(
    "DNa02",
    "L",
)

dna02_right = get_indices(
    "DNa02",
    "R",
)


if len(pfl2_indices) != 12:
    raise RuntimeError(
        f"Expected 12 PFL2 neurons, "
        f"got {len(pfl2_indices)}."
    )


if len(pfl3_indices) != 24:
    raise RuntimeError(
        f"Expected 24 PFL3 neurons, "
        f"got {len(pfl3_indices)}."
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


if (
    len(dna02_left) != 1
    or
    len(dna02_right) != 1
):
    raise RuntimeError(
        "Expected exactly one left/right DNa02."
    )


# ============================================================
# Real PFL3 -> shadow PFL3 mapping
# ============================================================

pfl3_network_indices = [
    int(index)
    for index
    in pfl3_indices
]


real_pfl3_to_shadow = {
    real_index:
        shadow_index
    for shadow_index, real_index
    in enumerate(
        pfl3_network_indices
    )
}


N_SHADOW_PFL3 = len(
    pfl3_network_indices
)


# ============================================================
# Functional pathways
# ============================================================

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
        f"{accounted_edges} != {len(edges)}"
    )


# Only the upstream edges that target PFL3 are copied to
# the passive PFL3 shadow population.

other_pfl3_edges = other_edges[
    other_edges[
        "target_type"
    ]
    ==
    "PFL3"
].copy()


delta_pfl3_edges = delta_pfl_edges[
    delta_pfl_edges[
        "target_type"
    ]
    ==
    "PFL3"
].copy()


# ============================================================
# Convert edge tables
# ============================================================

def prepare_edges(dataframe):

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


(
    other_pfl3_source,
    other_pfl3_target_real,
    other_pfl3_raw,
) = prepare_edges(
    other_pfl3_edges
)


(
    delta_pfl3_source,
    delta_pfl3_target_real,
    delta_pfl3_raw,
) = prepare_edges(
    delta_pfl3_edges
)


# ============================================================
# Frozen weight transform
# ============================================================

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


def scale_weights(raw_weights):

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

other_pfl3_weights = scale_weights(
    other_pfl3_raw
)

delta_pfl3_weights = scale_weights(
    delta_pfl3_raw
)


# ============================================================
# Shadow target mappings
# ============================================================

def map_real_pfl3_targets_to_shadow(
    real_targets,
):

    output = []


    for real_target in real_targets:

        real_target = int(
            real_target
        )


        if (
            real_target
            not in real_pfl3_to_shadow
        ):
            raise RuntimeError(
                "Non-PFL3 target sent to "
                f"shadow mapping: {real_target}"
            )


        output.append(
            real_pfl3_to_shadow[
                real_target
            ]
        )


    return output


other_pfl3_target_shadow = (
    map_real_pfl3_targets_to_shadow(
        other_pfl3_target_real
    )
)


delta_pfl3_target_shadow = (
    map_real_pfl3_targets_to_shadow(
        delta_pfl3_target_real
    )
)


# ============================================================
# PFL3 -> DNa graded-relay edges
#
# One continuous summed writer is used for ALL PFL3->DNa
# edges to avoid multiple summed-variable writers.
# ============================================================

graded_source_real = (
    list(
        pfl3_dna02_source
    )
    +
    list(
        pfl3_dna03_source
    )
)


graded_target_main = (
    list(
        pfl3_dna02_target
    )
    +
    list(
        pfl3_dna03_target
    )
)


graded_weights = np.concatenate(
    [
        pfl3_dna02_weights,
        pfl3_dna03_weights,
    ]
)


graded_source_shadow = [
    real_pfl3_to_shadow[
        int(
            real_source
        )
    ]
    for real_source
    in graded_source_real
]


# ============================================================
# Synapse helpers
# ============================================================

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


def create_shadow_positive(
    main_group,
    shadow_group,
    source,
    shadow_target,
    weights,
):

    if len(source) == 0:
        return None


    synapses = Synapses(
        main_group,
        shadow_group,
        model="w : 1",
        on_pre="v_post += w",
    )


    synapses.connect(
        i=source,
        j=shadow_target,
    )


    synapses.w = weights


    return synapses


def create_shadow_inhibitory(
    main_group,
    shadow_group,
    source,
    shadow_target,
    weights,
):

    if len(source) == 0:
        return None


    synapses = Synapses(
        main_group,
        shadow_group,
        model="w : 1",
        on_pre="v_post -= w",
    )


    synapses.connect(
        i=source,
        j=shadow_target,
    )


    synapses.w = weights


    return synapses


def create_graded_pfl3_to_dna(
    shadow_group,
    main_group,
    gain,
):

    synapses = Synapses(
        shadow_group,
        main_group,
        model="""
        w : 1
        relay_gain : 1 (shared)
        relay_baseline : 1 (shared)

        graded_drive_post =
            relay_gain
            * w
            * 0.5
            * (
                (v_pre - relay_baseline)
                +
                abs(v_pre - relay_baseline)
            )
            : 1 (summed)
        """,
    )


    synapses.connect(
        i=graded_source_shadow,
        j=graded_target_main,
    )


    synapses.w = (
        graded_weights
        *
        PFL_TO_DNA_GAIN
    )


    synapses.relay_gain = float(
        gain
    )


    synapses.relay_baseline = float(
        BASELINE_DRIVE
    )


    return synapses


# ============================================================
# Run one graded-relay condition
# ============================================================

def run_condition(
    heading_column,
    active_fc2_body_ids,
    graded_gain,
):

    # --------------------------------------------------------
    # Real 220-neuron network.
    #
    # graded_drive is zero for all neurons except DNa targets
    # receiving the continuous PFL3 relay.
    # --------------------------------------------------------

    equations = """
    dv/dt = (-v + drive + graded_drive) / (20*ms) : 1
    drive : 1
    graded_drive : 1
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
    group.drive = BASELINE_DRIVE
    group.graded_drive = 0.0


    # --------------------------------------------------------
    # Passive PFL3 observer / graded relay source.
    # --------------------------------------------------------

    shadow = NeuronGroup(
        N_SHADOW_PFL3,
        """
        dv/dt = (-v + drive) / (20*ms) : 1
        drive : 1
        """,
        method="euler",
    )


    shadow.v = 0.0
    shadow.drive = BASELINE_DRIVE


    # --------------------------------------------------------
    # EPG heading.
    # --------------------------------------------------------

    if heading_column not in epg_groups:
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


    # --------------------------------------------------------
    # Exact Step 31 recruited FC2 subset.
    # --------------------------------------------------------

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


        network_index = (
            body_to_index[
                body_id
            ]
        )


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


    objects = [
        group,
        shadow,
    ]


    # --------------------------------------------------------
    # REAL upstream network remains unchanged.
    # EPG + FC2 -> real PFL2/PFL3.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EPG -> Delta7.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Delta7 -> real PFL inhibitory hypothesis.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Shadow PFL3 receives same direct EPG/FC2 events.
    # --------------------------------------------------------

    shadow_exc = (
        create_shadow_positive(
            group,
            shadow,
            other_pfl3_source,
            other_pfl3_target_shadow,
            other_pfl3_weights,
        )
    )


    if shadow_exc is not None:
        objects.append(
            shadow_exc
        )


    # --------------------------------------------------------
    # Shadow PFL3 receives same Delta7 inhibition.
    # --------------------------------------------------------

    shadow_inh = (
        create_shadow_inhibitory(
            group,
            shadow,
            delta_pfl3_source,
            delta_pfl3_target_shadow,
            delta_pfl3_weights,
        )
    )


    if shadow_inh is not None:
        objects.append(
            shadow_inh
        )


    # --------------------------------------------------------
    # PFL2 -> DNa remains event/spike based.
    #
    # IMPORTANT:
    # Real PFL3 -> DNa spike routes are intentionally omitted.
    # --------------------------------------------------------

    for (
        source,
        target,
        weights,
    ) in [
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


    # --------------------------------------------------------
    # Continuous PFL3 -> DNa relay through original anatomy.
    # --------------------------------------------------------

    graded_synapses = (
        create_graded_pfl3_to_dna(
            shadow,
            group,
            graded_gain,
        )
    )


    objects.append(
        graded_synapses
    )


    # --------------------------------------------------------
    # DNa03 -> DNa02 remains event based.
    # --------------------------------------------------------

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


    objects.append(
        spike_monitor
    )


    network = Network(
        *objects
    )


    network.run(
        SIMULATION_TIME
    )


    counts = np.asarray(
        spike_monitor.count,
        dtype=float,
    )


    final_voltage = np.asarray(
        group.v,
        dtype=float,
    )


    left_spikes = float(
        counts[
            dna02_left
        ].sum()
    )


    right_spikes = float(
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
        "steering":
            steering,

        "activity_difference":
            activity_difference,

        "dna03_total_spikes":
            int(
                counts[
                    dna03_indices
                ].sum()
            ),

        "dna02_total_spikes":
            int(
                counts[
                    dna02_indices
                ].sum()
            ),

        "dna02_left_spikes":
            int(
                left_spikes
            ),

        "dna02_right_spikes":
            int(
                right_spikes
            ),

        "real_pfl3_total_spikes":
            int(
                counts[
                    pfl3_indices
                ].sum()
            ),

        "shadow_pfl3_mean_v":
            float(
                np.mean(
                    np.asarray(
                        shadow.v,
                        dtype=float,
                    )
                )
            ),

        "mean_dna_graded_drive":
            float(
                np.mean(
                    np.asarray(
                        group.graded_drive[
                            np.concatenate(
                                [
                                    dna03_indices,
                                    dna02_indices,
                                ]
                            )
                        ],
                        dtype=float,
                    )
                )
            ),
    }


# ============================================================
# Build 36 unique pure endpoints from Step 31
# ============================================================

endpoint_records = {}


for _, row in step31.iterrows():

    alpha = float(
        row[
            "alpha"
        ]
    )


    if np.isclose(
        alpha,
        0.0,
    ):

        goal_column = int(
            row[
                "column_a"
            ]
        )


    elif np.isclose(
        alpha,
        1.0,
    ):

        goal_column = int(
            row[
                "column_b"
            ]
        )


    else:
        continue


    heading = str(
        row[
            "heading_column"
        ]
    )


    body_ids_list = sorted(
        set(
            parse_body_ids(
                row[
                    "a_body_ids"
                ]
            )
            +
            parse_body_ids(
                row[
                    "b_body_ids"
                ]
            )
        )
    )


    key = (
        heading,
        goal_column,
    )


    record = {
        "heading_column":
            heading,

        "goal_column":
            goal_column,

        "body_ids":
            body_ids_list,

        "frozen_steering":
            float(
                row[
                    "steering"
                ]
            ),
    }


    if key in endpoint_records:

        existing = endpoint_records[
            key
        ]


        if (
            existing[
                "body_ids"
            ]
            !=
            body_ids_list
        ):
            raise RuntimeError(
                "Duplicate pure endpoint has "
                "different FC2 body IDs for "
                f"{key}."
            )


        if not np.isclose(
            existing[
                "frozen_steering"
            ],
            record[
                "frozen_steering"
            ],
            atol=1e-12,
        ):
            raise RuntimeError(
                "Duplicate pure endpoint has "
                "different frozen steering for "
                f"{key}."
            )


    else:
        endpoint_records[
            key
        ] = record


pure_endpoints = list(
    endpoint_records.values()
)


if len(pure_endpoints) != 36:
    raise RuntimeError(
        f"Expected 36 unique pure endpoints, "
        f"got {len(pure_endpoints)}."
    )


# ============================================================
# Calibration sweep - PURE ENDPOINTS ONLY
# ============================================================

print("=" * 120)
print(
    "STEP 38 - GRADED PFL3 RELAY"
)
print("=" * 120)

print()

print(
    f"Real neurons: "
    f"{len(neurons)}"
)

print(
    f"Functional anatomical edges: "
    f"{len(edges)}"
)

print(
    f"Real PFL3 neurons: "
    f"{len(pfl3_indices)}"
)

print(
    f"Passive graded PFL3 relays: "
    f"{N_SHADOW_PFL3}"
)

print(
    f"Unique pure endpoints for gain calibration: "
    f"{len(pure_endpoints)}"
)

print(
    f"Gain candidates: "
    f"{GRADED_GAIN_CANDIDATES}"
)

print()

print(
    "Calibration uses ONLY pure endpoint states."
)

print(
    "Intermediate alpha states are held out "
    "until after gain selection."
)


calibration_rows = []


total_calibration_runs = (
    len(
        GRADED_GAIN_CANDIDATES
    )
    *
    len(
        pure_endpoints
    )
)


run_counter = 0


for gain in GRADED_GAIN_CANDIDATES:

    print()

    print("=" * 120)

    print(
        f"GRADED RELAY GAIN = "
        f"{gain:.6f}"
    )

    print("=" * 120)


    for endpoint in pure_endpoints:

        run_counter += 1


        heading = endpoint[
            "heading_column"
        ]


        goal_column = endpoint[
            "goal_column"
        ]


        print(
            f"[{run_counter:3d}/"
            f"{total_calibration_runs}] "
            f"gain={gain:.6f} | "
            f"heading={heading:>2} | "
            f"C{goal_column}"
        )


        result = run_condition(
            heading_column=heading,
            active_fc2_body_ids=endpoint[
                "body_ids"
            ],
            graded_gain=gain,
        )


        calibration_rows.append(
            {
                "graded_gain":
                    gain,

                "heading_column":
                    heading,

                "goal_column":
                    goal_column,

                "frozen_steering":
                    endpoint[
                        "frozen_steering"
                    ],

                **result,
            }
        )


calibration_conditions = pd.DataFrame(
    calibration_rows
)


# ============================================================
# Calibration metrics
# ============================================================

calibration_summary_rows = []


for gain in GRADED_GAIN_CANDIDATES:

    data = calibration_conditions[
        np.isclose(
            calibration_conditions[
                "graded_gain"
            ],
            gain,
        )
    ]


    frozen = data[
        "frozen_steering"
    ].to_numpy(
        dtype=float,
    )


    actual = data[
        "steering"
    ].to_numpy(
        dtype=float,
    )


    errors = np.abs(
        actual
        -
        frozen
    )


    frozen_sign = np.asarray(
        [
            steering_sign(
                value
            )
            for value
            in frozen
        ]
    )


    actual_sign = np.asarray(
        [
            steering_sign(
                value
            )
            for value
            in actual
        ]
    )


    calibration_summary_rows.append(
        {
            "graded_gain":
                gain,

            "endpoint_conditions":
                len(
                    data
                ),

            "endpoint_mae":
                float(
                    errors.mean()
                ),

            "endpoint_max_error":
                float(
                    errors.max()
                ),

            "endpoint_corr":
                safe_corr(
                    frozen,
                    actual,
                ),

            "endpoint_sign_agreement":
                float(
                    np.mean(
                        frozen_sign
                        ==
                        actual_sign
                    )
                ),

            "dna03_active_fraction":
                float(
                    np.mean(
                        data[
                            "dna03_total_spikes"
                        ].to_numpy(
                            dtype=float
                        )
                        >
                        0
                    )
                ),

            "dna02_active_fraction":
                float(
                    np.mean(
                        data[
                            "dna02_total_spikes"
                        ].to_numpy(
                            dtype=float
                        )
                        >
                        0
                    )
                ),

            "mean_abs_steering":
                float(
                    np.mean(
                        np.abs(
                            actual
                        )
                    )
                ),

            "mean_graded_drive":
                float(
                    data[
                        "mean_dna_graded_drive"
                    ].mean()
                ),
        }
    )


calibration_summary = pd.DataFrame(
    calibration_summary_rows
)


calibration_summary[
    "endpoint_preservation_pass"
] = (
    (
        calibration_summary[
            "endpoint_mae"
        ]
        <=
        0.10
    )
    &
    (
        calibration_summary[
            "endpoint_corr"
        ]
        >=
        0.80
    )
    &
    (
        calibration_summary[
            "endpoint_sign_agreement"
        ]
        >=
        0.80
    )
    &
    (
        calibration_summary[
            "dna02_active_fraction"
        ]
        >=
        0.75
    )
)


# ============================================================
# Choose gain WITHOUT looking at interpolation states
# ============================================================

passing = calibration_summary[
    calibration_summary[
        "endpoint_preservation_pass"
    ]
].copy()


if len(
    passing
) > 0:

    selected_row = (
        passing
        .sort_values(
            [
                "endpoint_mae",
                "endpoint_max_error",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .iloc[
            0
        ]
    )


    selection_reason = (
        "lowest endpoint MAE among gains "
        "passing endpoint-preservation gates"
    )


else:

    # No gain passes the desired preservation gate.
    #
    # We still evaluate the most endpoint-faithful gain so
    # that the failure/success of the graded relay can be
    # measured on held-out interpolated conditions.

    selectable = (
        calibration_summary
        .copy()
    )


    selectable[
        "_corr_sort"
    ] = selectable[
        "endpoint_corr"
    ].fillna(
        -999.0
    )


    selected_row = (
        selectable
        .sort_values(
            [
                "endpoint_mae",
                "_corr_sort",
                "endpoint_sign_agreement",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .iloc[
            0
        ]
    )


    selection_reason = (
        "NO gain passed preservation gates; "
        "selected lowest endpoint MAE for diagnostic "
        "full interpolation evaluation"
    )


SELECTED_GAIN = float(
    selected_row[
        "graded_gain"
    ]
)


calibration_summary[
    "selected_for_full_test"
] = np.isclose(
    calibration_summary[
        "graded_gain"
    ],
    SELECTED_GAIN,
)


calibration_summary.to_csv(
    CALIBRATION_FILE,
    index=False,
)


print()

print("=" * 170)

print(
    "PURE-ENDPOINT GAIN CALIBRATION"
)

print("=" * 170)

print()


print(
    calibration_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print()

print(
    f"Selected graded relay gain: "
    f"{SELECTED_GAIN:.6f}"
)

print(
    f"Selection reason: "
    f"{selection_reason}"
)


# ============================================================
# Full primary interpolation test
#
# Gain has already been selected above using pure endpoints.
# Now the 320 adjacent-alpha transitions are evaluated.
# ============================================================

print()

print("=" * 120)

print(
    "FULL PRIMARY INTERPOLATION TEST"
)

print("=" * 120)

print()

print(
    f"Selected gain: "
    f"{SELECTED_GAIN:.6f}"
)

print(
    f"Conditions: "
    f"{len(step31)}"
)


condition_rows = []


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


    print(
        f"[{condition_number + 1:3d}/"
        f"{len(step31)}] "
        f"heading={heading:>2} | "
        f"C{column_a}->C{column_b} | "
        f"alpha={alpha:.1f}"
    )


    result = run_condition(
        heading_column=heading,
        active_fc2_body_ids=active_body_ids,
        graded_gain=SELECTED_GAIN,
    )


    condition_rows.append(
        {
            "condition_id":
                len(
                    condition_rows
                ),

            "graded_gain":
                SELECTED_GAIN,

            "heading_column":
                heading,

            "column_a":
                column_a,

            "column_b":
                column_b,

            "alpha":
                alpha,

            "frozen_steering":
                float(
                    source_row[
                        "steering"
                    ]
                ),

            **result,
        }
    )


conditions = pd.DataFrame(
    condition_rows
)


conditions.to_csv(
    CONDITION_FILE,
    index=False,
)


# ============================================================
# Trajectory metrics helper
# ============================================================

def build_trajectory_metrics(
    dataframe,
    steering_column,
    method_name,
):

    rows = []


    for (
        heading,
        column_a,
        column_b,
    ), group in dataframe.groupby(
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
            steering_column
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


        total_variation = float(
            steps.sum()
        )


        directness = (
            abs(
                end
                -
                start
            )
            /
            total_variation

            if total_variation
            >
            1e-12

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


        overshoot = max(
            max(
                0.0,
                endpoint_min
                -
                float(
                    steering.min()
                ),
            ),
            max(
                0.0,
                float(
                    steering.max()
                )
                -
                endpoint_max,
            ),
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


        sign_transitions = sum(
            1
            for index
            in range(
                1,
                len(
                    nonzero_signs
                ),
            )
            if (
                nonzero_signs[
                    index
                ]
                !=
                nonzero_signs[
                    index - 1
                ]
            )
        )


        rows.append(
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

                "mean_abs_step":
                    float(
                        steps.mean()
                    ),

                "max_abs_step":
                    float(
                        steps.max()
                    ),

                "directness_ratio":
                    float(
                        directness
                    ),

                "overshoot":
                    float(
                        overshoot
                    ),

                "sign_transitions":
                    int(
                        sign_transitions
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


graded_trajectories = (
    build_trajectory_metrics(
        conditions,
        "steering",
        "graded_PFL3",
    )
)


frozen_trajectories = (
    build_trajectory_metrics(
        conditions,
        "frozen_steering",
        "frozen_step31",
    )
)


trajectories = pd.concat(
    [
        frozen_trajectories,
        graded_trajectories,
    ],
    ignore_index=True,
)


trajectories.to_csv(
    TRAJECTORY_FILE,
    index=False,
)


# ============================================================
# Aggregate comparison
# ============================================================

summary_rows = []


for method in [
    "frozen_step31",
    "graded_PFL3",
]:

    data = trajectories[
        trajectories[
            "method"
        ]
        ==
        method
    ]


    summary_rows.append(
        {
            "method":
                method,

            "selected_gain":
                (
                    SELECTED_GAIN
                    if method
                    ==
                    "graded_PFL3"
                    else np.nan
                ),

            "trajectories":
                len(
                    data
                ),

            "mean_adjacent_step":
                float(
                    data[
                        "mean_abs_step"
                    ].mean()
                ),

            "mean_trajectory_max_step":
                float(
                    data[
                        "max_abs_step"
                    ].mean()
                ),

            "worst_step":
                float(
                    data[
                        "max_abs_step"
                    ].max()
                ),

            "mean_directness":
                float(
                    data[
                        "directness_ratio"
                    ].mean()
                ),

            "max_overshoot":
                float(
                    data[
                        "overshoot"
                    ].max()
                ),

            "trajectories_with_sign_flip":
                int(
                    (
                        data[
                            "sign_transitions"
                        ]
                        >
                        0
                    ).sum()
                ),

            "trajectories_with_multi_sign_flip":
                int(
                    (
                        data[
                            "sign_transitions"
                        ]
                        >
                        1
                    ).sum()
                ),

            "trajectories_step_gt_0_20":
                int(
                    (
                        data[
                            "max_abs_step"
                        ]
                        >
                        0.20
                    ).sum()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Endpoint preservation on selected full test
# ============================================================

endpoint_conditions = conditions[
    np.isclose(
        conditions[
            "alpha"
        ],
        0.0,
    )
    |
    np.isclose(
        conditions[
            "alpha"
        ],
        1.0,
    )
].copy()


endpoint_error = np.abs(
    endpoint_conditions[
        "steering"
    ].to_numpy(
        dtype=float,
    )
    -
    endpoint_conditions[
        "frozen_steering"
    ].to_numpy(
        dtype=float,
    )
)


endpoint_corr = safe_corr(
    endpoint_conditions[
        "frozen_steering"
    ],
    endpoint_conditions[
        "steering"
    ],
)


endpoint_sign_agreement = float(
    np.mean(
        np.asarray(
            [
                steering_sign(
                    value
                )
                for value
                in endpoint_conditions[
                    "frozen_steering"
                ]
            ]
        )
        ==
        np.asarray(
            [
                steering_sign(
                    value
                )
                for value
                in endpoint_conditions[
                    "steering"
                ]
            ]
        )
    )
)


graded_index = summary[
    summary[
        "method"
    ]
    ==
    "graded_PFL3"
].index[
    0
]


summary.loc[
    graded_index,
    "endpoint_mae_vs_frozen",
] = float(
    endpoint_error.mean()
)


summary.loc[
    graded_index,
    "endpoint_max_error_vs_frozen",
] = float(
    endpoint_error.max()
)


summary.loc[
    graded_index,
    "endpoint_corr_vs_frozen",
] = float(
    endpoint_corr
)


summary.loc[
    graded_index,
    "endpoint_sign_agreement",
] = float(
    endpoint_sign_agreement
)


summary.loc[
    graded_index,
    "dna03_active_fraction",
] = float(
    np.mean(
        conditions[
            "dna03_total_spikes"
        ].to_numpy(
            dtype=float
        )
        >
        0
    )
)


summary.loc[
    graded_index,
    "dna02_active_fraction",
] = float(
    np.mean(
        conditions[
            "dna02_total_spikes"
        ].to_numpy(
            dtype=float
        )
        >
        0
    )
)


summary.loc[
    summary[
        "method"
    ]
    ==
    "frozen_step31",
    [
        "endpoint_mae_vs_frozen",
        "endpoint_max_error_vs_frozen",
        "endpoint_corr_vs_frozen",
        "endpoint_sign_agreement",
    ],
] = [
    0.0,
    0.0,
    1.0,
    1.0,
]


# ============================================================
# Improvement metrics
# ============================================================

frozen_row = summary[
    summary[
        "method"
    ]
    ==
    "frozen_step31"
].iloc[
    0
]


graded_row = summary[
    summary[
        "method"
    ]
    ==
    "graded_PFL3"
].iloc[
    0
]


summary.loc[
    graded_index,
    "worst_step_reduction_pct",
] = (
    100.0
    *
    (
        float(
            frozen_row[
                "worst_step"
            ]
        )
        -
        float(
            graded_row[
                "worst_step"
            ]
        )
    )
    /
    float(
        frozen_row[
            "worst_step"
        ]
    )
)


summary.loc[
    graded_index,
    "mean_max_step_reduction_pct",
] = (
    100.0
    *
    (
        float(
            frozen_row[
                "mean_trajectory_max_step"
            ]
        )
        -
        float(
            graded_row[
                "mean_trajectory_max_step"
            ]
        )
    )
    /
    float(
        frozen_row[
            "mean_trajectory_max_step"
        ]
    )
)


# ============================================================
# Explicit engineering go/no-go screen
# ============================================================

graded_row = summary[
    summary[
        "method"
    ]
    ==
    "graded_PFL3"
].iloc[
    0
]


go_endpoint = (
    float(
        graded_row[
            "endpoint_mae_vs_frozen"
        ]
    )
    <=
    0.10
    and
    float(
        graded_row[
            "endpoint_corr_vs_frozen"
        ]
    )
    >=
    0.80
    and
    float(
        graded_row[
            "endpoint_sign_agreement"
        ]
    )
    >=
    0.80
)


go_smoothness = (
    float(
        graded_row[
            "worst_step"
        ]
    )
    <=
    0.80
    *
    float(
        frozen_row[
            "worst_step"
        ]
    )
    or
    float(
        graded_row[
            "mean_trajectory_max_step"
        ]
    )
    <=
    0.75
    *
    float(
        frozen_row[
            "mean_trajectory_max_step"
        ]
    )
)


go_activity = (
    float(
        graded_row[
            "dna02_active_fraction"
        ]
    )
    >=
    0.70
)


GO_FOR_VALIDATION = bool(
    go_endpoint
    and
    go_smoothness
    and
    go_activity
)


summary.loc[
    graded_index,
    "screen_endpoint_preserved",
] = go_endpoint


summary.loc[
    graded_index,
    "screen_smoothness_improved",
] = go_smoothness


summary.loc[
    graded_index,
    "screen_dna02_activity_preserved",
] = go_activity


summary.loc[
    graded_index,
    "screen_go_for_full_heading_validation",
] = (
    GO_FOR_VALIDATION
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Output
# ============================================================

print()

print("=" * 180)

print(
    "GRADED PFL3 RELAY VS FROZEN STEP 31"
)

print("=" * 180)

print()


print(
    summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# ============================================================
# Worst trajectory comparison
# ============================================================

print()

print("=" * 170)

print(
    "WORST TRAJECTORY BY METHOD"
)

print("=" * 170)

print()


worst_rows = []


for method in [
    "frozen_step31",
    "graded_PFL3",
]:

    data = trajectories[
        trajectories[
            "method"
        ]
        ==
        method
    ]


    worst = (
        data
        .sort_values(
            "max_abs_step",
            ascending=False,
        )
        .iloc[
            0
        ]
    )


    worst_rows.append(
        worst
    )


print(
    pd.DataFrame(
        worst_rows
    )[
        [
            "method",
            "heading_column",
            "column_a",
            "column_b",
            "mean_abs_step",
            "max_abs_step",
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


# ============================================================
# Selected graded worst trajectory in detail
# ============================================================

graded_worst = (
    graded_trajectories
    .sort_values(
        "max_abs_step",
        ascending=False,
    )
    .iloc[
        0
    ]
)


worst_heading = str(
    graded_worst[
        "heading_column"
    ]
)


worst_a = int(
    graded_worst[
        "column_a"
    ]
)


worst_b = int(
    graded_worst[
        "column_b"
    ]
)


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
].sort_values(
    "alpha"
)


print()

print("=" * 180)

print(
    "SELECTED GRADED RELAY - WORST TRAJECTORY DETAIL"
)

print("=" * 180)

print()


print(
    worst_conditions[
        [
            "alpha",
            "frozen_steering",
            "steering",
            "dna03_total_spikes",
            "dna02_total_spikes",
            "dna02_left_spikes",
            "dna02_right_spikes",
            "mean_dna_graded_drive",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.5f}",
    )
)


# ============================================================
# Final go/no-go
# ============================================================

print()

print("=" * 150)

print(
    "STEP 38 DECISION SCREEN"
)

print("=" * 150)

print()

print(
    f"Endpoint preservation pass: "
    f"{go_endpoint}"
)

print(
    f"Smoothness improvement pass: "
    f"{go_smoothness}"
)

print(
    f"DNa02 activity pass: "
    f"{go_activity}"
)

print()


if GO_FOR_VALIDATION:

    print(
        "GO: graded PFL3 relay is suitable for "
        "full-heading validation."
    )

else:

    print(
        "NO-GO: graded PFL3 relay does not yet "
        "satisfy the combined preservation/smoothness "
        "screen."
    )


print()

print("=" * 120)

print(
    "FILES CREATED"
)

print("=" * 120)

print(
    CALIBRATION_FILE
)

print(
    CONDITION_FILE
)

print(
    TRAJECTORY_FILE
)

print(
    SUMMARY_FILE
)
