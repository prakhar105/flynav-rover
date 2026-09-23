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
    / "fc2_population_recruitment.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "fc2_population_recruitment_summary.csv"
)


# =========================================================
# Frozen neural calibration
#
# DO NOT tune these.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN = 3.0

SPIKE_SCALE = 5.0
ACTIVITY_SCALE = 1.10599

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
# Normalize IDs
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


if len(neurons) != 220:

    raise RuntimeError(
        f"Expected 220 neurons, got {len(neurons)}"
    )


if len(edges) != 3231:

    raise RuntimeError(
        f"Expected 3231 edges, got {len(edges)}"
    )


print("=" * 120)
print("FC2 POPULATION RECRUITMENT TEST")
print("=" * 120)

print()

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)


# =========================================================
# Body index
# =========================================================

body_to_index = {

    int(body_id): index

    for index, body_id
    in enumerate(
        neurons["bodyId"]
    )
}


# =========================================================
# Parsers
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


# =========================================================
# FC2 groups
#
# IMPORTANT:
# Sort by bodyId so subset recruitment is deterministic.
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
        (
            int(
                row[
                    "bodyId"
                ]
            ),
            index,
        )
    )


for column in fc2_groups:

    fc2_groups[
        column
    ] = [

        index

        for _, index
        in sorted(
            fc2_groups[
                column
            ],
            key=lambda x:
                x[0],
        )
    ]


fc2_count = sum(

    len(indices)

    for indices
    in fc2_groups.values()
)


if fc2_count != 92:

    raise RuntimeError(
        f"Expected 92 FC2 neurons, "
        f"got {fc2_count}"
    )


print()

print("FC2 neurons by column:")


for column in range(
    1,
    10,
):

    print(
        f"C{column}: "
        f"{len(fc2_groups[column])}"
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


mask_other = (

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
    mask_other
].copy()


accounted = (

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


if accounted != len(edges):

    raise RuntimeError(
        f"Edge accounting mismatch: "
        f"{accounted} != {len(edges)}"
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

        source = int(
            row[
                "source_bodyId"
            ]
        )

        target = int(
            row[
                "target_bodyId"
            ]
        )


        if (
            source not in body_to_index
            or
            target not in body_to_index
        ):

            continue


        sources.append(
            body_to_index[
                source
            ]
        )

        targets.append(
            body_to_index[
                target
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
# Population recruitment
#
# alpha=0:
#     all A neurons active
#     zero B neurons active
#
# alpha=1:
#     zero A neurons active
#     all B neurons active
#
# intermediate:
#     recruit a fraction from both populations.
#
# Each selected neuron gets FULL FC2_DRIVE = 1.35.
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


    group_a = fc2_groups[
        column_a
    ]


    group_b = fc2_groups[
        column_b
    ]


    n_a_total = len(
        group_a
    )


    n_b_total = len(
        group_b
    )


    # -----------------------------------------------------
    # Number recruited from each population
    # -----------------------------------------------------

    n_a = int(
        round(
            (
                1.0
                -
                alpha
            )
            *
            n_a_total
        )
    )


    n_b = int(
        round(
            alpha
            *
            n_b_total
        )
    )


    n_a = int(
        np.clip(
            n_a,
            0,
            n_a_total,
        )
    )


    n_b = int(
        np.clip(
            n_b,
            0,
            n_b_total,
        )
    )


    # -----------------------------------------------------
    # Deterministic subset
    #
    # A shrinks from the END.
    # B grows from the START.
    # -----------------------------------------------------

    recruited_a = (
        group_a[
            :n_a
        ]
    )


    recruited_b = (
        group_b[
            :n_b
        ]
    )


    return (
        recruited_a,
        recruited_b,
        n_a,
        n_b,
        n_a_total,
        n_b_total,
    )


# =========================================================
# One neural condition
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
    # FC2 recruitment
    # =====================================================

    (
        recruited_a,
        recruited_b,
        n_a,
        n_b,
        n_a_total,
        n_b_total,
    ) = get_recruited_neurons(
        column_a,
        column_b,
        alpha,
    )


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
    # Delta7 inhibitory
    # =====================================================

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
    # DNa03 -> DNa02
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
    # DNa state
    # =====================================================

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

        "a_recruited":
            int(n_a),

        "a_total":
            int(n_a_total),

        "b_recruited":
            int(n_b),

        "b_total":
            int(n_b_total),

        "total_recruited":
            int(
                n_a
                +
                n_b
            ),

        "DNa03_L_spikes":
            dna03_l,

        "DNa03_R_spikes":
            dna03_r,

        "DNa02_L_spikes":
            dna02_l,

        "DNa02_R_spikes":
            dna02_r,

        "DNa02_L_v":
            dna02_l_v,

        "DNa02_R_v":
            dna02_r_v,

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
print("RUNNING FC2 POPULATION RECRUITMENT")
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
# Endpoint validation
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

            target_column = int(
                row[
                    "column_a"
                ]
            )


        elif np.isclose(
            alpha,
            1.0,
        ):

            target_column = int(
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
            target_column,
        )


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
            "PASS: population recruitment endpoints "
            "reproduce frozen Step 27."
        )


# =========================================================
# Trajectory summary
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


    nonzero = [

        sign

        for sign
        in signs

        if sign != 0
    ]


    sign_transitions = sum(

        1

        for index in range(
            1,
            len(nonzero),
        )

        if (
            nonzero[
                index
            ]
            !=
            nonzero[
                index - 1
            ]
        )
    )


    max_step_index = int(
        np.argmax(
            steps
        )
    )


    intermediate = steering[
        1:-1
    ]


    near_zero_intermediate = int(
        (
            np.abs(
                intermediate
            )
            < 0.01
        ).sum()
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
                float(
                    steps.max()
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
                directness,

            "overshoot":
                overshoot,

            "sign_transitions":
                int(
                    sign_transitions
                ),

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


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =========================================================
# Global diagnostic
# =========================================================

print()

print("=" * 120)
print("GLOBAL POPULATION RECRUITMENT DIAGNOSTIC")
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
    f"{len(summary)}"
)


print(
    f"Steps larger than 0.20: "
    f"{int((summary['max_abs_step'] > 0.20).sum())}/"
    f"{len(summary)}"
)


print(
    f"Trajectories with near-zero intermediate state: "
    f"{int((summary['near_zero_intermediate'] > 0).sum())}/"
    f"{len(summary)}"
)


# =========================================================
# Worst trajectories
# =========================================================

print()

print("=" * 160)
print("WORST POPULATION-RECRUITMENT TRAJECTORIES")
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


detail = results[
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

print("=" * 170)
print("DETAIL OF WORST POPULATION-RECRUITMENT TRAJECTORY")
print("=" * 170)

print()


print(
    detail[
        [
            "heading_column",
            "column_a",
            "column_b",
            "alpha",
            "a_recruited",
            "a_total",
            "b_recruited",
            "b_total",
            "total_recruited",
            "DNa03_L_spikes",
            "DNa03_R_spikes",
            "DNa02_L_spikes",
            "DNa02_R_spikes",
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