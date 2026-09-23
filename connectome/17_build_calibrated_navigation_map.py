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

RESULT_FILE = (
    OUTPUT_DIR
    / "calibrated_navigation_map.csv"
)

STEERING_MATRIX_FILE = (
    OUTPUT_DIR
    / "calibrated_steering_matrix.csv"
)

MEMBRANE_MATRIX_FILE = (
    OUTPUT_DIR
    / "calibrated_membrane_matrix.csv"
)

SPIKE_MATRIX_FILE = (
    OUTPUT_DIR
    / "calibrated_spike_matrix.csv"
)


# =========================================================
# Working surrogate calibration
#
# These are NOT claimed biological gains.
#
# They are simulator calibration values found by our
# controlled pathway sweeps.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN = 3.0


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
# Normalize
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
# Neurons
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
# Edges
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
# Functional navigation circuit
# =========================================================

functional = (

    # EPG -> Delta7
    (
        (edges["source_type"] == "EPG")
        &
        (edges["target_type"] == "Delta7")
    )

    |

    # Direct EPG -> PFL
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

    # Delta7 -> PFL
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

    # FC2 -> PFL
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

    # PFL -> DNa
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
print("CALIBRATED MALECNS NAVIGATION MAP")
print("=" * 100)

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)

print(
    f"EPG -> Delta7 gain: "
    f"{EPG_TO_DELTA7_GAIN:.1f}x"
)

print(
    f"PFL -> DNa gain: "
    f"{PFL_TO_DNA_GAIN:.1f}x"
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
# EPG anatomical groups
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
# FC2 anatomical groups
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
# Validate
# =========================================================

epg_count = sum(
    len(x)
    for x in epg_groups.values()
)


fc2_count = sum(
    len(x)
    for x in fc2_groups.values()
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
# Population indices
# =========================================================

def get_indices(
    neuron_type,
    side=None,
):

    output = []


    for index, row in neurons.iterrows():

        if row["type"] != neuron_type:
            continue

        if (
            side is not None
            and
            row.get(
                "somaSide",
                None,
            )
            != side
        ):
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


dna02_left = get_indices(
    "DNa02",
    "L",
)

dna02_right = get_indices(
    "DNa02",
    "R",
)


# =========================================================
# Separate pathways
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


pfl_dna_edges = edges[
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
                "DNa03",
                "DNa02",
            ]
        )
    )
].copy()


other_edges = edges[
    ~(
        (
            (edges["source_type"] == "EPG")
            &
            (edges["target_type"] == "Delta7")
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
                    "PFL2",
                    "PFL3",
                ]
            )
            &
            edges["target_type"].isin(
                [
                    "DNa03",
                    "DNa02",
                ]
            )
        )
    )
].copy()


print()

print(
    f"EPG -> Delta7 edges: "
    f"{len(epg_delta_edges)}"
)

print(
    f"Delta7 -> PFL edges: "
    f"{len(delta_pfl_edges)}"
)

print(
    f"PFL -> DNa edges: "
    f"{len(pfl_dna_edges)}"
)

print(
    f"Other edges: "
    f"{len(other_edges)}"
)


# =========================================================
# Prepare edge arrays
# =========================================================

def prepare_edges(df):

    sources = []
    targets = []
    raw_weights = []


    for _, row in df.iterrows():

        source = int(
            row["source_bodyId"]
        )

        target = int(
            row["target_bodyId"]
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
    pfl_dna_source,
    pfl_dna_target,
    pfl_dna_raw,
) = prepare_edges(
    pfl_dna_edges
)


(
    other_source,
    other_target,
    other_raw,
) = prepare_edges(
    other_edges
)


# =========================================================
# Common anatomical scaling
# =========================================================

all_raw = np.concatenate(
    [
        epg_delta_raw,
        delta_pfl_raw,
        pfl_dna_raw,
        other_raw,
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


epg_delta_weights = scale_weights(
    epg_delta_raw
)

delta_pfl_weights = scale_weights(
    delta_pfl_raw
)

pfl_dna_weights = scale_weights(
    pfl_dna_raw
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
# One anatomical heading + goal condition
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
    # External input
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
    # Other positive edges
    # -----------------------------------------------------

    S_other = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )


    S_other.connect(
        i=other_source,
        j=other_target,
    )


    S_other.w = (
        other_weights
    )


    # -----------------------------------------------------
    # EPG -> Delta7
    # -----------------------------------------------------

    S_epg_delta = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )


    S_epg_delta.connect(
        i=epg_delta_source,
        j=epg_delta_target,
    )


    S_epg_delta.w = (
        epg_delta_weights
        * EPG_TO_DELTA7_GAIN
    )


    # -----------------------------------------------------
    # Delta7 -> PFL inhibitory
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # PFL -> DNa
    # -----------------------------------------------------

    S_pfl_dna = Synapses(
        G,
        G,
        model="w : 1",
        on_pre="v_post += w",
    )


    S_pfl_dna.connect(
        i=pfl_dna_source,
        j=pfl_dna_target,
    )


    S_pfl_dna.w = (
        pfl_dna_weights
        * PFL_TO_DNA_GAIN
    )


    # -----------------------------------------------------
    # Monitor
    # -----------------------------------------------------

    monitor = SpikeMonitor(
        G
    )


    network = Network(
        G,
        S_other,
        S_epg_delta,
        S_delta_pfl,
        S_pfl_dna,
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


    # =====================================================
    # Layer activity
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
    # DNa02 output
    # =====================================================

    left_spikes = safe_sum(
        counts,
        dna02_left,
    )

    right_spikes = safe_sum(
        counts,
        dna02_right,
    )


    left_v = safe_mean(
        voltage,
        dna02_left,
    )

    right_v = safe_mean(
        voltage,
        dna02_right,
    )


    membrane_difference = (
        right_v
        -
        left_v
    )


    spike_total = (
        left_spikes
        +
        right_spikes
    )


    if spike_total > 0:

        spike_difference = (
            right_spikes
            -
            left_spikes
        ) / spike_total

    else:

        spike_difference = 0.0


    # -----------------------------------------------------
    # Existing fused output.
    #
    # We save BOTH components separately because we need
    # to determine whether 80/20 is suitable before Webots.
    # -----------------------------------------------------

    if spike_total > 0:

        steering = (
            0.80
            * spike_difference
            +
            0.20
            * membrane_difference
        )

    else:

        steering = (
            membrane_difference
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

        "DNa02_spikes":
            dna02_spikes,

        "DNa02_L_spikes":
            left_spikes,

        "DNa02_R_spikes":
            right_spikes,

        "DNa02_L_v":
            left_v,

        "DNa02_R_v":
            right_v,

        "membrane_difference":
            membrane_difference,

        "spike_difference":
            spike_difference,

        "steering":
            float(
                steering
            ),
    }


# =========================================================
# Full 16 x 9 map
# =========================================================

rows = []

total_conditions = (
    len(EPG_ORDER)
    * 9
)

condition_number = 0


print()
print("=" * 100)
print("BUILDING CALIBRATED 16 x 9 MAP")
print("=" * 100)
print()


for heading in EPG_ORDER:

    for goal in range(
        1,
        10,
    ):

        condition_number += 1


        print(
            f"[{condition_number:3d}/"
            f"{total_conditions}] "
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


results = pd.DataFrame(
    rows
)


results.to_csv(
    RESULT_FILE,
    index=False,
)


# =========================================================
# Matrices
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
        f"C{x}"
        for x in output.columns
    ]


    return output


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
# Fused steering matrix
# =========================================================

print()
print("=" * 120)
print("CALIBRATED FUSED STEERING MATRIX")
print("=" * 120)
print()


print(
    steering_matrix.to_string(
        float_format=lambda x:
        f"{x:+.3f}"
    )
)


# =========================================================
# Membrane-only matrix
# =========================================================

print()
print("=" * 120)
print("DNa02 MEMBRANE DIFFERENCE MATRIX")
print("=" * 120)
print()


print(
    membrane_matrix.to_string(
        float_format=lambda x:
        f"{x:+.3f}"
    )
)


# =========================================================
# Spike-only matrix
# =========================================================

print()
print("=" * 120)
print("DNa02 SPIKE DIFFERENCE MATRIX")
print("=" * 120)
print()


print(
    spike_matrix.to_string(
        float_format=lambda x:
        f"{x:+.3f}"
    )
)


# =========================================================
# Direction symbols
# =========================================================

NEUTRAL_THRESHOLD = 0.05


def direction_symbol(value):

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
    ] = (
        direction_matrix[
            column
        ].map(
            direction_symbol
        )
    )


print()
print("=" * 120)
print("CALIBRATED DIRECTION MAP")
print("=" * 120)
print()


print(
    direction_matrix.to_string()
)


# =========================================================
# Activity summary
# =========================================================

print()
print("=" * 120)
print("NETWORK ACTIVITY SUMMARY")
print("=" * 120)
print()


print(
    f"Conditions: "
    f"{len(results)}"
)


print(
    f"Conditions with Delta7 spikes: "
    f"{int((results['Delta7_spikes'] > 0).sum())}"
    f"/{len(results)}"
)


print(
    f"Conditions with DNa03 spikes: "
    f"{int((results['DNa03_spikes'] > 0).sum())}"
    f"/{len(results)}"
)


print(
    f"Conditions with DNa02 spikes: "
    f"{int((results['DNa02_spikes'] > 0).sum())}"
    f"/{len(results)}"
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
    f"Mean |fused steering|: "
    f"{results['steering'].abs().mean():.4f}"
)


# =========================================================
# Strongest outputs
# =========================================================

print()
print("=" * 120)
print("STRONGEST RIGHT CONDITIONS")
print("=" * 120)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",
            "DNa02_L_spikes",
            "DNa02_R_spikes",
            "membrane_difference",
            "spike_difference",
            "steering",
        ]
    ]
    .sort_values(
        "steering",
        ascending=False,
    )
    .head(15)
    .to_string(
        index=False
    )
)


print()
print("=" * 120)
print("STRONGEST LEFT CONDITIONS")
print("=" * 120)
print()


print(
    results[
        [
            "heading_column",
            "goal_column",
            "DNa02_L_spikes",
            "DNa02_R_spikes",
            "membrane_difference",
            "spike_difference",
            "steering",
        ]
    ]
    .sort_values(
        "steering",
        ascending=True,
    )
    .head(15)
    .to_string(
        index=False
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
    STEERING_MATRIX_FILE
)

print(
    MEMBRANE_MATRIX_FILE
)

print(
    SPIKE_MATRIX_FILE
)