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
    / "epg_delta7_gain_sweep.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "epg_delta7_gain_summary.csv"
)


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
    edges["source_bodyId"]
    .isin(body_ids)
    &
    edges["target_bodyId"]
    .isin(body_ids)
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
# Keep navigation circuit
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
        edges["target_type"]
        .isin(
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
        edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )

    |

    # FC2 -> PFL
    (
        edges["source_type"]
        .isin(
            [
                "FC2A",
                "FC2B",
                "FC2C",
            ]
        )
        &
        edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )

    |

    # PFL -> DNa
    (
        edges["source_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
        &
        edges["target_type"]
        .isin(
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
print("EPG -> DELTA7 GAIN SWEEP")
print("=" * 100)

print(
    f"Neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
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

if (
    sum(
        len(x)
        for x in epg_groups.values()
    )
    != 46
):

    raise RuntimeError(
        "EPG grouping failed."
    )


if (
    sum(
        len(x)
        for x in fc2_groups.values()
    )
    != 92
):

    raise RuntimeError(
        "FC2 grouping failed."
    )


# =========================================================
# Population indices
# =========================================================

def get_indices(
    neuron_type,
    side=None,
):

    result = []

    for index, row in neurons.iterrows():

        if row["type"] != neuron_type:
            continue

        if (
            side is not None
            and row.get(
                "somaSide",
                None,
            ) != side
        ):
            continue

        result.append(
            index
        )

    return np.asarray(
        result,
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
# Split biological pathways
#
# Experiment variable:
#
# ONLY EPG -> Delta7 changes gain.
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
        edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
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
            edges["target_type"]
            .isin(
                [
                    "PFL2",
                    "PFL3",
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
    f"Other navigation edges: "
    f"{len(other_edges)}"
)


# =========================================================
# Prepare edges
# =========================================================

def prepare_edges(df):

    sources = []
    targets = []
    raw_weights = []


    for _, row in df.iterrows():

        source_body = int(
            row["source_bodyId"]
        )

        target_body = int(
            row["target_bodyId"]
        )

        if (
            source_body
            not in body_to_index
            or
            target_body
            not in body_to_index
        ):
            continue


        sources.append(
            body_to_index[
                source_body
            ]
        )

        targets.append(
            body_to_index[
                target_body
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
    other_source,
    other_target,
    other_raw,
) = prepare_edges(
    other_edges
)


# =========================================================
# Common weight normalization
# =========================================================

all_raw = np.concatenate(
    [
        epg_delta_raw,
        delta_pfl_raw,
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

other_weights = scale_weights(
    other_raw
)


# =========================================================
# Utility
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
# Simulation
# =========================================================

def run_condition(
    heading_column,
    goal_column,
    epg_delta_gain,
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
    # Keep sensory drives FIXED.
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
    # Normal positive navigation synapses
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
    #
    # ONLY experimental variable.
    #
    # Biological topology and relative weights stay
    # unchanged. One uniform surrogate gain is applied.
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
        * epg_delta_gain
    )


    # -----------------------------------------------------
    # Delta7 -> PFL remains inhibitory.
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


    # -----------------------------------------------------
    # Delta7
    # -----------------------------------------------------

    delta7_spikes = safe_sum(
        counts,
        delta7_indices,
    )


    delta7_spiking_neurons = int(
        (
            counts[
                delta7_indices
            ] > 0
        ).sum()
    )


    delta7_mean_v = safe_mean(
        voltage,
        delta7_indices,
    )


    delta7_max_v = float(
        voltage[
            delta7_indices
        ].max()
    )


    # -----------------------------------------------------
    # Downstream layers
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # DNa02 steering
    # -----------------------------------------------------

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


    spike_total = (
        left_spikes
        +
        right_spikes
    )


    if spike_total > 0:

        spike_signal = (
            right_spikes
            - left_spikes
        ) / spike_total

    else:

        spike_signal = 0.0


    membrane_signal = (
        right_v
        - left_v
    )


    if spike_total > 0:

        steering = (
            0.80
            * spike_signal
            +
            0.20
            * membrane_signal
        )

    else:

        steering = (
            membrane_signal
        )


    return {
        "gain":
            epg_delta_gain,

        "heading_column":
            heading_column,

        "goal_column":
            goal_column,

        "Delta7_spikes":
            delta7_spikes,

        "Delta7_spiking_neurons":
            delta7_spiking_neurons,

        "Delta7_mean_v":
            delta7_mean_v,

        "Delta7_max_v":
            delta7_max_v,

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

        "steering":
            float(steering),
    }


# =========================================================
# Controlled sensitivity sweep
#
# We start at the exact current model = 1x.
#
# Then increase only the EPG -> Delta7 surrogate gain.
# =========================================================

GAINS = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
]


TEST_HEADINGS = [
    "L1",
    "L5",
    "R5",
    "R1",
]


TEST_GOALS = [
    1,
    5,
    9,
]


rows = []


print()
print("=" * 100)
print("RUNNING GAIN SWEEP")
print("=" * 100)
print()


for gain in GAINS:

    print()
    print(
        f"EPG -> Delta7 gain = "
        f"{gain:.1f}x"
    )


    for heading in TEST_HEADINGS:

        for goal in TEST_GOALS:

            print(
                f"  heading={heading:>2} | "
                f"goal=C{goal}"
            )


            result = run_condition(
                heading_column=heading,
                goal_column=goal,
                epg_delta_gain=gain,
            )


            rows.append(
                result
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
# Gain summary
# =========================================================

summary_rows = []


for gain in GAINS:

    subset = results[
        results["gain"]
        == gain
    ].copy()


    delta_active_conditions = int(
        (
            subset[
                "Delta7_spikes"
            ] > 0
        ).sum()
    )


    dna03_active_conditions = int(
        (
            subset[
                "DNa03_spikes"
            ] > 0
        ).sum()
    )


    dna02_active_conditions = int(
        (
            subset[
                "DNa02_spikes"
            ] > 0
        ).sum()
    )


    # -----------------------------------------------------
    # Heading sensitivity
    #
    # For each fixed goal column, measure how much the
    # DNa steering output changes when heading changes.
    #
    # If heading pathway is silent, this stays small.
    # -----------------------------------------------------

    heading_ranges = []


    for goal in TEST_GOALS:

        goal_subset = subset[
            subset[
                "goal_column"
            ] == goal
        ]


        heading_range = float(
            goal_subset[
                "steering"
            ].max()
            -
            goal_subset[
                "steering"
            ].min()
        )


        heading_ranges.append(
            heading_range
        )


    mean_heading_range = float(
        np.mean(
            heading_ranges
        )
    )


    summary_rows.append(
        {
            "gain":
                gain,

            "Delta7_total_spikes":
                int(
                    subset[
                        "Delta7_spikes"
                    ].sum()
                ),

            "Delta7_active_conditions":
                delta_active_conditions,

            "Delta7_mean_spiking_neurons":
                float(
                    subset[
                        "Delta7_spiking_neurons"
                    ].mean()
                ),

            "Delta7_mean_v":
                float(
                    subset[
                        "Delta7_mean_v"
                    ].mean()
                ),

            "Delta7_max_v_seen":
                float(
                    subset[
                        "Delta7_max_v"
                    ].max()
                ),

            "DNa03_active_conditions":
                dna03_active_conditions,

            "DNa02_active_conditions":
                dna02_active_conditions,

            "mean_heading_steering_range":
                mean_heading_range,

            "mean_abs_steering":
                float(
                    subset[
                        "steering"
                    ]
                    .abs()
                    .mean()
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
# Print summary
# =========================================================

print()
print("=" * 150)
print("GAIN SWEEP SUMMARY")
print("=" * 150)
print()


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}",
    )
)


# =========================================================
# Detailed Delta7 activation
# =========================================================

print()
print("=" * 130)
print("DELTA7 ACTIVATION BY CONDITION")
print("=" * 130)
print()


print(
    results[
        [
            "gain",
            "heading_column",
            "goal_column",

            "Delta7_spikes",
            "Delta7_spiking_neurons",

            "Delta7_mean_v",
            "Delta7_max_v",

            "steering",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.5f}",
    )
)


# =========================================================
# Downstream activity
# =========================================================

print()
print("=" * 130)
print("DOWNSTREAM ACTIVITY")
print("=" * 130)
print()


print(
    results[
        [
            "gain",
            "heading_column",
            "goal_column",

            "PFL2_spikes",
            "PFL3_spikes",

            "DNa03_spikes",
            "DNa02_spikes",

            "DNa02_L_v",
            "DNa02_R_v",

            "steering",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.5f}",
    )
)


# =========================================================
# Files
# =========================================================

print()
print("=" * 100)
print("FILES CREATED")
print("=" * 100)

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)