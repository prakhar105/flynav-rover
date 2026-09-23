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
    / "motor_pathway_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "motor_pathway_comparison_summary.csv"
)


# =========================================================
# Working surrogate calibration
#
# These are simulation calibration values,
# NOT biological constants.
# =========================================================

EPG_TO_DELTA7_GAIN = 3.0

# Keep the previously identified downstream gain
# temporarily fixed at 3x.
#
# In this experiment we are isolating pathways,
# not tuning them.
PFL_PATHWAY_GAIN = 3.0


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
# Combine neurons
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
# Functional circuit
# =========================================================

functional = (

    # EPG -> Delta7
    (
        (edges["source_type"] == "EPG")
        &
        (edges["target_type"] == "Delta7")
    )

    |

    # direct EPG -> PFL
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


print("=" * 110)
print("MOTOR PATHWAY ISOLATION EXPERIMENT")
print("=" * 110)

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
    f"PFL pathway test gain: "
    f"{PFL_PATHWAY_GAIN:.1f}x"
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
# Validate grouping
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
# Split biological pathways
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


# ---------------------------------------------------------
# Direct steering branch
#
# PFL3 -> DNa02
# ---------------------------------------------------------

pfl3_dna02_edges = edges[
    (
        (edges["source_type"] == "PFL3")
        &
        (edges["target_type"] == "DNa02")
    )
].copy()


# ---------------------------------------------------------
# PFL2 -> DNa03
#
# Main PFL2 branch.
# ---------------------------------------------------------

pfl2_dna03_edges = edges[
    (
        (edges["source_type"] == "PFL2")
        &
        (edges["target_type"] == "DNa03")
    )
].copy()


# ---------------------------------------------------------
# PFL3 -> DNa03
#
# Indirect PFL3 steering branch.
# ---------------------------------------------------------

pfl3_dna03_edges = edges[
    (
        (edges["source_type"] == "PFL3")
        &
        (edges["target_type"] == "DNa03")
    )
].copy()


# ---------------------------------------------------------
# DNa03 -> DNa02 relay
# ---------------------------------------------------------

dna03_dna02_edges = edges[
    (
        (edges["source_type"] == "DNa03")
        &
        (edges["target_type"] == "DNa02")
    )
].copy()


# ---------------------------------------------------------
# IMPORTANT:
#
# PFL2 -> DNa02 is deliberately NOT included.
#
# Its total raw anatomical weight is only 5,
# compared with PFL2 -> DNa03 = 1571.
# ---------------------------------------------------------


# =========================================================
# Upstream/default positive edges
#
# Remove ALL downstream branches listed above.
# =========================================================

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

        |

        (
            (edges["source_type"] == "PFL3")
            &
            (edges["target_type"] == "DNa02")
        )

        |

        (
            (edges["source_type"] == "PFL2")
            &
            (edges["target_type"] == "DNa03")
        )

        |

        (
            (edges["source_type"] == "PFL3")
            &
            (edges["target_type"] == "DNa03")
        )

        |

        (
            (edges["source_type"] == "DNa03")
            &
            (edges["target_type"] == "DNa02")
        )

        |

        # Remove tiny PFL2 -> DNa02 branch
        (
            (edges["source_type"] == "PFL2")
            &
            (edges["target_type"] == "DNa02")
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
    f"PFL3 -> DNa02 direct edges: "
    f"{len(pfl3_dna02_edges)}"
)

print(
    f"PFL2 -> DNa03 edges: "
    f"{len(pfl2_dna03_edges)}"
)

print(
    f"PFL3 -> DNa03 edges: "
    f"{len(pfl3_dna03_edges)}"
)

print(
    f"DNa03 -> DNa02 edges: "
    f"{len(dna03_dna02_edges)}"
)


# =========================================================
# Convert biological edges to Brian2 arrays
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
    pfl3_dna02_source,
    pfl3_dna02_target,
    pfl3_dna02_raw,
) = prepare_edges(
    pfl3_dna02_edges
)


(
    pfl2_dna03_source,
    pfl2_dna03_target,
    pfl2_dna03_raw,
) = prepare_edges(
    pfl2_dna03_edges
)


(
    pfl3_dna03_source,
    pfl3_dna03_target,
    pfl3_dna03_raw,
) = prepare_edges(
    pfl3_dna03_edges
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
# Common anatomical scaling
# =========================================================

all_raw = np.concatenate(
    [
        epg_delta_raw,
        delta_pfl_raw,
        pfl3_dna02_raw,
        pfl2_dna03_raw,
        pfl3_dna03_raw,
        dna03_dna02_raw,
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

pfl3_dna02_weights = scale_weights(
    pfl3_dna02_raw
)

pfl2_dna03_weights = scale_weights(
    pfl2_dna03_raw
)

pfl3_dna03_weights = scale_weights(
    pfl3_dna03_raw
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
# Add synapse population utility
# =========================================================

def add_positive_synapses(
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
# Pathway configurations
#
# All PFL-originating active branches use exactly 3x
# for this diagnostic.
#
# We are testing architecture, not gain.
# =========================================================

CONFIGURATIONS = {

    # -----------------------------------------------------
    # Direct PFL3 steering only
    # -----------------------------------------------------

    "DIRECT_PFL3": {
        "direct_pfl3": True,
        "pfl2_gain": False,
        "pfl3_indirect": False,
    },


    # -----------------------------------------------------
    # PFL2 -> DNa03 -> DNa02 only
    # -----------------------------------------------------

    "PFL2_GAIN_ONLY": {
        "direct_pfl3": False,
        "pfl2_gain": True,
        "pfl3_indirect": False,
    },


    # -----------------------------------------------------
    # PFL3 -> DNa03 -> DNa02 only
    # -----------------------------------------------------

    "PFL3_INDIRECT": {
        "direct_pfl3": False,
        "pfl2_gain": False,
        "pfl3_indirect": True,
    },


    # -----------------------------------------------------
    # CORE MOTOR CIRCUIT
    #
    # Direct PFL3 directional pathway
    # +
    # PFL2 gain pathway
    #
    # NO PFL3 -> DNa03 indirect pathway.
    #
    # Comparing this against FULL_SEPARATED tells us
    # exactly whether the smaller PFL3 indirect branch
    # contributes meaningfully.
    # -----------------------------------------------------

    "CORE_NO_PFL3_INDIRECT": {
        "direct_pfl3": True,
        "pfl2_gain": True,
        "pfl3_indirect": False,
    },


    # -----------------------------------------------------
    # Full separated anatomical motor architecture
    # -----------------------------------------------------

    "FULL_SEPARATED": {
        "direct_pfl3": True,
        "pfl2_gain": True,
        "pfl3_indirect": True,
    },
}


# =========================================================
# Simulation
# =========================================================

def run_condition(
    configuration_name,
    heading_column,
    goal_column,
):

    configuration = (
        CONFIGURATIONS[
            configuration_name
        ]
    )


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
    # Inputs
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


    network_objects = [
        G
    ]


    # -----------------------------------------------------
    # Default positive upstream connectivity
    # -----------------------------------------------------

    S_other = add_positive_synapses(
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


    # -----------------------------------------------------
    # EPG -> Delta7
    # -----------------------------------------------------

    S_epg_delta = add_positive_synapses(
        G,
        epg_delta_source,
        epg_delta_target,
        epg_delta_weights,
        gain=EPG_TO_DELTA7_GAIN,
    )


    network_objects.append(
        S_epg_delta
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


    network_objects.append(
        S_delta_pfl
    )


    # -----------------------------------------------------
    # DIRECT PFL3 -> DNa02
    # -----------------------------------------------------

    if configuration[
        "direct_pfl3"
    ]:

        synapses = add_positive_synapses(
            G,
            pfl3_dna02_source,
            pfl3_dna02_target,
            pfl3_dna02_weights,
            gain=PFL_PATHWAY_GAIN,
        )


        network_objects.append(
            synapses
        )


    # -----------------------------------------------------
    # PFL2 -> DNa03
    # -----------------------------------------------------

    if configuration[
        "pfl2_gain"
    ]:

        synapses = add_positive_synapses(
            G,
            pfl2_dna03_source,
            pfl2_dna03_target,
            pfl2_dna03_weights,
            gain=PFL_PATHWAY_GAIN,
        )


        network_objects.append(
            synapses
        )


    # -----------------------------------------------------
    # PFL3 -> DNa03
    # -----------------------------------------------------

    if configuration[
        "pfl3_indirect"
    ]:

        synapses = add_positive_synapses(
            G,
            pfl3_dna03_source,
            pfl3_dna03_target,
            pfl3_dna03_weights,
            gain=PFL_PATHWAY_GAIN,
        )


        network_objects.append(
            synapses
        )


    # -----------------------------------------------------
    # DNa03 -> DNa02
    #
    # Include whenever either indirect route is active.
    #
    # Keep anatomical relay at base 1x.
    # -----------------------------------------------------

    if (
        configuration[
            "pfl2_gain"
        ]
        or
        configuration[
            "pfl3_indirect"
        ]
    ):

        synapses = add_positive_synapses(
            G,
            dna03_dna02_source,
            dna03_dna02_target,
            dna03_dna02_weights,
            gain=1.0,
        )


        network_objects.append(
            synapses
        )


    # -----------------------------------------------------
    # Monitor
    # -----------------------------------------------------

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


    # =====================================================
    # DNa03
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
    # DNa02
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


    membrane_difference = (
        dna02_r_v
        -
        dna02_l_v
    )


    spike_total = (
        dna02_l_spikes
        +
        dna02_r_spikes
    )


    if spike_total > 0:

        spike_difference = (
            dna02_r_spikes
            -
            dna02_l_spikes
        ) / spike_total

    else:

        spike_difference = 0.0


    # -----------------------------------------------------
    # Keep old decoder ONLY for comparison.
    #
    # This is NOT being frozen yet.
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
        "configuration":
            configuration_name,

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

        "DNa03_L_spikes":
            dna03_l_spikes,

        "DNa03_R_spikes":
            dna03_r_spikes,

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

        "spike_difference":
            spike_difference,

        "steering":
            float(
                steering
            ),
    }


# =========================================================
# Representative conditions
#
# Same conditions used in earlier diagnostics so the
# results remain directly comparable.
# =========================================================

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
print("=" * 110)
print("RUNNING PATHWAY ISOLATION")
print("=" * 110)


for configuration in (
    CONFIGURATIONS.keys()
):

    print()

    print(
        f"CONFIGURATION: "
        f"{configuration}"
    )


    for heading in TEST_HEADINGS:

        for goal in TEST_GOALS:

            print(
                f"  heading={heading:>2} | "
                f"goal=C{goal}"
            )


            result = run_condition(
                configuration_name=configuration,
                heading_column=heading,
                goal_column=goal,
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
# Configuration summary
# =========================================================

summary_rows = []


for configuration in (
    CONFIGURATIONS.keys()
):

    subset = results[
        results[
            "configuration"
        ]
        == configuration
    ].copy()


    dna03_total = (
        subset[
            "DNa03_L_spikes"
        ]
        +
        subset[
            "DNa03_R_spikes"
        ]
    )


    dna02_total = (
        subset[
            "DNa02_L_spikes"
        ]
        +
        subset[
            "DNa02_R_spikes"
        ]
    )


    # -----------------------------------------------------
    # Heading sensitivity
    # -----------------------------------------------------

    heading_ranges = []


    for goal in TEST_GOALS:

        goal_subset = subset[
            subset[
                "goal_column"
            ]
            == goal
        ]


        heading_ranges.append(
            float(
                goal_subset[
                    "steering"
                ].max()
                -
                goal_subset[
                    "steering"
                ].min()
            )
        )


    # -----------------------------------------------------
    # Output diversity
    # -----------------------------------------------------

    RIGHT_THRESHOLD = 0.05
    LEFT_THRESHOLD = -0.05


    right_conditions = int(
        (
            subset[
                "steering"
            ]
            > RIGHT_THRESHOLD
        ).sum()
    )


    left_conditions = int(
        (
            subset[
                "steering"
            ]
            < LEFT_THRESHOLD
        ).sum()
    )


    neutral_conditions = int(
        len(subset)
        -
        right_conditions
        -
        left_conditions
    )


    saturated_conditions = int(
        (
            subset[
                "steering"
            ]
            .abs()
            > 0.75
        ).sum()
    )


    summary_rows.append(
        {
            "configuration":
                configuration,

            "DNa03_total_spikes":
                int(
                    dna03_total.sum()
                ),

            "DNa03_active_conditions":
                int(
                    (
                        dna03_total > 0
                    ).sum()
                ),

            "DNa02_total_spikes":
                int(
                    dna02_total.sum()
                ),

            "DNa02_active_conditions":
                int(
                    (
                        dna02_total > 0
                    ).sum()
                ),

            "right_conditions":
                right_conditions,

            "left_conditions":
                left_conditions,

            "neutral_conditions":
                neutral_conditions,

            "saturated_conditions":
                saturated_conditions,

            "mean_heading_range":
                float(
                    np.mean(
                        heading_ranges
                    )
                ),

            "mean_abs_membrane_difference":
                float(
                    subset[
                        "membrane_difference"
                    ]
                    .abs()
                    .mean()
                ),

            "mean_abs_spike_difference":
                float(
                    subset[
                        "spike_difference"
                    ]
                    .abs()
                    .mean()
                ),

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
print("=" * 170)
print("PATHWAY COMPARISON SUMMARY")
print("=" * 170)
print()


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}",
    )
)


# =========================================================
# Detailed outputs
# =========================================================

print()
print("=" * 160)
print("DNa OUTPUT BY PATHWAY")
print("=" * 160)
print()


print(
    results[
        [
            "configuration",
            "heading_column",
            "goal_column",

            "PFL2_spikes",
            "PFL3_spikes",

            "DNa03_L_spikes",
            "DNa03_R_spikes",

            "DNa02_L_spikes",
            "DNa02_R_spikes",

            "membrane_difference",
            "spike_difference",
            "steering",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.5f}",
    )
)


# =========================================================
# Full separated map for representative conditions
# =========================================================

full = results[
    results[
        "configuration"
    ]
    == "FULL_SEPARATED"
].copy()


print()
print("=" * 140)
print("FULL SEPARATED PATHWAY OUTPUT")
print("=" * 140)
print()


print(
    full[
        [
            "heading_column",
            "goal_column",

            "DNa03_L_spikes",
            "DNa03_R_spikes",

            "DNa02_L_spikes",
            "DNa02_R_spikes",

            "membrane_difference",
            "spike_difference",
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
print("=" * 110)
print("FILES CREATED")
print("=" * 110)

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)