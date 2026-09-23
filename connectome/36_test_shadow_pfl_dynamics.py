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
    OUTPUT_DIR / "steering_core_neurons.csv"
)

STEERING_EDGE_FILE = (
    OUTPUT_DIR / "steering_core_edges.csv"
)

DELTA7_NEURON_FILE = (
    OUTPUT_DIR / "delta7_candidates.csv"
)

DELTA7_EDGE_FILE = (
    OUTPUT_DIR / "delta7_bridge_edges.csv"
)

STEP31_RESULT_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment.csv"
)

CONDITION_FILE = (
    OUTPUT_DIR
    / "shadow_pfl_condition_metrics.csv"
)

TRANSITION_FILE = (
    OUTPUT_DIR
    / "shadow_pfl_transition_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "shadow_pfl_signal_summary.csv"
)


# =========================================================
# Frozen parameters
#
# DO NOT tune anything here.
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
        "Step 31 result file not found:\n"
        f"{STEP31_RESULT_FILE}"
    )


step31 = pd.read_csv(
    STEP31_RESULT_FILE
)


required_columns = [
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
    for column in required_columns
    if column not in step31.columns
]


if missing:

    raise RuntimeError(
        "Missing Step 31 columns: "
        f"{missing}"
    )


if len(step31) != 396:

    raise RuntimeError(
        f"Expected 396 Step 31 conditions, "
        f"got {len(step31)}."
    )


# =========================================================
# Load connectome-derived tables
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
# Frozen network neurons
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


if len(neurons) != 220:

    raise RuntimeError(
        f"Expected 220 neurons, "
        f"got {len(neurons)}."
    )


body_ids = set(
    neurons["bodyId"]
)


# =========================================================
# Combine / deduplicate edges
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


if len(edges) != 3231:

    raise RuntimeError(
        f"Expected 3231 edges, "
        f"got {len(edges)}."
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
# Helpers
# =========================================================

def normalize_side(value):

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


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


    neuron_types = set(
        neuron_types
    )


    output = []


    for index, row in neurons.iterrows():

        if row["type"] not in neuron_types:

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
        for piece
        in text.split(",")
        if piece.strip()
    ]


# =========================================================
# EPG parser / groups
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
    number = int(match.group(2))


    if not 1 <= number <= 8:
        return None


    return f"{side}{number}"


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


# =========================================================
# Populations
# =========================================================

pfl2_indices = get_indices(
    "PFL2"
)

pfl3_indices = get_indices(
    "PFL3"
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


dna02_left = get_indices(
    "DNa02",
    "L",
)

dna02_right = get_indices(
    "DNa02",
    "R",
)


# =========================================================
# Shadow PFL mapping
#
# One passive shadow neuron for every real PFL neuron.
# =========================================================

pfl_network_indices = sorted(
    [
        int(index)
        for index
        in np.concatenate(
            [
                pfl2_indices,
                pfl3_indices,
            ]
        )
    ]
)


N_SHADOW_PFL = len(
    pfl_network_indices
)


if N_SHADOW_PFL != 36:

    raise RuntimeError(
        f"Expected 36 total PFL neurons, "
        f"got {N_SHADOW_PFL}."
    )


real_pfl_to_shadow = {
    real_index: shadow_index
    for shadow_index, real_index
    in enumerate(
        pfl_network_indices
    )
}


shadow_pfl2_positions = np.asarray(
    [
        real_pfl_to_shadow[
            int(index)
        ]
        for index
        in pfl2_indices
    ],
    dtype=int,
)


shadow_pfl3_positions = np.asarray(
    [
        real_pfl_to_shadow[
            int(index)
        ]
        for index
        in pfl3_indices
    ],
    dtype=int,
)


def shadow_positions(
    real_indices,
):

    return np.asarray(
        [
            real_pfl_to_shadow[
                int(index)
            ]
            for index
            in real_indices
        ],
        dtype=int,
    )


shadow_pfl2_left = shadow_positions(
    pfl2_left
)

shadow_pfl2_right = shadow_positions(
    pfl2_right
)

shadow_pfl3_left = shadow_positions(
    pfl3_left
)

shadow_pfl3_right = shadow_positions(
    pfl3_right
)


# =========================================================
# Functional pathways
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


print("=" * 120)
print(
    "STEP 36 - SHADOW PFL DYNAMICS"
)
print("=" * 120)

print()

print(
    f"Real neurons: {len(neurons)}"
)

print(
    f"Functional edges: {len(edges)}"
)

print(
    f"Real PFL neurons: "
    f"{N_SHADOW_PFL}"
)

print(
    f"  PFL2: {len(pfl2_indices)}"
)

print(
    f"  PFL3: {len(pfl3_indices)}"
)

print(
    f"Passive shadow PFL: "
    f"{N_SHADOW_PFL}"
)

print(
    f"Conditions: {len(step31)}"
)


# =========================================================
# Prepare edges
# =========================================================

def prepare_edges(dataframe):

    sources = []
    targets = []
    raw = []


    for _, row in dataframe.iterrows():

        source_id = int(
            row["source_bodyId"]
        )

        target_id = int(
            row["target_bodyId"]
        )


        if (
            source_id not in body_to_index
            or
            target_id not in body_to_index
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

        raw.append(
            float(
                row["weight"]
            )
        )


    return (
        sources,
        targets,
        np.asarray(
            raw,
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
# Frozen synapse scaling
# =========================================================

all_raw = [
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
            in all_raw
            if len(array) > 0
        ]
    ).max()
)


def scale_weights(raw):

    return (
        0.02
        +
        0.30
        *
        np.sqrt(
            raw / MAX_RAW
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
# Shadow target mappings
# =========================================================

def map_pfl_targets_to_shadow(
    targets,
):

    result = []


    for real_target in targets:

        real_target = int(
            real_target
        )


        if (
            real_target
            not in real_pfl_to_shadow
        ):

            raise RuntimeError(
                "Attempted to map non-PFL "
                f"target {real_target}"
            )


        result.append(
            real_pfl_to_shadow[
                real_target
            ]
        )


    return result


shadow_other_targets = (
    map_pfl_targets_to_shadow(
        other_target
    )
)


shadow_delta_targets = (
    map_pfl_targets_to_shadow(
        delta_pfl_target
    )
)


# =========================================================
# Synapse constructors
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
        weights * gain
    )


    return synapses


def create_shadow_positive(
    real_group,
    shadow_group,
    source,
    target,
    weights,
):

    synapses = Synapses(
        real_group,
        shadow_group,
        model="w : 1",
        on_pre="""
        v_post += w
        exc_input_post += w
        """,
    )


    synapses.connect(
        i=source,
        j=target,
    )


    synapses.w = weights


    return synapses


def create_shadow_inhibitory(
    real_group,
    shadow_group,
    source,
    target,
    weights,
):

    synapses = Synapses(
        real_group,
        shadow_group,
        model="w : 1",
        on_pre="""
        v_post -= w
        inh_input_post += w
        """,
    )


    synapses.connect(
        i=source,
        j=target,
    )


    synapses.w = weights


    return synapses


# =========================================================
# Vector lateral difference
# =========================================================

def lateral_difference(
    vector,
    left_positions,
    right_positions,
):

    if (
        len(left_positions) == 0
        or
        len(right_positions) == 0
    ):

        return np.nan


    left = float(
        np.mean(
            vector[
                left_positions
            ]
        )
    )


    right = float(
        np.mean(
            vector[
                right_positions
            ]
        )
    )


    return right - left


# =========================================================
# Run one exact Step 31 condition
# =========================================================

def run_condition(
    heading,
    active_fc2_body_ids,
):

    # =====================================================
    # Real frozen circuit
    # =====================================================

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
    group.drive = BASELINE_DRIVE


    # =====================================================
    # Passive shadow PFL
    # =====================================================

    shadow_equations = """
    dv/dt = (-v + drive) / (20*ms) : 1
    drive : 1
    exc_input : 1
    inh_input : 1
    """


    shadow = NeuronGroup(
        N_SHADOW_PFL,
        shadow_equations,
        method="euler",
    )


    shadow.v = 0.0
    shadow.drive = BASELINE_DRIVE
    shadow.exc_input = 0.0
    shadow.inh_input = 0.0


    # =====================================================
    # EPG
    # =====================================================

    for index in epg_groups[
        heading
    ]:

        group.drive[
            index
        ] = EPG_DRIVE


    # =====================================================
    # Exact Step 31 FC2 subset
    # =====================================================

    for body_id in sorted(
        set(
            active_fc2_body_ids
        )
    ):

        network_index = body_to_index[
            body_id
        ]


        if neurons.loc[
            network_index,
            "type",
        ] not in [
            "FC2A",
            "FC2B",
            "FC2C",
        ]:

            raise RuntimeError(
                f"{body_id} is not FC2"
            )


        group.drive[
            network_index
        ] = FC2_DRIVE


    objects = [
        group,
        shadow,
    ]


    # =====================================================
    # Real EPG/FC2 -> PFL
    # =====================================================

    syn = create_positive_synapses(
        group,
        other_source,
        other_target,
        other_weights,
    )

    objects.append(
        syn
    )


    # =====================================================
    # Real EPG -> Delta7
    # =====================================================

    syn = create_positive_synapses(
        group,
        epg_delta_source,
        epg_delta_target,
        epg_delta_weights,
        gain=EPG_TO_DELTA7_GAIN,
    )

    objects.append(
        syn
    )


    # =====================================================
    # Real Delta7 -> PFL inhibitory
    # =====================================================

    delta_syn = Synapses(
        group,
        group,
        model="w : 1",
        on_pre="v_post -= w",
    )


    delta_syn.connect(
        i=delta_pfl_source,
        j=delta_pfl_target,
    )


    delta_syn.w = (
        delta_pfl_weights
    )


    objects.append(
        delta_syn
    )


    # =====================================================
    # Real PFL -> DNa
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


        objects.append(
            syn
        )


    # =====================================================
    # Real DNa03 -> DNa02
    # =====================================================

    syn = create_positive_synapses(
        group,
        dna03_dna02_source,
        dna03_dna02_target,
        dna03_dna02_weights,
    )


    objects.append(
        syn
    )


    # =====================================================
    # Shadow receives identical EPG + FC2 excitatory events
    # =====================================================

    shadow_exc = create_shadow_positive(
        group,
        shadow,
        other_source,
        shadow_other_targets,
        other_weights,
    )


    objects.append(
        shadow_exc
    )


    # =====================================================
    # Shadow receives identical Delta7 inhibition
    # =====================================================

    shadow_inh = create_shadow_inhibitory(
        group,
        shadow,
        delta_pfl_source,
        shadow_delta_targets,
        delta_pfl_weights,
    )


    objects.append(
        shadow_inh
    )


    # =====================================================
    # Monitors
    # =====================================================

    spike_monitor = SpikeMonitor(
        group
    )


    real_pfl_state = StateMonitor(
        group,
        "v",
        record=pfl_network_indices,
        dt=STATE_MONITOR_DT,
    )


    shadow_state = StateMonitor(
        shadow,
        "v",
        record=True,
        dt=STATE_MONITOR_DT,
    )


    objects.extend(
        [
            spike_monitor,
            real_pfl_state,
            shadow_state,
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


    counts = np.asarray(
        spike_monitor.count,
        dtype=float,
    )


    final_v = np.asarray(
        group.v,
        dtype=float,
    )


    duration_seconds = float(
        SIMULATION_TIME
        /
        (1000 * ms)
    )


    # =====================================================
    # Existing steering reproduction
    # =====================================================

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
        final_v[
            dna02_left
        ].mean()
    )


    right_final_v = float(
        final_v[
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


    steering = float(
        np.tanh(
            (
                right_activity
                -
                left_activity
            )
            /
            (
                2.0
                *
                ACTIVITY_SCALE
            )
        )
    )


    # =====================================================
    # Real PFL vectors
    # =====================================================

    real_pfl2_rate = (
        counts[
            pfl2_indices
        ]
        /
        duration_seconds
    )


    real_pfl3_rate = (
        counts[
            pfl3_indices
        ]
        /
        duration_seconds
    )


    real_trace = np.asarray(
        real_pfl_state.v,
        dtype=float,
    )


    real_mean_v_all = np.mean(
        real_trace,
        axis=1,
    )


    real_position = {
        network_index: row
        for row, network_index
        in enumerate(
            pfl_network_indices
        )
    }


    real_pfl2_rows = [
        real_position[
            int(index)
        ]
        for index
        in pfl2_indices
    ]


    real_pfl3_rows = [
        real_position[
            int(index)
        ]
        for index
        in pfl3_indices
    ]


    real_pfl2_mean_v = (
        real_mean_v_all[
            real_pfl2_rows
        ]
    )


    real_pfl3_mean_v = (
        real_mean_v_all[
            real_pfl3_rows
        ]
    )


    # =====================================================
    # Shadow vectors
    # =====================================================

    shadow_trace = np.asarray(
        shadow_state.v,
        dtype=float,
    )


    shadow_mean_v_all = np.mean(
        shadow_trace,
        axis=1,
    )


    shadow_pfl2_mean_v = (
        shadow_mean_v_all[
            shadow_pfl2_positions
        ]
    )


    shadow_pfl3_mean_v = (
        shadow_mean_v_all[
            shadow_pfl3_positions
        ]
    )


    shadow_exc_input = np.asarray(
        shadow.exc_input,
        dtype=float,
    )


    shadow_inh_input = np.asarray(
        shadow.inh_input,
        dtype=float,
    )


    shadow_net_input = (
        shadow_exc_input
        -
        shadow_inh_input
    )


    shadow_pfl2_net_input = (
        shadow_net_input[
            shadow_pfl2_positions
        ]
    )


    shadow_pfl3_net_input = (
        shadow_net_input[
            shadow_pfl3_positions
        ]
    )


    # =====================================================
    # Lateral differences
    # =====================================================

    pfl2_real_positions = {
        int(index): position
        for position, index
        in enumerate(
            pfl2_indices
        )
    }


    pfl3_real_positions = {
        int(index): position
        for position, index
        in enumerate(
            pfl3_indices
        )
    }


    real_pfl2_left_pos = np.asarray(
        [
            pfl2_real_positions[
                int(index)
            ]
            for index
            in pfl2_left
        ],
        dtype=int,
    )


    real_pfl2_right_pos = np.asarray(
        [
            pfl2_real_positions[
                int(index)
            ]
            for index
            in pfl2_right
        ],
        dtype=int,
    )


    real_pfl3_left_pos = np.asarray(
        [
            pfl3_real_positions[
                int(index)
            ]
            for index
            in pfl3_left
        ],
        dtype=int,
    )


    real_pfl3_right_pos = np.asarray(
        [
            pfl3_real_positions[
                int(index)
            ]
            for index
            in pfl3_right
        ],
        dtype=int,
    )


    return {

        "steering":
            steering,

        "real_PFL2_rate_diff":
            lateral_difference(
                real_pfl2_rate,
                real_pfl2_left_pos,
                real_pfl2_right_pos,
            ),

        "real_PFL3_rate_diff":
            lateral_difference(
                real_pfl3_rate,
                real_pfl3_left_pos,
                real_pfl3_right_pos,
            ),

        "shadow_PFL2_mean_v_diff":
            lateral_difference(
                shadow_mean_v_all,
                shadow_pfl2_left,
                shadow_pfl2_right,
            ),

        "shadow_PFL3_mean_v_diff":
            lateral_difference(
                shadow_mean_v_all,
                shadow_pfl3_left,
                shadow_pfl3_right,
            ),

        "shadow_PFL2_input_diff":
            lateral_difference(
                shadow_net_input,
                shadow_pfl2_left,
                shadow_pfl2_right,
            ),

        "shadow_PFL3_input_diff":
            lateral_difference(
                shadow_net_input,
                shadow_pfl3_left,
                shadow_pfl3_right,
            ),

            "vectors": {

            "real_PFL2_rate":
                real_pfl2_rate,

            "real_PFL3_rate":
                real_pfl3_rate,

            "real_PFL2_meanV":
                real_pfl2_mean_v,

            "real_PFL3_meanV":
                real_pfl3_mean_v,

            "shadow_PFL2_meanV":
                shadow_pfl2_mean_v,

            "shadow_PFL3_meanV":
                shadow_pfl3_mean_v,

            "shadow_PFL2_input":
                shadow_pfl2_net_input,

            "shadow_PFL3_input":
                shadow_pfl3_net_input,
        },
    }


# =========================================================
# Run exact Step 31 conditions
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


for index, source_row in step31.iterrows():

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
        f"[{index + 1:3d}/396] "
        f"heading={heading:>2} | "
        f"C{column_a}->C{column_b} | "
        f"alpha={alpha:.1f}"
    )


    result = run_condition(
        heading,
        active_body_ids,
    )


    vectors = result.pop(
        "vectors"
    )


    reproduced = float(
        result[
            "steering"
        ]
    )


    expected = float(
        source_row[
            "steering"
        ]
    )


    condition_rows.append(
        {

            "condition_id":
                len(
                    condition_rows
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
                expected,

            "reproduction_error":
                abs(
                    reproduced
                    -
                    expected
                ),

            **result,
        }
    )


    feature_records.append(
        vectors
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
    f"{mean_error:.12f}"
)

print(
    f"Max steering error: "
    f"{max_error:.12f}"
)


if max_error <= 1e-8:

    print(
        "PASS: passive PFL shadows do not "
        "alter the frozen circuit."
    )


# =========================================================
# Vector scales
# =========================================================

FEATURE_NAMES = [
    "real_PFL2_rate",
    "real_PFL3_rate",
    "real_PFL2_meanV",
    "real_PFL3_meanV",
    "shadow_PFL2_meanV",
    "shadow_PFL3_meanV",
    "shadow_PFL2_input",
    "shadow_PFL3_input",
]


scales = {}


for feature in FEATURE_NAMES:

    matrix = np.stack(
        [
            record[
                feature
            ]
            for record
            in feature_records
        ]
    )


    norms = np.linalg.norm(
        matrix,
        axis=1,
    )


    scale = float(
        np.percentile(
            norms,
            95.0,
        )
    )


    if scale < 1e-12:
        scale = 1.0


    scales[
        feature
    ] = scale


# =========================================================
# Transition analysis
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
        len(group) - 1
    ):

        first = group.iloc[
            index
        ]

        second = group.iloc[
            index + 1
        ]


        first_id = int(
            first[
                "condition_id"
            ]
        )


        second_id = int(
            second[
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

            "steering_abs_step":
                abs(
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
                ),
        }


        for feature in FEATURE_NAMES:

            first_vector = (
                feature_records[
                    first_id
                ][
                    feature
                ]
            )


            second_vector = (
                feature_records[
                    second_id
                ][
                    feature
                ]
            )


            raw_step = float(
                np.linalg.norm(
                    second_vector
                    -
                    first_vector
                )
            )


            record[
                f"{feature}_norm_step"
            ] = (
                raw_step
                /
                scales[
                    feature
                ]
            )


        # ---------------------------------------------
        # lateral scalar changes
        # ---------------------------------------------

        for column in [
            "real_PFL2_rate_diff",
            "real_PFL3_rate_diff",
            "shadow_PFL2_mean_v_diff",
            "shadow_PFL3_mean_v_diff",
            "shadow_PFL2_input_diff",
            "shadow_PFL3_input_diff",
        ]:

            record[
                f"{column}_step"
            ] = abs(
                float(
                    second[
                        column
                    ]
                )
                -
                float(
                    first[
                        column
                    ]
                )
            )


        record[
            "is_wrap"
        ] = (
            int(
                column_a
            )
            ==
            9
            and
            int(
                column_b
            )
            ==
            1
        )


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


primary = transitions[
    ~transitions[
        "is_wrap"
    ]
].copy()


wrap = transitions[
    transitions[
        "is_wrap"
    ]
].copy()


if len(primary) != 320:

    raise RuntimeError(
        f"Expected 320 primary transitions, "
        f"got {len(primary)}."
    )


# =========================================================
# Correlation helper
# =========================================================

def correlation(
    dataframe,
    first_column,
    second_column,
):

    x = dataframe[
        first_column
    ].to_numpy(
        dtype=float,
    )

    y = dataframe[
        second_column
    ].to_numpy(
        dtype=float,
    )


    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    x = x[mask]
    y = y[mask]


    if (
        len(x) < 3
        or
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


# =========================================================
# Summary
# =========================================================

summary_rows = []


for feature in FEATURE_NAMES:

    column = (
        f"{feature}_norm_step"
    )


    values = primary[
        column
    ].to_numpy(
        dtype=float,
    )


    summary_rows.append(
        {

            "signal":
                feature,

            "mean_norm_step":
                float(
                    values.mean()
                ),

            "median_norm_step":
                float(
                    np.median(
                        values
                    )
                ),

            "p95_norm_step":
                float(
                    np.percentile(
                        values,
                        95.0,
                    )
                ),

            "max_norm_step":
                float(
                    values.max()
                ),

            "steps_gt_0_25":
                int(
                    (
                        values > 0.25
                    ).sum()
                ),

            "steps_gt_0_50":
                int(
                    (
                        values > 0.50
                    ).sum()
                ),

            "corr_with_steering_jump":
                correlation(
                    primary,
                    column,
                    "steering_abs_step",
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
# Output
# =========================================================

print()

print("=" * 170)
print(
    "REAL VS SHADOW PFL SUMMARY - PRIMARY ADJACENCIES"
)
print("=" * 170)

print()


print(
    summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# PFL3 threshold amplification diagnostic
# =========================================================

print()

print("=" * 150)
print(
    "PFL3 THRESHOLD / RESET AMPLIFICATION TEST"
)
print("=" * 150)

print()


print(
    "Shadow PFL3 net-input -> "
    "real PFL3 rate correlation: "
    f"{correlation(
        primary,
        'shadow_PFL3_input_norm_step',
        'real_PFL3_rate_norm_step',
    ):+.4f}"
)


print(
    "Shadow PFL3 net-input -> "
    "steering correlation: "
    f"{correlation(
        primary,
        'shadow_PFL3_input_norm_step',
        'steering_abs_step',
    ):+.4f}"
)


print(
    "Shadow PFL3 mean-V -> "
    "steering correlation: "
    f"{correlation(
        primary,
        'shadow_PFL3_meanV_norm_step',
        'steering_abs_step',
    ):+.4f}"
)


print(
    "Real PFL3 spike-rate -> "
    "steering correlation: "
    f"{correlation(
        primary,
        'real_PFL3_rate_norm_step',
        'steering_abs_step',
    ):+.4f}"
)


# =========================================================
# Threshold-amplification candidates
#
# Small continuous shadow-input change but large real
# spike-rate change.
# =========================================================

amplification_candidates = primary[
    (
        primary[
            "shadow_PFL3_input_norm_step"
        ]
        <
        0.15
    )
    &
    (
        primary[
            "real_PFL3_rate_norm_step"
        ]
        >
        0.40
    )
].copy()


print()

print(
    "Transitions with shadow PFL3 input < 0.15 "
    "but real PFL3 rate > 0.40:"
)

print(
    f"{len(amplification_candidates)}/"
    f"{len(primary)}"
)


# =========================================================
# Current worst steering transition
# =========================================================

worst = (
    primary
    .sort_values(
        "steering_abs_step",
        ascending=False,
    )
    .iloc[
        0
    ]
)


print()

print("=" * 160)
print(
    "CURRENT PRIMARY WORST STEERING TRANSITION"
)
print("=" * 160)

print()


print(
    f"Heading: "
    f"{worst['heading_column']}"
)

print(
    f"Pair: "
    f"C{int(worst['column_a'])}"
    f"->C{int(worst['column_b'])}"
)

print(
    f"Alpha: "
    f"{worst['alpha_from']:.1f}"
    f" -> "
    f"{worst['alpha_to']:.1f}"
)

print(
    f"Steering jump: "
    f"{worst['steering_abs_step']:.4f}"
)

print()

print(
    f"Shadow PFL2 input step: "
    f"{worst['shadow_PFL2_input_norm_step']:.4f}"
)

print(
    f"Shadow PFL3 input step: "
    f"{worst['shadow_PFL3_input_norm_step']:.4f}"
)

print()

print(
    f"Shadow PFL2 mean-V step: "
    f"{worst['shadow_PFL2_meanV_norm_step']:.4f}"
)

print(
    f"Shadow PFL3 mean-V step: "
    f"{worst['shadow_PFL3_meanV_norm_step']:.4f}"
)

print()

print(
    f"Real PFL2 rate step: "
    f"{worst['real_PFL2_rate_norm_step']:.4f}"
)

print(
    f"Real PFL3 rate step: "
    f"{worst['real_PFL3_rate_norm_step']:.4f}"
)


# =========================================================
# Raw trajectory around worst pair
# =========================================================

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

print("=" * 200)
print(
    "FULL WORST TRAJECTORY - REAL VS SHADOW PFL"
)
print("=" * 200)

print()


print(
    worst_conditions[
        [
            "alpha",
            "real_PFL2_rate_diff",
            "shadow_PFL2_mean_v_diff",
            "shadow_PFL2_input_diff",
            "real_PFL3_rate_diff",
            "shadow_PFL3_mean_v_diff",
            "shadow_PFL3_input_diff",
            "steering",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.6f}",
    )
)


# =========================================================
# Strongest PFL3 amplification cases
# =========================================================

if len(
    amplification_candidates
) > 0:

    print()

    print("=" * 180)
    print(
        "STRONG PFL3 THRESHOLD-AMPLIFICATION CANDIDATES"
    )
    print("=" * 180)

    print()


    print(
        amplification_candidates
        .sort_values(
            "real_PFL3_rate_norm_step",
            ascending=False,
        )
        .head(
            20
        )[
            [
                "heading_column",
                "column_a",
                "column_b",
                "alpha_from",
                "alpha_to",
                "shadow_PFL3_input_norm_step",
                "shadow_PFL3_meanV_norm_step",
                "real_PFL3_rate_norm_step",
                "steering_abs_step",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


# =========================================================
# Wrap separately
# =========================================================

print()

print("=" * 120)
print(
    "EXPLORATORY C9 -> C1 WRAP"
)
print("=" * 120)

print()

print(
    f"Transitions: "
    f"{len(wrap)}"
)

print(
    f"Worst steering jump: "
    f"{wrap['steering_abs_step'].max():.4f}"
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