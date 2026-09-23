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
# Step 37: PFL3-only threshold sweep
#
# Purpose
# -------
# Test whether the discontinuity localized in Step 36 can be
# reduced by changing ONLY the PFL3 LIF threshold.
#
# Frozen:
#   - 220-neuron / 3231-edge topology
#   - anatomical raw weights + existing scaling
#   - EPG -> Delta7 gain = 3.0
#   - PFL -> DNa gain = 3.0
#   - Delta7 -> PFL inhibitory model hypothesis
#   - FC2 recruited subsets from Step 31
#   - all non-PFL3 thresholds = 1.0
#   - tau = 20 ms, reset = 0, refractory = 3 ms
#   - DNa steering decoder
#
# Varied:
#   - PFL3 threshold only: 1.0, 0.9, 0.8, 0.7, 0.6
#
# Primary analysis only:
#   C1->C2, ..., C8->C9
#   C9->C1 is intentionally excluded because wrap adjacency
#   was never established.
# ============================================================

prefs.codegen.target = "numpy"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

STEERING_NEURON_FILE = OUTPUT_DIR / "steering_core_neurons.csv"
STEERING_EDGE_FILE = OUTPUT_DIR / "steering_core_edges.csv"
DELTA7_NEURON_FILE = OUTPUT_DIR / "delta7_candidates.csv"
DELTA7_EDGE_FILE = OUTPUT_DIR / "delta7_bridge_edges.csv"
STEP31_RESULT_FILE = OUTPUT_DIR / "fc2_anatomically_balanced_recruitment.csv"

CONDITION_FILE = OUTPUT_DIR / "pfl3_threshold_sweep_conditions.csv"
TRAJECTORY_FILE = OUTPUT_DIR / "pfl3_threshold_sweep_trajectories.csv"
SUMMARY_FILE = OUTPUT_DIR / "pfl3_threshold_sweep_summary.csv"
ENDPOINT_FILE = OUTPUT_DIR / "pfl3_threshold_sweep_endpoint_metrics.csv"

# -----------------------------
# Frozen surrogate calibration
# -----------------------------
EPG_TO_DELTA7_GAIN = 3.0
PFL_TO_DNA_GAIN = 3.0

SPIKE_SCALE = 5.0
ACTIVITY_SCALE = 1.10599

BASELINE_DRIVE = 0.03
EPG_DRIVE = 1.35
FC2_DRIVE = 1.35

SIMULATION_TIME = 300 * ms

PFL3_THRESHOLDS = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
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

SIGN_THRESHOLD = 0.05


# ============================================================
# Load Step 31 exact conditions
# ============================================================

if not STEP31_RESULT_FILE.exists():
    raise FileNotFoundError(
        f"Step 31 result file not found:\n{STEP31_RESULT_FILE}"
    )

step31_all = pd.read_csv(STEP31_RESULT_FILE)

required_step31 = [
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
    for column in required_step31
    if column not in step31_all.columns
]

if missing:
    raise RuntimeError(
        f"Step 31 result file is missing columns: {missing}"
    )

if len(step31_all) != 396:
    raise RuntimeError(
        f"Expected 396 Step 31 conditions, got {len(step31_all)}."
    )

# Primary established adjacency only.
primary_mask = step31_all.apply(
    lambda row: (int(row["column_a"]), int(row["column_b"])) in PRIMARY_PAIRS,
    axis=1,
)

step31 = (
    step31_all.loc[primary_mask]
    .copy()
    .reset_index(drop=True)
)

if len(step31) != 352:
    raise RuntimeError(
        f"Expected 352 primary Step 31 conditions, got {len(step31)}."
    )


# ============================================================
# Load connectome-derived tables
# ============================================================

steering_neurons = pd.read_csv(STEERING_NEURON_FILE)
steering_edges = pd.read_csv(STEERING_EDGE_FILE)
delta7_neurons = pd.read_csv(DELTA7_NEURON_FILE)
delta7_edges = pd.read_csv(DELTA7_EDGE_FILE)

for dataframe in [steering_neurons, delta7_neurons]:
    dataframe["bodyId"] = dataframe["bodyId"].astype("int64")

for dataframe in [steering_edges, delta7_edges]:
    dataframe["source_bodyId"] = dataframe["source_bodyId"].astype("int64")
    dataframe["target_bodyId"] = dataframe["target_bodyId"].astype("int64")
    dataframe["weight"] = pd.to_numeric(
        dataframe["weight"],
        errors="coerce",
    ).fillna(0.0)


# ============================================================
# Frozen 220-neuron population
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
    [steering_neurons, delta7_neurons],
    ignore_index=True,
)

neurons = neurons.drop_duplicates(subset=["bodyId"])

neurons = (
    neurons[
        neurons["type"].isin(NETWORK_TYPES)
    ]
    .copy()
    .reset_index(drop=True)
)

if len(neurons) != 220:
    raise RuntimeError(
        f"Expected 220 neurons, got {len(neurons)}."
    )

body_ids = set(neurons["bodyId"])

edges = pd.concat(
    [steering_edges, delta7_edges],
    ignore_index=True,
)

edges = edges[
    edges["source_bodyId"].isin(body_ids)
    &
    edges["target_bodyId"].isin(body_ids)
].copy()

# Same duplicate handling as frozen Step 27.
edges = (
    edges
    .sort_values("weight", ascending=False)
    .drop_duplicates(
        subset=["source_bodyId", "target_bodyId"],
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
            edges["target_type"].isin(["PFL2", "PFL3"])
        )
    )
    |
    (
        (edges["source_type"] == "Delta7")
        &
        edges["target_type"].isin(["PFL2", "PFL3"])
    )
    |
    (
        edges["source_type"].isin(["FC2A", "FC2B", "FC2C"])
        &
        edges["target_type"].isin(["PFL2", "PFL3"])
    )
    |
    (
        edges["source_type"].isin(["PFL2", "PFL3"])
        &
        edges["target_type"].isin(["DNa02", "DNa03"])
    )
    |
    (
        (edges["source_type"] == "DNa03")
        &
        (edges["target_type"] == "DNa02")
    )
)

edges = (
    edges.loc[functional_mask]
    .copy()
    .reset_index(drop=True)
)

if len(edges) != 3231:
    raise RuntimeError(
        f"Expected 3231 functional edges, got {len(edges)}."
    )

body_to_index = {
    int(body_id): index
    for index, body_id in enumerate(neurons["bodyId"])
}


# ============================================================
# Population helpers
# ============================================================

def normalize_side(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def get_indices(neuron_type, side=None):
    output = []

    for index, row in neurons.iterrows():
        if row["type"] != neuron_type:
            continue

        if side is not None:
            if normalize_side(row.get("somaSide", "")) != str(side).upper():
                continue

        output.append(index)

    return np.asarray(output, dtype=int)


pfl3_indices = get_indices("PFL3")
dna03_indices = get_indices("DNa03")
dna02_indices = get_indices("DNa02")

dna02_left = get_indices("DNa02", "L")
dna02_right = get_indices("DNa02", "R")

if len(pfl3_indices) != 24:
    raise RuntimeError(
        f"Expected 24 PFL3 neurons, got {len(pfl3_indices)}."
    )

if len(dna03_indices) != 2:
    raise RuntimeError(
        f"Expected 2 DNa03 neurons, got {len(dna03_indices)}."
    )

if len(dna02_indices) != 2:
    raise RuntimeError(
        f"Expected 2 DNa02 neurons, got {len(dna02_indices)}."
    )

if len(dna02_left) != 1 or len(dna02_right) != 1:
    raise RuntimeError(
        "Expected exactly one left and one right DNa02 neuron."
    )


# ============================================================
# EPG groups
# ============================================================

def parse_epg_column(instance):
    if not isinstance(instance, str):
        return None

    match = re.search(r"_([LR])(\d+)$", instance)
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

    column = parse_epg_column(row["instance"])
    if column is None:
        continue

    epg_groups.setdefault(column, []).append(index)

if sum(len(values) for values in epg_groups.values()) != 46:
    raise RuntimeError(
        "Expected 46 grouped EPG neurons."
    )


# ============================================================
# Frozen functional pathways
# ============================================================

mask_epg_delta = (
    (edges["source_type"] == "EPG")
    &
    (edges["target_type"] == "Delta7")
)

mask_delta_pfl = (
    (edges["source_type"] == "Delta7")
    &
    edges["target_type"].isin(["PFL2", "PFL3"])
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
        edges["target_type"].isin(["PFL2", "PFL3"])
    )
    |
    (
        edges["source_type"].isin(["FC2A", "FC2B", "FC2C"])
        &
        edges["target_type"].isin(["PFL2", "PFL3"])
    )
)

epg_delta_edges = edges.loc[mask_epg_delta].copy()
delta_pfl_edges = edges.loc[mask_delta_pfl].copy()
pfl3_dna02_edges = edges.loc[mask_pfl3_dna02].copy()
pfl3_dna03_edges = edges.loc[mask_pfl3_dna03].copy()
pfl2_dna03_edges = edges.loc[mask_pfl2_dna03].copy()
pfl2_dna02_edges = edges.loc[mask_pfl2_dna02].copy()
dna03_dna02_edges = edges.loc[mask_dna03_dna02].copy()
other_edges = edges.loc[mask_other_upstream].copy()

accounted = (
    len(epg_delta_edges)
    + len(delta_pfl_edges)
    + len(pfl3_dna02_edges)
    + len(pfl3_dna03_edges)
    + len(pfl2_dna03_edges)
    + len(pfl2_dna02_edges)
    + len(dna03_dna02_edges)
    + len(other_edges)
)

if accounted != len(edges):
    raise RuntimeError(
        f"Functional edge accounting failed: {accounted} != {len(edges)}"
    )


# ============================================================
# Convert edges to Brian2 arrays
# ============================================================

def prepare_edges(dataframe):
    sources = []
    targets = []
    raw_weights = []

    for _, row in dataframe.iterrows():
        source_id = int(row["source_bodyId"])
        target_id = int(row["target_bodyId"])

        if source_id not in body_to_index:
            continue

        if target_id not in body_to_index:
            continue

        sources.append(body_to_index[source_id])
        targets.append(body_to_index[target_id])
        raw_weights.append(float(row["weight"]))

    return (
        sources,
        targets,
        np.asarray(raw_weights, dtype=float),
    )


(
    epg_delta_source,
    epg_delta_target,
    epg_delta_raw,
) = prepare_edges(epg_delta_edges)

(
    delta_pfl_source,
    delta_pfl_target,
    delta_pfl_raw,
) = prepare_edges(delta_pfl_edges)

(
    pfl3_dna02_source,
    pfl3_dna02_target,
    pfl3_dna02_raw,
) = prepare_edges(pfl3_dna02_edges)

(
    pfl3_dna03_source,
    pfl3_dna03_target,
    pfl3_dna03_raw,
) = prepare_edges(pfl3_dna03_edges)

(
    pfl2_dna03_source,
    pfl2_dna03_target,
    pfl2_dna03_raw,
) = prepare_edges(pfl2_dna03_edges)

(
    pfl2_dna02_source,
    pfl2_dna02_target,
    pfl2_dna02_raw,
) = prepare_edges(pfl2_dna02_edges)

(
    dna03_dna02_source,
    dna03_dna02_target,
    dna03_dna02_raw,
) = prepare_edges(dna03_dna02_edges)

(
    other_source,
    other_target,
    other_raw,
) = prepare_edges(other_edges)


# ============================================================
# Frozen anatomical weight scaling
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
            for array in raw_arrays
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
        np.sqrt(raw_weights / MAX_RAW)
    )


epg_delta_weights = scale_weights(epg_delta_raw)
delta_pfl_weights = scale_weights(delta_pfl_raw)
pfl3_dna02_weights = scale_weights(pfl3_dna02_raw)
pfl3_dna03_weights = scale_weights(pfl3_dna03_raw)
pfl2_dna03_weights = scale_weights(pfl2_dna03_raw)
pfl2_dna02_weights = scale_weights(pfl2_dna02_raw)
dna03_dna02_weights = scale_weights(dna03_dna02_raw)
other_weights = scale_weights(other_raw)


# ============================================================
# General helpers
# ============================================================

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

    synapses.w = weights * gain

    return synapses


def steering_sign(value):
    if value > SIGN_THRESHOLD:
        return 1

    if value < -SIGN_THRESHOLD:
        return -1

    return 0


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ============================================================
# Run one exact Step 31 condition with ONLY PFL3 threshold
# changed.
# ============================================================

def run_condition(
    heading_column,
    active_fc2_body_ids,
    pfl3_threshold,
):
    equations = """
    dv/dt = (-v + drive) / (20*ms) : 1
    drive : 1
    threshold_value : 1
    """

    group = NeuronGroup(
        len(neurons),
        equations,
        threshold="v > threshold_value",
        reset="v = 0.0",
        refractory=3 * ms,
        method="euler",
    )

    group.v = 0.0
    group.drive = BASELINE_DRIVE

    # All neurons retain frozen threshold 1.0.
    group.threshold_value = 1.0

    # Only PFL3 changes.
    group.threshold_value[
        pfl3_indices
    ] = float(pfl3_threshold)

    # EPG heading.
    if heading_column not in epg_groups:
        raise RuntimeError(
            f"Unknown EPG heading: {heading_column}"
        )

    for index in epg_groups[heading_column]:
        group.drive[index] = EPG_DRIVE

    # Exact Step 31 FC2 recruited subset.
    for body_id in sorted(set(active_fc2_body_ids)):
        if body_id not in body_to_index:
            raise RuntimeError(
                f"Unknown Step 31 body ID: {body_id}"
            )

        network_index = body_to_index[body_id]

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
                f"Body ID {body_id} is {neuron_type}, not FC2."
            )

        group.drive[
            network_index
        ] = FC2_DRIVE

    objects = [group]

    # EPG + FC2 -> PFL.
    synapses = create_positive_synapses(
        group,
        other_source,
        other_target,
        other_weights,
        gain=1.0,
    )

    if synapses is not None:
        objects.append(synapses)

    # EPG -> Delta7.
    synapses = create_positive_synapses(
        group,
        epg_delta_source,
        epg_delta_target,
        epg_delta_weights,
        gain=EPG_TO_DELTA7_GAIN,
    )

    if synapses is not None:
        objects.append(synapses)

    # Delta7 -> PFL inhibitory model hypothesis.
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

        delta_synapses.w = delta_pfl_weights

        objects.append(delta_synapses)

    # Complete frozen PFL -> DNa topology.
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

    for source, target, weights in motor_routes:
        synapses = create_positive_synapses(
            group,
            source,
            target,
            weights,
            gain=PFL_TO_DNA_GAIN,
        )

        if synapses is not None:
            objects.append(synapses)

    # DNa03 -> DNa02.
    synapses = create_positive_synapses(
        group,
        dna03_dna02_source,
        dna03_dna02_target,
        dna03_dna02_weights,
        gain=1.0,
    )

    if synapses is not None:
        objects.append(synapses)

    spike_monitor = SpikeMonitor(group)
    objects.append(spike_monitor)

    network = Network(*objects)
    network.run(SIMULATION_TIME)

    counts = np.asarray(
        spike_monitor.count,
        dtype=float,
    )

    final_voltage = np.asarray(
        group.v,
        dtype=float,
    )

    duration_seconds = float(
        SIMULATION_TIME / (1000 * ms)
    )

    # PFL3 spike-rate vector.
    pfl3_rate_vector = (
        counts[pfl3_indices]
        /
        duration_seconds
    )

    # DNa activity.
    dna03_total_spikes = int(
        counts[dna03_indices].sum()
    )

    dna02_total_spikes = int(
        counts[dna02_indices].sum()
    )

    left_spikes = float(
        counts[dna02_left].sum()
    )

    right_spikes = float(
        counts[dna02_right].sum()
    )

    left_final_v = float(
        final_voltage[dna02_left].mean()
    )

    right_final_v = float(
        final_voltage[dna02_right].mean()
    )

    left_activity = (
        left_final_v
        +
        left_spikes / SPIKE_SCALE
    )

    right_activity = (
        right_final_v
        +
        right_spikes / SPIKE_SCALE
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
        "steering": steering,
        "activity_difference": activity_difference,
        "dna03_total_spikes": dna03_total_spikes,
        "dna02_total_spikes": dna02_total_spikes,
        "dna02_left_spikes": int(left_spikes),
        "dna02_right_spikes": int(right_spikes),
        "pfl3_total_spikes": int(
            counts[pfl3_indices].sum()
        ),
        "pfl3_rate_vector": pfl3_rate_vector,
    }


# ============================================================
# Run sweep
# ============================================================

print("=" * 120)
print("STEP 37 - PFL3-ONLY THRESHOLD SWEEP")
print("=" * 120)
print()
print(f"Neurons: {len(neurons)}")
print(f"Functional edges: {len(edges)}")
print(f"PFL3 neurons: {len(pfl3_indices)}")
print(f"Primary Step 31 conditions per threshold: {len(step31)}")
print(f"Thresholds: {PFL3_THRESHOLDS}")
print(
    f"Total simulations: "
    f"{len(step31) * len(PFL3_THRESHOLDS)}"
)
print()
print("Only the PFL3 threshold is varied.")
print("All other topology/dynamics remain frozen.")

condition_rows = []
pfl3_vectors = []

total_runs = (
    len(step31)
    *
    len(PFL3_THRESHOLDS)
)

run_counter = 0

for threshold in PFL3_THRESHOLDS:
    print()
    print("=" * 120)
    print(
        f"PFL3 THRESHOLD = {threshold:.2f}"
    )
    print("=" * 120)

    for _, source_row in step31.iterrows():
        run_counter += 1

        heading = str(
            source_row["heading_column"]
        )

        column_a = int(
            source_row["column_a"]
        )

        column_b = int(
            source_row["column_b"]
        )

        alpha = float(
            source_row["alpha"]
        )

        active_body_ids = (
            parse_body_ids(
                source_row["a_body_ids"]
            )
            +
            parse_body_ids(
                source_row["b_body_ids"]
            )
        )

        print(
            f"[{run_counter:4d}/{total_runs}] "
            f"thr={threshold:.2f} | "
            f"heading={heading:>2} | "
            f"C{column_a}->C{column_b} | "
            f"alpha={alpha:.1f}"
        )

        result = run_condition(
            heading_column=heading,
            active_fc2_body_ids=active_body_ids,
            pfl3_threshold=threshold,
        )

        pfl3_rate_vector = result.pop(
            "pfl3_rate_vector"
        )

        condition_id = len(
            condition_rows
        )

        condition_rows.append(
            {
                "condition_id": condition_id,
                "pfl3_threshold": threshold,
                "heading_column": heading,
                "column_a": column_a,
                "column_b": column_b,
                "alpha": alpha,
                "frozen_step31_steering": float(
                    source_row["steering"]
                ),
                **result,
            }
        )

        pfl3_vectors.append(
            pfl3_rate_vector
        )


conditions = pd.DataFrame(
    condition_rows
)

conditions.to_csv(
    CONDITION_FILE,
    index=False,
)


# ============================================================
# Baseline reproduction check
# ============================================================

baseline_conditions = conditions[
    np.isclose(
        conditions["pfl3_threshold"],
        1.0,
    )
].copy()

baseline_error = np.abs(
    baseline_conditions["steering"].to_numpy(dtype=float)
    -
    baseline_conditions["frozen_step31_steering"].to_numpy(dtype=float)
)

print()
print("=" * 120)
print("THRESHOLD 1.00 FROZEN-BASELINE REPRODUCTION CHECK")
print("=" * 120)
print()
print(
    f"Conditions checked: "
    f"{len(baseline_conditions)}"
)
print(
    f"Mean steering error: "
    f"{baseline_error.mean():.12f}"
)
print(
    f"Max steering error: "
    f"{baseline_error.max():.12f}"
)

if baseline_error.max() <= 1e-8:
    print(
        "PASS: threshold=1.00 exactly reproduces "
        "the primary frozen Step 31 conditions."
    )
else:
    print(
        "WARNING: threshold=1.00 does not reproduce "
        "the frozen baseline exactly."
    )


# ============================================================
# PFL3 vector characteristic scale per threshold
# ============================================================

vector_scales = {}

for threshold in PFL3_THRESHOLDS:
    threshold_rows = conditions[
        np.isclose(
            conditions["pfl3_threshold"],
            threshold,
        )
    ]

    ids = threshold_rows[
        "condition_id"
    ].astype(int).to_numpy()

    matrix = np.stack(
        [
            pfl3_vectors[index]
            for index in ids
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
        scale = float(
            norms.max()
        )

    if scale < 1e-12:
        scale = 1.0

    vector_scales[
        float(threshold)
    ] = scale


# ============================================================
# Trajectory / adjacent-alpha analysis
# ============================================================

trajectory_rows = []

for threshold in PFL3_THRESHOLDS:
    threshold_conditions = conditions[
        np.isclose(
            conditions["pfl3_threshold"],
            threshold,
        )
    ]

    for (
        heading,
        column_a,
        column_b,
    ), group in threshold_conditions.groupby(
        [
            "heading_column",
            "column_a",
            "column_b",
        ]
    ):
        group = (
            group
            .sort_values("alpha")
            .reset_index(drop=True)
        )

        alpha = group[
            "alpha"
        ].to_numpy(dtype=float)

        steering = group[
            "steering"
        ].to_numpy(dtype=float)

        steering_steps = np.abs(
            np.diff(steering)
        )

        pfl3_steps = []

        for index in range(
            len(group) - 1
        ):
            first_id = int(
                group.iloc[index]["condition_id"]
            )

            second_id = int(
                group.iloc[index + 1]["condition_id"]
            )

            raw_step = float(
                np.linalg.norm(
                    pfl3_vectors[second_id]
                    -
                    pfl3_vectors[first_id]
                )
            )

            pfl3_steps.append(
                raw_step
                /
                vector_scales[
                    float(threshold)
                ]
            )

        pfl3_steps = np.asarray(
            pfl3_steps,
            dtype=float,
        )

        signs = [
            steering_sign(value)
            for value in steering
        ]

        nonzero_signs = [
            sign
            for sign in signs
            if sign != 0
        ]

        sign_transitions = sum(
            1
            for index in range(
                1,
                len(nonzero_signs),
            )
            if nonzero_signs[index]
            != nonzero_signs[index - 1]
        )

        start = float(steering[0])
        end = float(steering[-1])

        total_variation = float(
            steering_steps.sum()
        )

        directness = (
            abs(end - start)
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
                float(steering.min()),
            ),
            max(
                0.0,
                float(steering.max())
                -
                endpoint_max,
            ),
        )

        trajectory_rows.append(
            {
                "pfl3_threshold": threshold,
                "heading_column": heading,
                "column_a": int(column_a),
                "column_b": int(column_b),
                "mean_abs_steering_step": float(
                    steering_steps.mean()
                ),
                "max_abs_steering_step": float(
                    steering_steps.max()
                ),
                "mean_pfl3_rate_norm_step": float(
                    pfl3_steps.mean()
                ),
                "max_pfl3_rate_norm_step": float(
                    pfl3_steps.max()
                ),
                "sign_transitions": int(
                    sign_transitions
                ),
                "directness_ratio": float(
                    directness
                ),
                "overshoot": float(
                    overshoot
                ),
            }
        )


trajectories = pd.DataFrame(
    trajectory_rows
)

trajectories.to_csv(
    TRAJECTORY_FILE,
    index=False,
)


# ============================================================
# Endpoint preservation metrics
# ============================================================

endpoint_rows = []

for threshold in PFL3_THRESHOLDS:
    data = conditions[
        np.isclose(
            conditions["pfl3_threshold"],
            threshold,
        )
        &
        (
            np.isclose(
                conditions["alpha"],
                0.0,
            )
            |
            np.isclose(
                conditions["alpha"],
                1.0,
            )
        )
    ].copy()

    actual = data[
        "steering"
    ].to_numpy(dtype=float)

    frozen = data[
        "frozen_step31_steering"
    ].to_numpy(dtype=float)

    mae = float(
        np.mean(
            np.abs(
                actual - frozen
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(
                actual - frozen
            )
        )
    )

    corr = safe_corr(
        frozen,
        actual,
    )

    frozen_sign = np.asarray(
        [
            steering_sign(value)
            for value in frozen
        ]
    )

    actual_sign = np.asarray(
        [
            steering_sign(value)
            for value in actual
        ]
    )

    sign_agreement = float(
        np.mean(
            frozen_sign
            ==
            actual_sign
        )
    )

    dna03_active_fraction = float(
        np.mean(
            data[
                "dna03_total_spikes"
            ].to_numpy(dtype=float)
            >
            0
        )
    )

    dna02_active_fraction = float(
        np.mean(
            data[
                "dna02_total_spikes"
            ].to_numpy(dtype=float)
            >
            0
        )
    )

    endpoint_rows.append(
        {
            "pfl3_threshold": threshold,
            "endpoint_conditions": len(data),
            "endpoint_mae_vs_frozen": mae,
            "endpoint_max_error_vs_frozen": max_error,
            "endpoint_corr_vs_frozen": corr,
            "endpoint_sign_agreement": sign_agreement,
            "endpoint_dna03_active_fraction": dna03_active_fraction,
            "endpoint_dna02_active_fraction": dna02_active_fraction,
        }
    )


endpoint_metrics = pd.DataFrame(
    endpoint_rows
)

endpoint_metrics.to_csv(
    ENDPOINT_FILE,
    index=False,
)


# ============================================================
# Threshold summary
# ============================================================

summary_rows = []

for threshold in PFL3_THRESHOLDS:
    trajectory_data = trajectories[
        np.isclose(
            trajectories["pfl3_threshold"],
            threshold,
        )
    ]

    condition_data = conditions[
        np.isclose(
            conditions["pfl3_threshold"],
            threshold,
        )
    ]

    endpoint_data = endpoint_metrics[
        np.isclose(
            endpoint_metrics["pfl3_threshold"],
            threshold,
        )
    ].iloc[0]

    summary_rows.append(
        {
            "pfl3_threshold": threshold,
            "conditions": len(condition_data),
            "trajectories": len(trajectory_data),
            "mean_adjacent_steering_step": float(
                trajectory_data[
                    "mean_abs_steering_step"
                ].mean()
            ),
            "mean_trajectory_max_step": float(
                trajectory_data[
                    "max_abs_steering_step"
                ].mean()
            ),
            "worst_steering_step": float(
                trajectory_data[
                    "max_abs_steering_step"
                ].max()
            ),
            "mean_pfl3_rate_norm_step": float(
                trajectory_data[
                    "mean_pfl3_rate_norm_step"
                ].mean()
            ),
            "worst_pfl3_rate_norm_step": float(
                trajectory_data[
                    "max_pfl3_rate_norm_step"
                ].max()
            ),
            "trajectories_with_sign_flip": int(
                (
                    trajectory_data[
                        "sign_transitions"
                    ]
                    >
                    0
                ).sum()
            ),
            "trajectories_with_multi_sign_flip": int(
                (
                    trajectory_data[
                        "sign_transitions"
                    ]
                    >
                    1
                ).sum()
            ),
            "trajectories_step_gt_0_20": int(
                (
                    trajectory_data[
                        "max_abs_steering_step"
                    ]
                    >
                    0.20
                ).sum()
            ),
            "mean_directness": float(
                trajectory_data[
                    "directness_ratio"
                ].mean()
            ),
            "max_overshoot": float(
                trajectory_data[
                    "overshoot"
                ].max()
            ),
            "all_condition_dna03_active_fraction": float(
                np.mean(
                    condition_data[
                        "dna03_total_spikes"
                    ].to_numpy(dtype=float)
                    >
                    0
                )
            ),
            "all_condition_dna02_active_fraction": float(
                np.mean(
                    condition_data[
                        "dna02_total_spikes"
                    ].to_numpy(dtype=float)
                    >
                    0
                )
            ),
            "endpoint_mae_vs_frozen": float(
                endpoint_data[
                    "endpoint_mae_vs_frozen"
                ]
            ),
            "endpoint_max_error_vs_frozen": float(
                endpoint_data[
                    "endpoint_max_error_vs_frozen"
                ]
            ),
            "endpoint_corr_vs_frozen": float(
                endpoint_data[
                    "endpoint_corr_vs_frozen"
                ]
            ),
            "endpoint_sign_agreement": float(
                endpoint_data[
                    "endpoint_sign_agreement"
                ]
            ),
            "endpoint_dna03_active_fraction": float(
                endpoint_data[
                    "endpoint_dna03_active_fraction"
                ]
            ),
            "endpoint_dna02_active_fraction": float(
                endpoint_data[
                    "endpoint_dna02_active_fraction"
                ]
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Engineering screening relative to baseline
#
# This is NOT automatic scientific truth. It is only a useful
# screen for deciding whether any lower hard threshold deserves
# follow-up validation.
# ============================================================

baseline = summary[
    np.isclose(
        summary["pfl3_threshold"],
        1.0,
    )
].iloc[0]

baseline_worst = float(
    baseline["worst_steering_step"]
)

baseline_mean_max = float(
    baseline["mean_trajectory_max_step"]
)

baseline_dna02_activity = float(
    baseline[
        "all_condition_dna02_active_fraction"
    ]
)

summary[
    "worst_step_reduction_pct"
] = (
    100.0
    *
    (
        baseline_worst
        -
        summary["worst_steering_step"]
    )
    /
    baseline_worst
)

summary[
    "mean_max_step_reduction_pct"
] = (
    100.0
    *
    (
        baseline_mean_max
        -
        summary["mean_trajectory_max_step"]
    )
    /
    baseline_mean_max
)

summary[
    "screen_substantial_smoothing"
] = (
    summary[
        "worst_steering_step"
    ]
    <=
    0.80 * baseline_worst
)

summary[
    "screen_endpoint_preserved"
] = (
    (
        summary[
            "endpoint_mae_vs_frozen"
        ]
        <=
        0.08
    )
    &
    (
        summary[
            "endpoint_corr_vs_frozen"
        ]
        >=
        0.90
    )
    &
    (
        summary[
            "endpoint_sign_agreement"
        ]
        >=
        0.90
    )
)

summary[
    "screen_dna02_coverage_preserved"
] = (
    summary[
        "all_condition_dna02_active_fraction"
    ]
    >=
    max(
        0.50,
        0.90
        *
        baseline_dna02_activity,
    )
)

summary[
    "screen_followup_candidate"
] = (
    summary[
        "screen_substantial_smoothing"
    ]
    &
    summary[
        "screen_endpoint_preserved"
    ]
    &
    summary[
        "screen_dna02_coverage_preserved"
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 180)
print("PFL3 THRESHOLD SWEEP SUMMARY")
print("=" * 180)
print()

display_columns = [
    "pfl3_threshold",
    "mean_adjacent_steering_step",
    "mean_trajectory_max_step",
    "worst_steering_step",
    "worst_step_reduction_pct",
    "mean_pfl3_rate_norm_step",
    "worst_pfl3_rate_norm_step",
    "trajectories_with_sign_flip",
    "trajectories_with_multi_sign_flip",
    "mean_directness",
    "all_condition_dna02_active_fraction",
    "endpoint_mae_vs_frozen",
    "endpoint_corr_vs_frozen",
    "endpoint_sign_agreement",
    "screen_followup_candidate",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print()
print("=" * 150)
print("ENDPOINT PRESERVATION")
print("=" * 150)
print()

print(
    endpoint_metrics.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# ============================================================
# Worst trajectory at every threshold
# ============================================================

print()
print("=" * 170)
print("WORST TRAJECTORY AT EACH PFL3 THRESHOLD")
print("=" * 170)
print()

worst_rows = []

for threshold in PFL3_THRESHOLDS:
    data = trajectories[
        np.isclose(
            trajectories["pfl3_threshold"],
            threshold,
        )
    ]

    worst = (
        data
        .sort_values(
            "max_abs_steering_step",
            ascending=False,
        )
        .iloc[0]
    )

    worst_rows.append(
        {
            "pfl3_threshold": threshold,
            "heading_column": worst["heading_column"],
            "column_a": int(worst["column_a"]),
            "column_b": int(worst["column_b"]),
            "max_abs_steering_step": float(
                worst["max_abs_steering_step"]
            ),
            "mean_abs_steering_step": float(
                worst["mean_abs_steering_step"]
            ),
            "max_pfl3_rate_norm_step": float(
                worst["max_pfl3_rate_norm_step"]
            ),
            "sign_transitions": int(
                worst["sign_transitions"]
            ),
            "directness_ratio": float(
                worst["directness_ratio"]
            ),
            "overshoot": float(
                worst["overshoot"]
            ),
        }
    )

worst_dataframe = pd.DataFrame(
    worst_rows
)

print(
    worst_dataframe.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# ============================================================
# Print current baseline vs lower thresholds
# ============================================================

print()
print("=" * 160)
print("CHANGE RELATIVE TO FROZEN PFL3 THRESHOLD = 1.00")
print("=" * 160)
print()

for _, row in summary.iterrows():
    print(
        f"threshold={row['pfl3_threshold']:.2f} "
        f"| worst step="
        f"{row['worst_steering_step']:.4f} "
        f"| worst reduction="
        f"{row['worst_step_reduction_pct']:+.1f}% "
        f"| endpoint MAE="
        f"{row['endpoint_mae_vs_frozen']:.4f} "
        f"| endpoint corr="
        f"{row['endpoint_corr_vs_frozen']:.4f} "
        f"| sign agreement="
        f"{row['endpoint_sign_agreement']:.3f} "
        f"| DNa02 coverage="
        f"{row['all_condition_dna02_active_fraction']:.3f} "
        f"| follow-up candidate="
        f"{bool(row['screen_followup_candidate'])}"
    )


print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)
print(CONDITION_FILE)
print(TRAJECTORY_FILE)
print(SUMMARY_FILE)
print(ENDPOINT_FILE)
