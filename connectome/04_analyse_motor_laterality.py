from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

NEURON_FILE = OUTPUT_DIR / "steering_core_neurons.csv"
EDGE_FILE = OUTPUT_DIR / "steering_core_edges.csv"

DIRECT_OUTPUT = (
    OUTPUT_DIR
    / "motor_direct_edges_with_sides.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "motor_laterality_summary.csv"
)

PREFERENCE_OUTPUT = (
    OUTPUT_DIR
    / "motor_side_preference.csv"
)


# =========================================================
# Load data
# =========================================================

neurons = pd.read_csv(NEURON_FILE)
edges = pd.read_csv(EDGE_FILE)

neurons["bodyId"] = neurons["bodyId"].astype("int64")

edges["source_bodyId"] = (
    edges["source_bodyId"]
    .astype("int64")
)

edges["target_bodyId"] = (
    edges["target_bodyId"]
    .astype("int64")
)

edges["weight"] = pd.to_numeric(
    edges["weight"],
    errors="coerce",
).fillna(0.0)


# =========================================================
# Build neuron metadata lookup
# =========================================================

meta = (
    neurons
    .set_index("bodyId")
    [
        [
            "type",
            "instance",
            "somaSide",
        ]
    ]
)


def get_side(body_id):
    if body_id not in meta.index:
        return None

    side = meta.loc[body_id, "somaSide"]

    if pd.isna(side):
        return None

    return str(side)


# =========================================================
# Attach biological side annotations
# =========================================================

edges["source_side"] = (
    edges["source_bodyId"]
    .apply(get_side)
)

edges["target_side"] = (
    edges["target_bodyId"]
    .apply(get_side)
)


# =========================================================
# PFL -> descending motor pathways
# =========================================================

motor_edges = edges[
    edges["source_type"].isin(
        ["PFL2", "PFL3"]
    )
    &
    edges["target_type"].isin(
        ["DNa02", "DNa03"]
    )
].copy()


motor_edges.to_csv(
    DIRECT_OUTPUT,
    index=False,
)


print()
print("=" * 100)
print("DIRECT PFL -> DESCENDING CONNECTIONS")
print("=" * 100)
print()

print(
    motor_edges[
        [
            "source_type",
            "source_instance",
            "source_side",
            "target_type",
            "target_instance",
            "target_side",
            "weight",
        ]
    ]
    .sort_values(
        "weight",
        ascending=False,
    )
    .head(80)
    .to_string(index=False)
)


# =========================================================
# Aggregate by biological hemisphere
# =========================================================

summary = (
    motor_edges
    .groupby(
        [
            "source_type",
            "source_side",
            "target_type",
            "target_side",
        ],
        dropna=False,
    )
    .agg(
        total_weight=(
            "weight",
            "sum",
        ),
        connection_count=(
            "weight",
            "size",
        ),
        max_weight=(
            "weight",
            "max",
        ),
        mean_weight=(
            "weight",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "total_weight",
        ascending=False,
    )
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print()
print("=" * 100)
print("HEMISPHERE -> MOTOR OUTPUT SUMMARY")
print("=" * 100)
print()

print(
    summary.to_string(index=False)
)


# =========================================================
# Ipsilateral vs contralateral analysis
# =========================================================

laterality = motor_edges[
    motor_edges["source_side"].isin(
        ["L", "R"]
    )
    &
    motor_edges["target_side"].isin(
        ["L", "R"]
    )
].copy()


laterality["relationship"] = laterality.apply(
    lambda row: (
        "ipsilateral"
        if row["source_side"]
        == row["target_side"]
        else "contralateral"
    ),
    axis=1,
)


laterality_summary = (
    laterality
    .groupby(
        [
            "source_type",
            "target_type",
            "relationship",
        ]
    )
    .agg(
        total_weight=(
            "weight",
            "sum",
        ),
        connection_count=(
            "weight",
            "size",
        ),
    )
    .reset_index()
)


print()
print("=" * 100)
print("IPSILATERAL VS CONTRALATERAL")
print("=" * 100)
print()

print(
    laterality_summary
    .to_string(index=False)
)


# =========================================================
# Left vs right target preference
#
# This is particularly important for turning the biological
# outputs into left/right rover steering signals.
# =========================================================

preferences = (
    motor_edges[
        motor_edges["target_side"].isin(
            ["L", "R"]
        )
    ]
    .groupby(
        [
            "source_type",
            "source_side",
            "target_type",
            "target_side",
        ]
    )
    ["weight"]
    .sum()
    .reset_index()
)


pivot = preferences.pivot_table(
    index=[
        "source_type",
        "source_side",
        "target_type",
    ],
    columns="target_side",
    values="weight",
    fill_value=0.0,
).reset_index()


if "L" not in pivot.columns:
    pivot["L"] = 0.0

if "R" not in pivot.columns:
    pivot["R"] = 0.0


pivot["total"] = (
    pivot["L"]
    + pivot["R"]
)


pivot["left_fraction"] = (
    pivot["L"]
    / pivot["total"]
)


pivot["right_fraction"] = (
    pivot["R"]
    / pivot["total"]
)


pivot["preferred_output"] = pivot.apply(
    lambda row: (
        "LEFT"
        if row["L"] > row["R"]
        else (
            "RIGHT"
            if row["R"] > row["L"]
            else "BALANCED"
        )
    ),
    axis=1,
)


pivot.to_csv(
    PREFERENCE_OUTPUT,
    index=False,
)


print()
print("=" * 100)
print("BIOLOGICAL MOTOR-SIDE PREFERENCE")
print("=" * 100)
print()

print(
    pivot[
        [
            "source_type",
            "source_side",
            "target_type",
            "L",
            "R",
            "left_fraction",
            "right_fraction",
            "preferred_output",
        ]
    ]
    .to_string(
        index=False,
    )
)


# =========================================================
# DNa03 -> DNa02
#
# Important because your previous extraction showed
# exceptionally strong DNa03 -> DNa02 connections.
# =========================================================

dna_edges = edges[
    (edges["source_type"] == "DNa03")
    &
    (edges["target_type"] == "DNa02")
].copy()


print()
print("=" * 100)
print("DNa03 -> DNa02 CONNECTIONS")
print("=" * 100)
print()

print(
    dna_edges[
        [
            "source_instance",
            "source_side",
            "target_instance",
            "target_side",
            "weight",
        ]
    ]
    .sort_values(
        "weight",
        ascending=False,
    )
    .to_string(index=False)
)


print()
print("=" * 100)
print("FILES CREATED")
print("=" * 100)

print(DIRECT_OUTPUT)
print(SUMMARY_OUTPUT)
print(PREFERENCE_OUTPUT)