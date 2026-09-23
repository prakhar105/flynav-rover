from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

NEURON_FILE = (
    OUTPUT_DIR
    / "steering_core_neurons.csv"
)

EDGE_FILE = (
    OUTPUT_DIR
    / "steering_core_edges.csv"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "pfl_dna_pathway_edges.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "pfl_dna_pathway_summary.csv"
)


# =========================================================
# Load
# =========================================================

neurons = pd.read_csv(
    NEURON_FILE
)

edges = pd.read_csv(
    EDGE_FILE
)


neurons["bodyId"] = (
    neurons["bodyId"]
    .astype("int64")
)


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

metadata = (
    neurons[
        [
            "bodyId",
            "type",
            "instance",
            "somaSide",
        ]
    ]
    .drop_duplicates(
        subset=["bodyId"]
    )
    .copy()
)


source_metadata = (
    metadata.rename(
        columns={
            "bodyId":
                "source_bodyId",

            "type":
                "source_type_meta",

            "instance":
                "source_instance_meta",

            "somaSide":
                "source_side_meta",
        }
    )
)


target_metadata = (
    metadata.rename(
        columns={
            "bodyId":
                "target_bodyId",

            "type":
                "target_type_meta",

            "instance":
                "target_instance_meta",

            "somaSide":
                "target_side_meta",
        }
    )
)


edges = (
    edges
    .merge(
        source_metadata,
        on="source_bodyId",
        how="left",
    )
    .merge(
        target_metadata,
        on="target_bodyId",
        how="left",
    )
)


# =========================================================
# Use metadata-derived values
#
# This avoids depending on whether the edge CSV itself
# already contains source_type / target_type / side fields.
# =========================================================

edges["source_type_final"] = (
    edges[
        "source_type_meta"
    ]
)

edges["target_type_final"] = (
    edges[
        "target_type_meta"
    ]
)

edges["source_instance_final"] = (
    edges[
        "source_instance_meta"
    ]
)

edges["target_instance_final"] = (
    edges[
        "target_instance_meta"
    ]
)

edges["source_side_final"] = (
    edges[
        "source_side_meta"
    ]
)

edges["target_side_final"] = (
    edges[
        "target_side_meta"
    ]
)


# =========================================================
# Select steering output pathways
#
# 1. PFL2 / PFL3 -> DNa02 / DNa03
# 2. DNa03 -> DNa02
# =========================================================

pfl_to_dna = edges[
    edges[
        "source_type_final"
    ].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
    &
    edges[
        "target_type_final"
    ].isin(
        [
            "DNa02",
            "DNa03",
        ]
    )
].copy()


dna03_to_dna02 = edges[
    (
        edges[
            "source_type_final"
        ]
        == "DNa03"
    )
    &
    (
        edges[
            "target_type_final"
        ]
        == "DNa02"
    )
].copy()


pathway_edges = pd.concat(
    [
        pfl_to_dna,
        dna03_to_dna02,
    ],
    ignore_index=True,
)


print("=" * 110)
print("PFL / DNa MOTOR PATHWAY ANALYSIS")
print("=" * 110)

print()

print(
    f"PFL -> DNa edges: "
    f"{len(pfl_to_dna)}"
)

print(
    f"DNa03 -> DNa02 edges: "
    f"{len(dna03_to_dna02)}"
)


# =========================================================
# Type -> type summary
# =========================================================

summary = (
    pathway_edges
    .groupby(
        [
            "source_type_final",
            "target_type_final",
        ]
    )
    .agg(
        total_weight=(
            "weight",
            "sum",
        ),

        edge_count=(
            "weight",
            "size",
        ),

        mean_weight=(
            "weight",
            "mean",
        ),

        max_weight=(
            "weight",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        "total_weight",
        ascending=False,
    )
)


print()
print("=" * 110)
print("TYPE -> TYPE PATHWAY SUMMARY")
print("=" * 110)
print()

print(
    summary.to_string(
        index=False
    )
)


# =========================================================
# Direct DNa02 inputs
# =========================================================

direct_dna02 = pfl_to_dna[
    pfl_to_dna[
        "target_type_final"
    ]
    == "DNa02"
].copy()


direct_summary = (
    direct_dna02
    .groupby(
        [
            "source_type_final",
            "source_side_final",
            "target_side_final",
        ]
    )
    .agg(
        total_weight=(
            "weight",
            "sum",
        ),

        edge_count=(
            "weight",
            "size",
        ),

        max_weight=(
            "weight",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        "total_weight",
        ascending=False,
    )
)


print()
print("=" * 110)
print("DIRECT PFL -> DNa02")
print("=" * 110)
print()

if len(direct_summary) > 0:

    print(
        direct_summary.to_string(
            index=False
        )
    )

else:

    print(
        "No direct PFL -> DNa02 "
        "connections found."
    )


# =========================================================
# DNa03 input
# =========================================================

to_dna03 = pfl_to_dna[
    pfl_to_dna[
        "target_type_final"
    ]
    == "DNa03"
].copy()


dna03_input_summary = (
    to_dna03
    .groupby(
        [
            "source_type_final",
            "source_side_final",
            "target_side_final",
        ]
    )
    .agg(
        total_weight=(
            "weight",
            "sum",
        ),

        edge_count=(
            "weight",
            "size",
        ),

        max_weight=(
            "weight",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "source_type_final",
            "total_weight",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


print()
print("=" * 110)
print("PFL -> DNa03 INDIRECT PATHWAY")
print("=" * 110)
print()

if len(dna03_input_summary) > 0:

    print(
        dna03_input_summary.to_string(
            index=False
        )
    )

else:

    print(
        "No PFL -> DNa03 "
        "connections found."
    )


# =========================================================
# DNa03 -> DNa02
# =========================================================

dna_chain_summary = (
    dna03_to_dna02
    .groupby(
        [
            "source_side_final",
            "target_side_final",
        ]
    )
    .agg(
        total_weight=(
            "weight",
            "sum",
        ),

        edge_count=(
            "weight",
            "size",
        ),

        max_weight=(
            "weight",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        "total_weight",
        ascending=False,
    )
)


print()
print("=" * 110)
print("DNa03 -> DNa02")
print("=" * 110)
print()

print(
    dna_chain_summary.to_string(
        index=False
    )
)


# =========================================================
# Pathway totals
# =========================================================

def total_weight(
    df,
):

    return float(
        df[
            "weight"
        ].sum()
    )


pfl2_to_dna02 = pfl_to_dna[
    (
        pfl_to_dna[
            "source_type_final"
        ]
        == "PFL2"
    )
    &
    (
        pfl_to_dna[
            "target_type_final"
        ]
        == "DNa02"
    )
]


pfl2_to_dna03 = pfl_to_dna[
    (
        pfl_to_dna[
            "source_type_final"
        ]
        == "PFL2"
    )
    &
    (
        pfl_to_dna[
            "target_type_final"
        ]
        == "DNa03"
    )
]


pfl3_to_dna02 = pfl_to_dna[
    (
        pfl_to_dna[
            "source_type_final"
        ]
        == "PFL3"
    )
    &
    (
        pfl_to_dna[
            "target_type_final"
        ]
        == "DNa02"
    )
]


pfl3_to_dna03 = pfl_to_dna[
    (
        pfl_to_dna[
            "source_type_final"
        ]
        == "PFL3"
    )
    &
    (
        pfl_to_dna[
            "target_type_final"
        ]
        == "DNa03"
    )
]


print()
print("=" * 110)
print("FUNCTIONAL PATHWAY INVENTORY")
print("=" * 110)
print()

print(
    f"PFL3 -> DNa02 DIRECT steering weight: "
    f"{total_weight(pfl3_to_dna02):.0f}"
)

print(
    f"PFL3 -> DNa03 INDIRECT steering weight: "
    f"{total_weight(pfl3_to_dna03):.0f}"
)

print(
    f"PFL2 -> DNa03 gain-path weight: "
    f"{total_weight(pfl2_to_dna03):.0f}"
)

print(
    f"PFL2 -> DNa02 direct weight: "
    f"{total_weight(pfl2_to_dna02):.0f}"
)

print(
    f"DNa03 -> DNa02 weight: "
    f"{total_weight(dna03_to_dna02):.0f}"
)


# =========================================================
# PFL3 -> DNa02 side matrix
# =========================================================

pfl3_direct = pfl3_to_dna02.copy()


if len(pfl3_direct) > 0:

    side_matrix = (
        pfl3_direct
        .pivot_table(
            index="source_side_final",
            columns="target_side_final",
            values="weight",
            aggfunc="sum",
            fill_value=0,
        )
    )


    print()
    print("=" * 110)
    print("PFL3 -> DNa02 SIDE MATRIX")
    print("=" * 110)
    print()

    print(
        side_matrix.to_string()
    )


# =========================================================
# Save
# =========================================================

pathway_edges[
    [
        "source_bodyId",
        "source_type_final",
        "source_instance_final",
        "source_side_final",

        "target_bodyId",
        "target_type_final",
        "target_instance_final",
        "target_side_final",

        "weight",
    ]
].to_csv(
    DETAIL_FILE,
    index=False,
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print()
print("=" * 110)
print("FILES CREATED")
print("=" * 110)

print(
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)