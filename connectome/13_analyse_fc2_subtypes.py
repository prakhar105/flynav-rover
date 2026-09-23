import re
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
    / "delta7_bridge_edges.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fc2_subtype_column_connectivity.csv"
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
# FC2 column parser
#
# IMPORTANT:
# Must match explicit _C1_, _C2_, etc.
# Do not match the C2 inside "FC2".
# =========================================================

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
# FC2 neuron inventory
# =========================================================

FC2_TYPES = [
    "FC2A",
    "FC2B",
    "FC2C",
]


fc2_neurons = neurons[
    neurons["type"]
    .isin(FC2_TYPES)
].copy()


fc2_neurons["column"] = (
    fc2_neurons[
        "instance"
    ]
    .apply(
        parse_fc2_column
    )
)


# =========================================================
# Validate parser
# =========================================================

if (
    fc2_neurons["column"]
    .isna()
    .any()
):

    failed = fc2_neurons[
        fc2_neurons["column"]
        .isna()
    ]

    print(
        failed[
            [
                "bodyId",
                "type",
                "instance",
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Some FC2 neurons could not "
        "be assigned a column."
    )


fc2_neurons[
    "column"
] = (
    fc2_neurons[
        "column"
    ].astype(int)
)


# =========================================================
# Population counts
# =========================================================

count_table = (
    fc2_neurons
    .groupby(
        [
            "type",
            "column",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


count_table = (
    count_table
    .reindex(
        index=FC2_TYPES,
        columns=range(
            1,
            10,
        ),
        fill_value=0,
    )
)


print("=" * 110)
print("FC2 SUBTYPE x COLUMN NEURON COUNTS")
print("=" * 110)
print()

print(
    count_table.to_string()
)


print()

print(
    "Subtype totals:"
)

print(
    fc2_neurons
    .groupby("type")
    .size()
    .to_string()
)


print()

print(
    f"Grand total: "
    f"{len(fc2_neurons)}"
)


# =========================================================
# FC2 -> PFL edges
# =========================================================

fc2_edges = edges[
    edges["source_type"]
    .isin(FC2_TYPES)
    &
    edges["target_type"]
    .isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
].copy()


fc2_edges[
    "column"
] = (
    fc2_edges[
        "source_instance"
    ]
    .apply(
        parse_fc2_column
    )
)


if (
    fc2_edges["column"]
    .isna()
    .any()
):

    raise RuntimeError(
        "Could not parse one or more "
        "FC2 edge source columns."
    )


fc2_edges[
    "column"
] = (
    fc2_edges[
        "column"
    ].astype(int)
)


# =========================================================
# Overall subtype connectivity
# =========================================================

overall = (
    fc2_edges
    .groupby(
        [
            "source_type",
            "target_type",
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
)


print()
print("=" * 110)
print("OVERALL FC2 SUBTYPE -> PFL CONNECTIVITY")
print("=" * 110)
print()

print(
    overall.to_string(
        index=False
    )
)


# =========================================================
# Subtype + column + PFL target
# =========================================================

target_summary = (
    fc2_edges
    .groupby(
        [
            "source_type",
            "column",
            "target_type",
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
)


print()
print("=" * 110)
print("FC2 SUBTYPE + COLUMN -> PFL TYPE")
print("=" * 110)
print()

print(
    target_summary.to_string(
        index=False
    )
)


# =========================================================
# Target-side connectivity
#
# NOTE:
# This is descriptive anatomical laterality only.
#
# PFL soma side must NOT automatically be interpreted
# as motor direction.
# =========================================================

side_edges = fc2_edges[
    fc2_edges[
        "target_side"
    ].isin(
        [
            "L",
            "R",
        ]
    )
].copy()


side_summary = (
    side_edges
    .groupby(
        [
            "source_type",
            "column",
            "target_side",
        ]
    )
    ["weight"]
    .sum()
    .unstack(
        fill_value=0.0
    )
    .reset_index()
)


if "L" not in side_summary.columns:
    side_summary["L"] = 0.0

if "R" not in side_summary.columns:
    side_summary["R"] = 0.0


side_summary = (
    side_summary
    .rename(
        columns={
            "L": "target_L_weight",
            "R": "target_R_weight",
        }
    )
)


side_summary[
    "total_side_weight"
] = (
    side_summary[
        "target_L_weight"
    ]
    +
    side_summary[
        "target_R_weight"
    ]
)


# =========================================================
# Anatomical laterality bias
#
# +1 -> all weight to R-sided PFL neurons
# -1 -> all weight to L-sided PFL neurons
#  0 -> balanced
#
# Again:
# this is NOT directly a steering score.
# =========================================================

side_summary[
    "laterality_bias"
] = np.where(
    side_summary[
        "total_side_weight"
    ] > 0,

    (
        side_summary[
            "target_R_weight"
        ]
        -
        side_summary[
            "target_L_weight"
        ]
    )
    /
    side_summary[
        "total_side_weight"
    ],

    0.0,
)


print()
print("=" * 110)
print("FC2 SUBTYPE + COLUMN TARGET LATERALITY")
print("=" * 110)
print()

print(
    side_summary[
        [
            "source_type",
            "column",
            "target_L_weight",
            "target_R_weight",
            "laterality_bias",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.3f}",
    )
)


# =========================================================
# PFL2 / PFL3 weight split per FC2 subtype + column
# =========================================================

pfl_split = (
    fc2_edges
    .groupby(
        [
            "source_type",
            "column",
            "target_type",
        ]
    )
    ["weight"]
    .sum()
    .unstack(
        fill_value=0.0
    )
    .reset_index()
)


if "PFL2" not in pfl_split.columns:
    pfl_split["PFL2"] = 0.0

if "PFL3" not in pfl_split.columns:
    pfl_split["PFL3"] = 0.0


pfl_split = (
    pfl_split.rename(
        columns={
            "PFL2":
                "PFL2_weight",

            "PFL3":
                "PFL3_weight",
        }
    )
)


pfl_split[
    "total_weight"
] = (
    pfl_split[
        "PFL2_weight"
    ]
    +
    pfl_split[
        "PFL3_weight"
    ]
)


pfl_split[
    "PFL3_fraction"
] = np.where(
    pfl_split[
        "total_weight"
    ] > 0,

    pfl_split[
        "PFL3_weight"
    ]
    /
    pfl_split[
        "total_weight"
    ],

    0.0,
)


# =========================================================
# Combine diagnostic tables
# =========================================================

diagnostic = (
    pfl_split
    .merge(
        side_summary[
            [
                "source_type",
                "column",
                "target_L_weight",
                "target_R_weight",
                "laterality_bias",
            ]
        ],

        on=[
            "source_type",
            "column",
        ],

        how="left",
    )
)


diagnostic = (
    diagnostic
    .sort_values(
        [
            "source_type",
            "column",
        ]
    )
)


print()
print("=" * 110)
print("FINAL FC2 SUBTYPE/COLUMN DIAGNOSTIC")
print("=" * 110)
print()

print(
    diagnostic.to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.3f}",
    )
)


# =========================================================
# Strongest lateralized columns
# =========================================================

print()
print("=" * 110)
print("STRONGEST ANATOMICAL LATERALITY BIASES")
print("=" * 110)
print()


strongest = (
    diagnostic
    .assign(
        absolute_bias=lambda df:
        df[
            "laterality_bias"
        ].abs()
    )
    .sort_values(
        "absolute_bias",
        ascending=False,
    )
)


print(
    strongest[
        [
            "source_type",
            "column",
            "PFL2_weight",
            "PFL3_weight",
            "target_L_weight",
            "target_R_weight",
            "laterality_bias",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.3f}",
    )
)


# =========================================================
# Save
# =========================================================

diagnostic.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(
    OUTPUT_FILE
)