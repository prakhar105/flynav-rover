import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
ENV_FILE = BASE_DIR / ".env"

STEERING_FILE = (
    OUTPUT_DIR
    / "steering_core_neurons.csv"
)

DELTA7_FILE = (
    OUTPUT_DIR
    / "delta7_candidates.csv"
)

EDGE_OUTPUT = (
    OUTPUT_DIR
    / "delta7_bridge_edges.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "delta7_bridge_summary.csv"
)


# =========================================================
# Credentials
# =========================================================

load_dotenv(ENV_FILE)

token = os.getenv(
    "NEUPRINT_TOKEN"
)

if not token:
    raise RuntimeError(
        f"NEUPRINT_TOKEN not found in {ENV_FILE}"
    )


# =========================================================
# neuPrint
# =========================================================

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

print(
    "Connected to:",
    client.dataset
)


# =========================================================
# Load neurons
# =========================================================

steering = pd.read_csv(
    STEERING_FILE
)

delta7 = pd.read_csv(
    DELTA7_FILE
)

steering["bodyId"] = (
    steering["bodyId"]
    .astype("int64")
)

delta7["bodyId"] = (
    delta7["bodyId"]
    .astype("int64")
)


# =========================================================
# Keep navigation-relevant populations
# =========================================================

STEERING_TYPES = [
    "EPG",
    "FC2A",
    "FC2B",
    "FC2C",
    "PFL2",
    "PFL3",
]


steering = steering[
    steering["type"].isin(
        STEERING_TYPES
    )
].copy()


combined = pd.concat(
    [
        steering,
        delta7,
    ],
    ignore_index=True,
)


combined = (
    combined
    .drop_duplicates(
        subset=["bodyId"]
    )
)


body_ids = [
    int(x)
    for x in combined["bodyId"]
]


body_id_string = ", ".join(
    str(x)
    for x in body_ids
)


print()
print("=" * 100)
print("DELTA7 NAVIGATION NETWORK")
print("=" * 100)

print(
    f"Total neurons considered: "
    f"{len(combined)}"
)

print()

print(
    combined.groupby("type")
    .size()
    .sort_values(
        ascending=False
    )
    .to_string()
)


# =========================================================
# Query actual direct connectivity
# =========================================================

query = f"""
MATCH
    (source:Neuron)-[c:ConnectsTo]->(target:Neuron)

WHERE source.bodyId IN [{body_id_string}]
AND target.bodyId IN [{body_id_string}]

RETURN
    source.bodyId AS source_bodyId,
    source.type AS source_type,
    source.instance AS source_instance,
    source.somaSide AS source_side,
    source.predictedNt AS source_nt,

    target.bodyId AS target_bodyId,
    target.type AS target_type,
    target.instance AS target_instance,
    target.somaSide AS target_side,

    c.weight AS weight

ORDER BY
    c.weight DESC
"""


edges = client.fetch_custom(
    query
)


if edges.empty:

    raise RuntimeError(
        "No connectivity found."
    )


edges["weight"] = pd.to_numeric(
    edges["weight"],
    errors="coerce",
).fillna(0.0)


edges.to_csv(
    EDGE_OUTPUT,
    index=False,
)


print()
print(
    f"Total direct connections: "
    f"{len(edges)}"
)


# =========================================================
# Keep functional paths we care about
# =========================================================

functional_edges = edges[
    (
        (
            edges["source_type"]
            == "EPG"
        )
        &
        (
            edges["target_type"]
            == "Delta7"
        )
    )
    |
    (
        (
            edges["source_type"]
            == "Delta7"
        )
        &
        (
            edges["target_type"]
            .isin(
                [
                    "PFL2",
                    "PFL3",
                ]
            )
        )
    )
    |
    (
        (
            edges["source_type"]
            .isin(
                [
                    "FC2A",
                    "FC2B",
                    "FC2C",
                ]
            )
        )
        &
        (
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


# =========================================================
# Type-to-type summary
# =========================================================

summary = (
    functional_edges
    .groupby(
        [
            "source_type",
            "target_type",
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
print("TYPE -> TYPE CONNECTIVITY")
print("=" * 100)
print()

print(
    summary.to_string(
        index=False
    )
)


# =========================================================
# Strongest EPG -> Delta7
# =========================================================

epg_delta = functional_edges[
    (
        functional_edges["source_type"]
        == "EPG"
    )
    &
    (
        functional_edges["target_type"]
        == "Delta7"
    )
].copy()


print()
print("=" * 100)
print("STRONGEST EPG -> DELTA7")
print("=" * 100)
print()

print(
    epg_delta[
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
    .head(40)
    .to_string(
        index=False
    )
)


# =========================================================
# Strongest Delta7 -> PFL
# =========================================================

delta_pfl = functional_edges[
    (
        functional_edges["source_type"]
        == "Delta7"
    )
    &
    (
        functional_edges["target_type"]
        .isin(
            [
                "PFL2",
                "PFL3",
            ]
        )
    )
].copy()


print()
print("=" * 100)
print("STRONGEST DELTA7 -> PFL")
print("=" * 100)
print()

print(
    delta_pfl[
        [
            "source_instance",
            "source_side",
            "source_nt",
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
    .head(60)
    .to_string(
        index=False
    )
)


# =========================================================
# Hemisphere behaviour
# =========================================================

laterality = (
    delta_pfl[
        delta_pfl["source_side"]
        .isin(["L", "R"])
        &
        delta_pfl["target_side"]
        .isin(["L", "R"])
    ]
    .copy()
)


laterality["relationship"] = (
    laterality.apply(
        lambda row:
        "ipsilateral"
        if (
            row["source_side"]
            == row["target_side"]
        )
        else "contralateral",
        axis=1,
    )
)


laterality_summary = (
    laterality
    .groupby(
        [
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
print("DELTA7 -> PFL LATERALITY")
print("=" * 100)
print()

print(
    laterality_summary.to_string(
        index=False
    )
)


# =========================================================
# Delta7 target distribution
# =========================================================

target_distribution = (
    delta_pfl
    .groupby(
        [
            "source_instance",
            "target_type",
        ]
    )
    ["weight"]
    .sum()
    .reset_index()
    .sort_values(
        "weight",
        ascending=False,
    )
)


print()
print("=" * 100)
print("DELTA7 TARGET DISTRIBUTION")
print("=" * 100)
print()

print(
    target_distribution
    .head(50)
    .to_string(
        index=False
    )
)


# =========================================================
# FC2 comparison
# =========================================================

fc2_pfl = functional_edges[
    functional_edges[
        "source_type"
    ].isin(
        [
            "FC2A",
            "FC2B",
            "FC2C",
        ]
    )
].copy()


fc2_summary = (
    fc2_pfl
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
        connection_count=(
            "weight",
            "size",
        ),
    )
    .reset_index()
)


print()
print("=" * 100)
print("FC2 -> PFL REFERENCE")
print("=" * 100)
print()

print(
    fc2_summary.to_string(
        index=False
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
    EDGE_OUTPUT
)

print(
    SUMMARY_OUTPUT
)