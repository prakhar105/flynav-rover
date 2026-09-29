import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client


# ============================================================
# STEP 52 - TRACE LC10a TARGET-VISION PATHWAYS TO DESCENDING
#            MOTOR CHANNELS
#
# Why:
#   Step 51 showed that TuBu/ER/EPG visual profiles are real
#   but broad/diffuse. That route is useful for visual-compass
#   influence, but it is not a clean LEFT/CENTER/RIGHT motor
#   command.
#
#   For actual target pursuit we now ask a different question:
#
#       Does LC10a reach descending neurons directly,
#       or through one central-brain intermediate?
#
# Scientific boundary:
#   - Neuron identities and edge weights come from MaleCNS.
#   - "bottleneck_score" is an engineering ranking heuristic:
#         min(weight_1, weight_2)
#     summed across matching paths.
#   - This is NOT a biological gain or firing-rate model.
#
# Outputs:
#   52_lc10a_to_descending_direct_edges.csv
#   52_lc10a_to_descending_direct_summary.csv
#   52_lc10a_to_descending_two_hop_summary.csv
#   52_lc10a_to_descending_side_summary.csv
#   52_lc10a_descending_candidates.csv
#   52_candidate_dn_to_flynav_core.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "connectome" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

TOKEN = os.getenv("NEUPRINT_TOKEN")
if not TOKEN:
    raise RuntimeError(
        "NEUPRINT_TOKEN was not found in the project .env file."
    )

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=TOKEN,
)


def save(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


print()
print("=" * 118)
print("STEP 52 - TRACE LC10a TARGET-VISION PATHWAYS TO DESCENDING MOTOR CHANNELS")
print("=" * 118)
print()
print(
    "Question: can LC10a target vision reach descending neurons "
    "without forcing the broad TuBu/ER/EPG route to act like a motor command?"
)
print()


# ------------------------------------------------------------
# 1. Direct LC10a -> descending-neuron edges
# ------------------------------------------------------------

direct_query = """
MATCH (s:Neuron)-[e:ConnectsTo]->(d:Neuron)
WHERE s.type = 'LC10a'
  AND d.superclass = 'descending_neuron'
RETURN
    s.bodyId AS source_bodyId,
    s.instance AS source_instance,
    s.somaSide AS source_side,
    d.bodyId AS target_bodyId,
    d.type AS target_type,
    d.instance AS target_instance,
    d.somaSide AS target_side,
    d.predictedNt AS target_predictedNt,
    e.weight AS weight
ORDER BY weight DESC
"""

direct_edges = client.fetch_custom(direct_query)

direct_edges_path = save(
    direct_edges,
    "52_lc10a_to_descending_direct_edges.csv",
)

if direct_edges.empty:
    direct_summary = pd.DataFrame(
        columns=[
            "target_type",
            "total_weight",
            "edge_count",
            "source_neurons",
            "target_neurons",
            "max_weight",
            "mean_weight",
        ]
    )
else:
    direct_summary = (
        direct_edges.groupby(
            "target_type",
            dropna=False,
        )
        .agg(
            total_weight=("weight", "sum"),
            edge_count=("weight", "count"),
            source_neurons=("source_bodyId", "nunique"),
            target_neurons=("target_bodyId", "nunique"),
            max_weight=("weight", "max"),
            mean_weight=("weight", "mean"),
        )
        .reset_index()
        .sort_values(
            ["total_weight", "edge_count"],
            ascending=[False, False],
        )
    )

direct_summary_path = save(
    direct_summary,
    "52_lc10a_to_descending_direct_summary.csv",
)


# ------------------------------------------------------------
# 2. LC10a -> central-brain intermediate -> descending
#
# We exclude recurrent LC10a-as-intermediate paths because they
# can dominate counts without telling us which downstream
# central-brain pathway is useful for action.
# ------------------------------------------------------------

two_hop_query = """
MATCH (s:Neuron)-[e1:ConnectsTo]->(m:Neuron)
      -[e2:ConnectsTo]->(d:Neuron)
WHERE s.type = 'LC10a'
  AND d.superclass = 'descending_neuron'
  AND m.type <> 'LC10a'
  AND m.bodyId <> s.bodyId
WITH
    s.somaSide AS source_side,
    m.type AS intermediate_type,
    m.superclass AS intermediate_superclass,
    d.type AS target_type,
    d.somaSide AS target_side,
    CASE
        WHEN e1.weight < e2.weight
        THEN e1.weight
        ELSE e2.weight
    END AS bottleneck_weight
RETURN
    source_side,
    intermediate_type,
    intermediate_superclass,
    target_type,
    target_side,
    count(*) AS path_count,
    sum(bottleneck_weight) AS bottleneck_score,
    max(bottleneck_weight) AS max_bottleneck_weight,
    avg(toFloat(bottleneck_weight)) AS mean_bottleneck_weight
ORDER BY bottleneck_score DESC, path_count DESC
LIMIT 500
"""

two_hop = client.fetch_custom(two_hop_query)

two_hop_path = save(
    two_hop,
    "52_lc10a_to_descending_two_hop_summary.csv",
)


# ------------------------------------------------------------
# 3. Side / laterality summary
# ------------------------------------------------------------

side_rows = []

if not direct_edges.empty:
    tmp = (
        direct_edges.groupby(
            [
                "source_side",
                "target_side",
            ],
            dropna=False,
        )
        .agg(
            total_weight=("weight", "sum"),
            edge_count=("weight", "count"),
            source_neurons=("source_bodyId", "nunique"),
            target_neurons=("target_bodyId", "nunique"),
        )
        .reset_index()
    )

    for row in tmp.itertuples(index=False):
        side_rows.append(
            {
                "path_kind": "direct",
                "source_side": row.source_side,
                "target_side": row.target_side,
                "total_score": float(row.total_weight),
                "path_or_edge_count": int(row.edge_count),
                "source_neurons": int(row.source_neurons),
                "target_neurons": int(row.target_neurons),
            }
        )

if not two_hop.empty:
    tmp = (
        two_hop.groupby(
            [
                "source_side",
                "target_side",
            ],
            dropna=False,
        )
        .agg(
            total_score=("bottleneck_score", "sum"),
            path_or_edge_count=("path_count", "sum"),
        )
        .reset_index()
    )

    for row in tmp.itertuples(index=False):
        side_rows.append(
            {
                "path_kind": "two_hop",
                "source_side": row.source_side,
                "target_side": row.target_side,
                "total_score": float(row.total_score),
                "path_or_edge_count": int(row.path_or_edge_count),
                "source_neurons": "",
                "target_neurons": "",
            }
        )

side_summary = pd.DataFrame(side_rows)

side_summary_path = save(
    side_summary,
    "52_lc10a_to_descending_side_summary.csv",
)


# ------------------------------------------------------------
# 4. Merge direct + two-hop evidence into candidate DN ranking
#
# The combined score is ONLY for experiment selection.
# Direct anatomical evidence gets priority.
# ------------------------------------------------------------

candidate_rows = {}

for row in direct_summary.itertuples(index=False):
    target = str(row.target_type)

    candidate_rows.setdefault(
        target,
        {
            "target_type": target,
            "direct_total_weight": 0.0,
            "direct_edge_count": 0,
            "direct_source_neurons": 0,
            "two_hop_bottleneck_score": 0.0,
            "two_hop_path_count": 0,
            "two_hop_intermediate_types": 0,
        },
    )

    candidate_rows[target][
        "direct_total_weight"
    ] = float(row.total_weight)

    candidate_rows[target][
        "direct_edge_count"
    ] = int(row.edge_count)

    candidate_rows[target][
        "direct_source_neurons"
    ] = int(row.source_neurons)


if not two_hop.empty:
    grouped = (
        two_hop.groupby(
            "target_type",
            dropna=False,
        )
        .agg(
            two_hop_bottleneck_score=(
                "bottleneck_score",
                "sum",
            ),
            two_hop_path_count=(
                "path_count",
                "sum",
            ),
            two_hop_intermediate_types=(
                "intermediate_type",
                "nunique",
            ),
        )
        .reset_index()
    )

    for row in grouped.itertuples(index=False):
        target = str(row.target_type)

        candidate_rows.setdefault(
            target,
            {
                "target_type": target,
                "direct_total_weight": 0.0,
                "direct_edge_count": 0,
                "direct_source_neurons": 0,
                "two_hop_bottleneck_score": 0.0,
                "two_hop_path_count": 0,
                "two_hop_intermediate_types": 0,
            },
        )

        candidate_rows[target][
            "two_hop_bottleneck_score"
        ] = float(
            row.two_hop_bottleneck_score
        )

        candidate_rows[target][
            "two_hop_path_count"
        ] = int(
            row.two_hop_path_count
        )

        candidate_rows[target][
            "two_hop_intermediate_types"
        ] = int(
            row.two_hop_intermediate_types
        )


candidates = pd.DataFrame(
    list(candidate_rows.values())
)

if not candidates.empty:
    direct_max = max(
        float(
            candidates[
                "direct_total_weight"
            ].max()
        ),
        1.0,
    )

    two_hop_max = max(
        float(
            candidates[
                "two_hop_bottleneck_score"
            ].max()
        ),
        1.0,
    )

    candidates[
        "direct_norm"
    ] = (
        candidates[
            "direct_total_weight"
        ]
        / direct_max
    )

    candidates[
        "two_hop_norm"
    ] = (
        candidates[
            "two_hop_bottleneck_score"
        ]
        / two_hop_max
    )

    candidates[
        "candidate_score"
    ] = (
        0.70
        * candidates["direct_norm"]
        +
        0.30
        * candidates["two_hop_norm"]
    )

    candidates = candidates.sort_values(
        [
            "candidate_score",
            "direct_total_weight",
        ],
        ascending=[
            False,
            False,
        ],
    )

candidate_path = save(
    candidates,
    "52_lc10a_descending_candidates.csv",
)


# ------------------------------------------------------------
# 5. Do top candidate DNs connect onward to the already-used
#    FlyNav steering core?
#
# This is exploratory. Absence does NOT invalidate the visual
# target pathway; it may simply mean a separate descending
# action channel is required.
# ------------------------------------------------------------

top_types = []

if not candidates.empty:
    top_types = (
        candidates[
            "target_type"
        ]
        .dropna()
        .astype(str)
        .head(20)
        .tolist()
    )

if top_types:
    escaped = [
        x.replace("\\", "\\\\").replace(
            "'", "\\'"
        )
        for x in top_types
    ]

    cypher_list = ", ".join(
        f"'{x}'"
        for x in escaped
    )

    candidate_to_core_query = f"""
    MATCH (d:Neuron)-[e:ConnectsTo]->(t:Neuron)
    WHERE d.type IN [{cypher_list}]
      AND (
           t.type = 'EPG'
        OR t.type STARTS WITH 'FC2'
        OR t.type IN ['PFL2', 'PFL3', 'DNa02', 'DNa03']
      )
    RETURN
        d.type AS source_dn_type,
        t.type AS target_type,
        count(e) AS edge_count,
        sum(e.weight) AS total_weight,
        max(e.weight) AS max_weight,
        count(DISTINCT d) AS source_neurons,
        count(DISTINCT t) AS target_neurons
    ORDER BY total_weight DESC, edge_count DESC
    """

    candidate_to_core = client.fetch_custom(
        candidate_to_core_query
    )

else:
    candidate_to_core = pd.DataFrame(
        columns=[
            "source_dn_type",
            "target_type",
            "edge_count",
            "total_weight",
            "max_weight",
            "source_neurons",
            "target_neurons",
        ]
    )

candidate_to_core_path = save(
    candidate_to_core,
    "52_candidate_dn_to_flynav_core.csv",
)


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("=" * 118)
print("DIRECT LC10a -> DESCENDING")
print("=" * 118)

if direct_summary.empty:
    print("NO DIRECT LC10a -> descending-neuron edges found.")
else:
    print(
        direct_summary.head(60).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("TOP TWO-HOP LC10a -> INTERMEDIATE -> DESCENDING")
print("=" * 118)

if two_hop.empty:
    print("NO TWO-HOP PATHS FOUND.")
else:
    print(
        two_hop.head(80).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("LATERALITY SUMMARY")
print("=" * 118)

if side_summary.empty:
    print("No laterality summary available.")
else:
    print(
        side_summary.to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("DESCENDING CANDIDATES FOR TARGET-PURSUIT EXPERIMENT")
print("=" * 118)

if candidates.empty:
    print("No descending candidates found.")
else:
    print(
        candidates.head(30).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("TOP CANDIDATE DN -> EXISTING FLYNAV CORE")
print("=" * 118)

if candidate_to_core.empty:
    print(
        "No direct edges from the top visual target DNs "
        "into EPG/FC2/PFL2/PFL3/DNa02/DNa03 were found."
    )
else:
    print(
        candidate_to_core.to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("FILES CREATED")
print("=" * 118)

for path in [
    direct_edges_path,
    direct_summary_path,
    two_hop_path,
    side_summary_path,
    candidate_path,
    candidate_to_core_path,
]:
    print(path)

print()
print("STEP 52 COMPLETE.")
print(
    "NEXT: decide whether target pursuit should use a "
    "separate visual descending action channel or the "
    "existing goal-navigation interface, based on this evidence."
)
