import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client


# ============================================================
# STEP 47 - TRACE THE ANTERIOR VISUAL PATHWAY INTO EPG
#
# Goal:
#   Test the canonical visual-navigation route in MaleCNS:
#
#       MeTu -> TuBu -> ER -> EPG
#
#   and also ask which visual_projection types feed TuBu.
#
# Scientific boundary:
#   - Neuron identities and edge weights come from MaleCNS.
#   - Raw synapse counts are anatomical evidence, not assumed
#     functional gains.
#   - No Webots camera mapping is introduced in this step.
#
# Outputs:
#   47_avp_neuron_inventory.csv
#   47_visual_projection_to_tubu_summary.csv
#   47_metu_to_tubu_edges.csv
#   47_metu_to_tubu_summary.csv
#   47_tubu_to_er_edges.csv
#   47_tubu_to_er_summary.csv
#   47_er_to_epg_edges.csv
#   47_er_to_epg_summary.csv
#   47_avp_stage_summary.csv
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


def fetch_stage_edges(source_condition, target_condition):
    query = f"""
    MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron)
    WHERE ({source_condition})
      AND ({target_condition})
    RETURN
        a.bodyId AS source_bodyId,
        a.type AS source_type,
        a.instance AS source_instance,
        a.somaSide AS source_side,
        a.superclass AS source_superclass,
        b.bodyId AS target_bodyId,
        b.type AS target_type,
        b.instance AS target_instance,
        b.somaSide AS target_side,
        b.superclass AS target_superclass,
        e.weight AS weight
    ORDER BY weight DESC
    """
    return client.fetch_custom(query)


def summarize_edges(edges):
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "source_type",
                "target_type",
                "total_weight",
                "edge_count",
                "source_neurons",
                "target_neurons",
                "max_weight",
                "mean_weight",
            ]
        )

    return (
        edges.groupby(
            ["source_type", "target_type"],
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


print()
print("=" * 116)
print("STEP 47 - TRACE MALECNS ANTERIOR VISUAL PATHWAY INTO EPG")
print("=" * 116)
print()
print("Canonical route under test:")
print("    MeTu -> TuBu -> ER -> EPG")
print()
print(
    "Raw MaleCNS edge weights are retained as anatomical "
    "synapse counts; no functional gain is inferred."
)
print()


# ------------------------------------------------------------
# 1. Inventory candidate AVP neuron types.
# ------------------------------------------------------------

inventory_query = """
MATCH (n:Neuron)
WHERE
      n.type STARTS WITH 'MeTu'
   OR n.type STARTS WITH 'TuBu'
   OR n.type STARTS WITH 'ER'
   OR n.type = 'EPG'
RETURN
    n.bodyId AS bodyId,
    n.type AS type,
    n.instance AS instance,
    n.somaSide AS somaSide,
    n.superclass AS superclass,
    n.class AS class,
    n.subclass AS subclass,
    n.predictedNt AS predictedNt,
    n.pre AS pre,
    n.post AS post
ORDER BY type, somaSide, bodyId
"""

inventory = client.fetch_custom(inventory_query)
inventory_path = save(
    inventory,
    "47_avp_neuron_inventory.csv",
)

print(f"AVP/EPG candidate neurons found: {len(inventory):,}")

if not inventory.empty:
    type_inventory = (
        inventory.groupby(
            ["type", "superclass"],
            dropna=False,
        )
        .agg(
            neuron_count=("bodyId", "count"),
            left_count=(
                "somaSide",
                lambda s: int(
                    sum(str(x).lower().startswith("l") for x in s)
                ),
            ),
            right_count=(
                "somaSide",
                lambda s: int(
                    sum(str(x).lower().startswith("r") for x in s)
                ),
            ),
        )
        .reset_index()
        .sort_values(
            ["neuron_count", "type"],
            ascending=[False, True],
        )
    )

    print()
    print("Candidate type inventory:")
    print(type_inventory.to_string(index=False))


# ------------------------------------------------------------
# 2. Ask which visual_projection types actually feed TuBu.
# ------------------------------------------------------------

vpn_to_tubu_query = """
MATCH (v:Neuron)-[e:ConnectsTo]->(t:Neuron)
WHERE v.superclass = 'visual_projection'
  AND t.type STARTS WITH 'TuBu'
RETURN
    v.type AS source_type,
    t.type AS target_type,
    count(e) AS edge_count,
    sum(e.weight) AS total_weight,
    max(e.weight) AS max_weight,
    count(DISTINCT v) AS source_neurons,
    count(DISTINCT t) AS target_neurons
ORDER BY total_weight DESC, edge_count DESC
"""

vpn_to_tubu = client.fetch_custom(vpn_to_tubu_query)

vpn_to_tubu_path = save(
    vpn_to_tubu,
    "47_visual_projection_to_tubu_summary.csv",
)


# ------------------------------------------------------------
# 3. Canonical stage 1: MeTu -> TuBu.
# ------------------------------------------------------------

metu_tubu_edges = fetch_stage_edges(
    "a.type STARTS WITH 'MeTu'",
    "b.type STARTS WITH 'TuBu'",
)

metu_tubu_summary = summarize_edges(metu_tubu_edges)

metu_tubu_edges_path = save(
    metu_tubu_edges,
    "47_metu_to_tubu_edges.csv",
)

metu_tubu_summary_path = save(
    metu_tubu_summary,
    "47_metu_to_tubu_summary.csv",
)


# ------------------------------------------------------------
# 4. Canonical stage 2: TuBu -> ER.
# ------------------------------------------------------------

tubu_er_edges = fetch_stage_edges(
    "a.type STARTS WITH 'TuBu'",
    "b.type STARTS WITH 'ER'",
)

tubu_er_summary = summarize_edges(tubu_er_edges)

tubu_er_edges_path = save(
    tubu_er_edges,
    "47_tubu_to_er_edges.csv",
)

tubu_er_summary_path = save(
    tubu_er_summary,
    "47_tubu_to_er_summary.csv",
)


# ------------------------------------------------------------
# 5. Canonical stage 3: ER -> EPG.
# ------------------------------------------------------------

er_epg_edges = fetch_stage_edges(
    "a.type STARTS WITH 'ER'",
    "b.type = 'EPG'",
)

er_epg_summary = summarize_edges(er_epg_edges)

er_epg_edges_path = save(
    er_epg_edges,
    "47_er_to_epg_edges.csv",
)

er_epg_summary_path = save(
    er_epg_summary,
    "47_er_to_epg_summary.csv",
)


# ------------------------------------------------------------
# 6. Compact stage-level summary.
# ------------------------------------------------------------

def stage_row(name, edges):
    if edges.empty:
        return {
            "stage": name,
            "edge_count": 0,
            "total_weight": 0,
            "source_neurons": 0,
            "target_neurons": 0,
            "source_types": 0,
            "target_types": 0,
            "max_weight": 0,
        }

    return {
        "stage": name,
        "edge_count": int(len(edges)),
        "total_weight": int(edges["weight"].sum()),
        "source_neurons": int(edges["source_bodyId"].nunique()),
        "target_neurons": int(edges["target_bodyId"].nunique()),
        "source_types": int(edges["source_type"].nunique()),
        "target_types": int(edges["target_type"].nunique()),
        "max_weight": int(edges["weight"].max()),
    }


stage_summary = pd.DataFrame(
    [
        stage_row("MeTu -> TuBu", metu_tubu_edges),
        stage_row("TuBu -> ER", tubu_er_edges),
        stage_row("ER -> EPG", er_epg_edges),
    ]
)

stage_summary_path = save(
    stage_summary,
    "47_avp_stage_summary.csv",
)


# ------------------------------------------------------------
# Console report.
# ------------------------------------------------------------

print()
print("=" * 116)
print("VISUAL_PROJECTION -> TuBu")
print("=" * 116)

if vpn_to_tubu.empty:
    print("No matching edges found.")
else:
    print(
        vpn_to_tubu.head(40).to_string(
            index=False
        )
    )


def print_stage(title, edges, summary):
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)

    if edges.empty:
        print("NO EDGES FOUND")
        return

    print(
        f"Edges: {len(edges):,} | "
        f"total raw weight: {int(edges['weight'].sum()):,} | "
        f"source neurons: {edges['source_bodyId'].nunique():,} | "
        f"target neurons: {edges['target_bodyId'].nunique():,}"
    )

    print()
    print("Strongest type -> type pairs:")
    print(
        summary.head(30).to_string(
            index=False
        )
    )

    print()
    print("Strongest individual edges:")
    cols = [
        "source_type",
        "source_instance",
        "source_side",
        "target_type",
        "target_instance",
        "target_side",
        "weight",
    ]

    cols = [
        c for c in cols
        if c in edges.columns
    ]

    print(
        edges[cols]
        .head(30)
        .to_string(index=False)
    )


print_stage(
    "STAGE 1 - MeTu -> TuBu",
    metu_tubu_edges,
    metu_tubu_summary,
)

print_stage(
    "STAGE 2 - TuBu -> ER",
    tubu_er_edges,
    tubu_er_summary,
)

print_stage(
    "STAGE 3 - ER -> EPG",
    er_epg_edges,
    er_epg_summary,
)

print()
print("=" * 116)
print("STAGE SUMMARY")
print("=" * 116)
print(stage_summary.to_string(index=False))

print()
print("=" * 116)
print("FILES CREATED")
print("=" * 116)

for path in [
    inventory_path,
    vpn_to_tubu_path,
    metu_tubu_edges_path,
    metu_tubu_summary_path,
    tubu_er_edges_path,
    tubu_er_summary_path,
    er_epg_edges_path,
    er_epg_summary_path,
    stage_summary_path,
]:
    print(path)

print()
print("STEP 47 COMPLETE.")
print(
    "NEXT: use the measured pathway to choose the first "
    "visual navigation interface, then separately trace "
    "object-target and looming/threat visual channels."
)
