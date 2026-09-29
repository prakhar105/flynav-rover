import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client


# ============================================================
# STEP 48 - TRACE VISUAL TARGET + THREAT CHANNELS
#
# Purpose:
#   After Step 47 confirmed MeTu -> TuBu -> ER -> EPG,
#   examine two additional visual channels that matter for a
#   rover navigation unit:
#
#   A) LC10a  -> object / target channel
#      Test whether LC10a enters the TuBu -> ER -> EPG route.
#
#   B) LPLC2 / LC4 -> descending channel
#      Test direct and two-hop access to descending neurons.
#
# Scientific boundary:
#   - All neuron IDs/types and synapse weights come from MaleCNS.
#   - Raw weights are anatomical synapse counts.
#   - Two-hop "bottleneck score" is ONLY a structural ranking
#     heuristic: min(weight1, weight2), not a biological gain.
#   - No camera pixels are mapped to neurons in this step.
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


def fetch(query):
    return client.fetch_custom(query)


print()
print("=" * 118)
print("STEP 48 - MALECNS VISUAL TARGET + THREAT CHANNELS")
print("=" * 118)
print()
print("Target channel under test:")
print("    LC10a -> TuBu -> ER -> EPG")
print()
print("Threat / avoidance channels under test:")
print("    LPLC2 / LC4 -> descending neurons")
print("    LPLC2 / LC4 -> intermediate -> descending neurons")
print()


# ============================================================
# 1. Source inventory
# ============================================================

source_inventory = fetch(
    """
    MATCH (n:Neuron)
    WHERE n.type IN ['LC10a', 'LPLC2', 'LC4']
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
)

source_inventory_path = save(
    source_inventory,
    "48_visual_source_inventory.csv",
)

source_counts = (
    source_inventory.groupby(
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
)

print("Source inventory:")
print(source_counts.to_string(index=False))


# ============================================================
# 2. LC10a -> TuBu
# ============================================================

lc10a_tubu = fetch(
    """
    MATCH (s:Neuron)-[e:ConnectsTo]->(t:Neuron)
    WHERE s.type = 'LC10a'
      AND t.type STARTS WITH 'TuBu'
    RETURN
        s.bodyId AS source_bodyId,
        s.instance AS source_instance,
        s.somaSide AS source_side,
        t.bodyId AS target_bodyId,
        t.type AS target_type,
        t.instance AS target_instance,
        t.somaSide AS target_side,
        e.weight AS weight
    ORDER BY weight DESC
    """
)

lc10a_tubu_path = save(
    lc10a_tubu,
    "48_lc10a_to_tubu_edges.csv",
)


# ============================================================
# 3. LC10a-linked TuBu -> ER
#
# DISTINCT is important: we want each anatomical TuBu->ER
# relationship once, even if many LC10a neurons reach the TuBu.
# ============================================================

lc10a_tubu_er = fetch(
    """
    MATCH (lc:Neuron)-[:ConnectsTo]->(tb:Neuron)
          -[e:ConnectsTo]->(er:Neuron)
    WHERE lc.type = 'LC10a'
      AND tb.type STARTS WITH 'TuBu'
      AND er.type STARTS WITH 'ER'
    WITH DISTINCT tb, er, e
    RETURN
        tb.bodyId AS source_bodyId,
        tb.type AS source_type,
        tb.instance AS source_instance,
        tb.somaSide AS source_side,
        er.bodyId AS target_bodyId,
        er.type AS target_type,
        er.instance AS target_instance,
        er.somaSide AS target_side,
        e.weight AS weight
    ORDER BY weight DESC
    """
)

lc10a_tubu_er_path = save(
    lc10a_tubu_er,
    "48_lc10a_linked_tubu_to_er_edges.csv",
)


# ============================================================
# 4. LC10a-linked ER -> EPG
# ============================================================

lc10a_er_epg = fetch(
    """
    MATCH (lc:Neuron)-[:ConnectsTo]->(tb:Neuron)
          -[:ConnectsTo]->(er:Neuron)
          -[e:ConnectsTo]->(epg:Neuron)
    WHERE lc.type = 'LC10a'
      AND tb.type STARTS WITH 'TuBu'
      AND er.type STARTS WITH 'ER'
      AND epg.type = 'EPG'
    WITH DISTINCT er, epg, e
    RETURN
        er.bodyId AS source_bodyId,
        er.type AS source_type,
        er.instance AS source_instance,
        er.somaSide AS source_side,
        epg.bodyId AS target_bodyId,
        epg.type AS target_type,
        epg.instance AS target_instance,
        epg.somaSide AS target_side,
        e.weight AS weight
    ORDER BY weight DESC
    """
)

lc10a_er_epg_path = save(
    lc10a_er_epg,
    "48_lc10a_linked_er_to_epg_edges.csv",
)


# ============================================================
# 5. Compact LC10a target-path summary
# ============================================================

def edge_stage(name, df):
    if df.empty:
        return {
            "stage": name,
            "edge_count": 0,
            "total_weight": 0,
            "source_neurons": 0,
            "target_neurons": 0,
            "max_weight": 0,
        }

    return {
        "stage": name,
        "edge_count": int(len(df)),
        "total_weight": int(df["weight"].sum()),
        "source_neurons": int(df["source_bodyId"].nunique()),
        "target_neurons": int(df["target_bodyId"].nunique()),
        "max_weight": int(df["weight"].max()),
    }


target_path_summary = pd.DataFrame(
    [
        edge_stage("LC10a -> TuBu", lc10a_tubu),
        edge_stage(
            "LC10a-linked TuBu -> ER",
            lc10a_tubu_er,
        ),
        edge_stage(
            "LC10a-linked ER -> EPG",
            lc10a_er_epg,
        ),
    ]
)

target_path_summary_path = save(
    target_path_summary,
    "48_lc10a_target_path_summary.csv",
)


# ============================================================
# 6. LC10a-linked EPG column distribution
#
# This does NOT mean LC10a "commands" a heading column.
# It simply reports which existing EPG instance labels receive
# anatomical input from ER neurons reachable through the
# LC10a-linked TuBu route.
# ============================================================

if lc10a_er_epg.empty:
    epg_distribution = pd.DataFrame(
        columns=[
            "target_instance",
            "total_weight",
            "edge_count",
            "source_er_neurons",
        ]
    )
else:
    epg_distribution = (
        lc10a_er_epg.groupby(
            "target_instance",
            dropna=False,
        )
        .agg(
            total_weight=("weight", "sum"),
            edge_count=("weight", "count"),
            source_er_neurons=("source_bodyId", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["total_weight", "edge_count"],
            ascending=[False, False],
        )
    )

epg_distribution_path = save(
    epg_distribution,
    "48_lc10a_linked_epg_distribution.csv",
)


# ============================================================
# 7. DIRECT LPLC2 / LC4 -> descending-neuron edges
# ============================================================

threat_direct = fetch(
    """
    MATCH (s:Neuron)-[e:ConnectsTo]->(d:Neuron)
    WHERE s.type IN ['LPLC2', 'LC4']
      AND (
           d.superclass = 'descending_neuron'
        OR d.type IN ['DNa02', 'DNa03']
      )
    RETURN
        s.bodyId AS source_bodyId,
        s.type AS source_type,
        s.instance AS source_instance,
        s.somaSide AS source_side,
        d.bodyId AS target_bodyId,
        d.type AS target_type,
        d.instance AS target_instance,
        d.somaSide AS target_side,
        d.superclass AS target_superclass,
        e.weight AS weight
    ORDER BY weight DESC
    """
)

threat_direct_path = save(
    threat_direct,
    "48_threat_to_descending_direct_edges.csv",
)


# ============================================================
# 8. TWO-HOP LPLC2 / LC4 -> intermediate -> descending
#
# Group at type level to keep the query/result manageable.
#
# Structural score:
#     min(e1.weight, e2.weight)
#
# This is just a bottleneck-strength ranking heuristic.
# ============================================================

threat_two_hop = fetch(
    """
    MATCH (s:Neuron)-[e1:ConnectsTo]->(m:Neuron)
          -[e2:ConnectsTo]->(d:Neuron)
    WHERE s.type IN ['LPLC2', 'LC4']
      AND (
           d.superclass = 'descending_neuron'
        OR d.type IN ['DNa02', 'DNa03']
      )
    WITH
        s.type AS source_type,
        m.type AS intermediate_type,
        m.superclass AS intermediate_superclass,
        d.type AS target_type,
        d.superclass AS target_superclass,
        e1,
        e2,
        CASE
            WHEN e1.weight < e2.weight
            THEN e1.weight
            ELSE e2.weight
        END AS bottleneck_weight
    RETURN
        source_type,
        intermediate_type,
        intermediate_superclass,
        target_type,
        target_superclass,
        count(*) AS path_count,
        sum(bottleneck_weight) AS bottleneck_score,
        max(bottleneck_weight) AS max_bottleneck_weight,
        avg(toFloat(bottleneck_weight)) AS mean_bottleneck_weight
    ORDER BY bottleneck_score DESC, path_count DESC
    LIMIT 250
    """
)

threat_two_hop_path = save(
    threat_two_hop,
    "48_threat_to_descending_two_hop_summary.csv",
)


# ============================================================
# 9. Specifically ask whether target/threat visual channels
#    touch the already-used FlyNav navigation core directly.
# ============================================================

navigation_core_direct = fetch(
    """
    MATCH (s:Neuron)-[e:ConnectsTo]->(t:Neuron)
    WHERE s.type IN ['LC10a', 'LPLC2', 'LC4']
      AND (
           t.type = 'EPG'
        OR t.type STARTS WITH 'FC2'
        OR t.type IN ['PFL2', 'PFL3', 'DNa02', 'DNa03']
      )
    RETURN
        s.type AS source_type,
        t.type AS target_type,
        count(e) AS edge_count,
        sum(e.weight) AS total_weight,
        max(e.weight) AS max_weight,
        count(DISTINCT s) AS source_neurons,
        count(DISTINCT t) AS target_neurons
    ORDER BY total_weight DESC, edge_count DESC
    """
)

navigation_core_direct_path = save(
    navigation_core_direct,
    "48_visual_to_flynav_core_direct_summary.csv",
)


# ============================================================
# Console report
# ============================================================

print()
print("=" * 118)
print("A) LC10a TARGET CHANNEL")
print("=" * 118)
print(target_path_summary.to_string(index=False))

print()
print("Strongest LC10a -> TuBu edges:")
if lc10a_tubu.empty:
    print("NONE")
else:
    print(
        lc10a_tubu[
            [
                "source_instance",
                "source_side",
                "target_type",
                "target_instance",
                "target_side",
                "weight",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

print()
print("LC10a-linked EPG distribution:")
if epg_distribution.empty:
    print("NONE")
else:
    print(
        epg_distribution.head(40).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("B) DIRECT LPLC2 / LC4 -> DESCENDING")
print("=" * 118)

if threat_direct.empty:
    print("NO DIRECT EDGES FOUND")
else:
    direct_summary = (
        threat_direct.groupby(
            ["source_type", "target_type"],
            dropna=False,
        )
        .agg(
            total_weight=("weight", "sum"),
            edge_count=("weight", "count"),
            source_neurons=("source_bodyId", "nunique"),
            target_neurons=("target_bodyId", "nunique"),
            max_weight=("weight", "max"),
        )
        .reset_index()
        .sort_values(
            ["total_weight", "edge_count"],
            ascending=[False, False],
        )
    )
    print(direct_summary.head(50).to_string(index=False))

print()
print("=" * 118)
print("C) TWO-HOP LPLC2 / LC4 -> INTERMEDIATE -> DESCENDING")
print("=" * 118)

if threat_two_hop.empty:
    print("NO TWO-HOP PATHS FOUND")
else:
    print(
        threat_two_hop.head(60).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("D) DIRECT VISUAL -> EXISTING FLYNAV CORE")
print("=" * 118)

if navigation_core_direct.empty:
    print("NO DIRECT EDGES FOUND")
else:
    print(
        navigation_core_direct.to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("FILES CREATED")
print("=" * 118)

for path in [
    source_inventory_path,
    lc10a_tubu_path,
    lc10a_tubu_er_path,
    lc10a_er_epg_path,
    target_path_summary_path,
    epg_distribution_path,
    threat_direct_path,
    threat_two_hop_path,
    navigation_core_direct_path,
]:
    print(path)

print()
print("STEP 48 COMPLETE.")
print(
    "NEXT: choose the first camera-to-neural interface from "
    "the measured target and threat pathways, then add the "
    "Webots E-puck camera in a separate controller."
)
