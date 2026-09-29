import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from neuprint import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "connectome" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
TOKEN = os.getenv("NEUPRINT_TOKEN")
if not TOKEN:
    raise RuntimeError("NEUPRINT_TOKEN was not found in the project .env file.")

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=TOKEN,
)

def save(df, name):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    return path

print()
print("=" * 120)
print("STEP 53 - VALIDATE LC10a -> AOTU019 -> DNa02/DNa03 VISUAL-PURSUIT BRIDGE")
print("=" * 120)

inventory = client.fetch_custom("""
MATCH (n:Neuron)
WHERE n.type = 'AOTU019'
RETURN n.bodyId AS bodyId,
       n.type AS type,
       n.instance AS instance,
       n.somaSide AS somaSide,
       n.superclass AS superclass,
       n.class AS class,
       n.subclass AS subclass,
       n.predictedNt AS predictedNt,
       n.consensusNt AS consensusNt,
       n.pre AS pre,
       n.post AS post
ORDER BY somaSide, bodyId
""")
inventory_path = save(inventory, "53_aotu019_inventory.csv")

lc_to_aotu = client.fetch_custom("""
MATCH (lc:Neuron)-[e:ConnectsTo]->(a:Neuron)
WHERE lc.type = 'LC10a'
  AND a.type = 'AOTU019'
RETURN lc.bodyId AS lc10a_bodyId,
       lc.instance AS lc10a_instance,
       lc.somaSide AS lc10a_side,
       lc.predictedNt AS lc10a_predictedNt,
       a.bodyId AS aotu_bodyId,
       a.instance AS aotu_instance,
       a.somaSide AS aotu_side,
       a.predictedNt AS aotu_predictedNt,
       e.weight AS weight
ORDER BY weight DESC
""")
lc_to_aotu_path = save(lc_to_aotu, "53_lc10a_to_aotu019_edges.csv")

aotu_to_dna = client.fetch_custom("""
MATCH (a:Neuron)-[e:ConnectsTo]->(d:Neuron)
WHERE a.type = 'AOTU019'
  AND d.type IN ['DNa02', 'DNa03']
RETURN a.bodyId AS aotu_bodyId,
       a.instance AS aotu_instance,
       a.somaSide AS aotu_side,
       a.predictedNt AS aotu_predictedNt,
       d.bodyId AS dna_bodyId,
       d.type AS dna_type,
       d.instance AS dna_instance,
       d.somaSide AS dna_side,
       d.predictedNt AS dna_predictedNt,
       e.weight AS weight
ORDER BY weight DESC
""")
aotu_to_dna_path = save(aotu_to_dna, "53_aotu019_to_dna_edges.csv")

paths = client.fetch_custom("""
MATCH (lc:Neuron)-[e1:ConnectsTo]->(a:Neuron)
      -[e2:ConnectsTo]->(d:Neuron)
WHERE lc.type = 'LC10a'
  AND a.type = 'AOTU019'
  AND d.type IN ['DNa02', 'DNa03']
WITH lc, a, d, e1, e2,
     CASE WHEN e1.weight < e2.weight THEN e1.weight ELSE e2.weight END AS bottleneck_weight
RETURN lc.bodyId AS lc10a_bodyId,
       lc.instance AS lc10a_instance,
       lc.somaSide AS lc10a_side,
       a.bodyId AS aotu_bodyId,
       a.instance AS aotu_instance,
       a.somaSide AS aotu_side,
       d.bodyId AS dna_bodyId,
       d.type AS dna_type,
       d.instance AS dna_instance,
       d.somaSide AS dna_side,
       e1.weight AS lc10a_to_aotu_weight,
       e2.weight AS aotu_to_dna_weight,
       bottleneck_weight
ORDER BY bottleneck_weight DESC
""")
paths_path = save(paths, "53_lc10a_aotu019_dna_paths.csv")

if paths.empty:
    laterality = pd.DataFrame()
else:
    laterality = (
        paths.groupby(
            ["lc10a_side", "aotu_side", "dna_type", "dna_side"],
            dropna=False,
        )
        .agg(
            path_count=("bottleneck_weight", "count"),
            bottleneck_score=("bottleneck_weight", "sum"),
            max_bottleneck=("bottleneck_weight", "max"),
            mean_bottleneck=("bottleneck_weight", "mean"),
            lc10a_neurons=("lc10a_bodyId", "nunique"),
            aotu_neurons=("aotu_bodyId", "nunique"),
            dna_neurons=("dna_bodyId", "nunique"),
        )
        .reset_index()
        .sort_values(["bottleneck_score", "path_count"], ascending=[False, False])
    )

laterality_path = save(laterality, "53_lc10a_aotu019_laterality_summary.csv")

if aotu_to_dna.empty:
    dna_summary = pd.DataFrame()
else:
    dna_summary = (
        aotu_to_dna.groupby(
            ["aotu_side", "dna_type", "dna_side"],
            dropna=False,
        )
        .agg(
            edge_count=("weight", "count"),
            total_weight=("weight", "sum"),
            max_weight=("weight", "max"),
            mean_weight=("weight", "mean"),
            aotu_neurons=("aotu_bodyId", "nunique"),
            dna_neurons=("dna_bodyId", "nunique"),
        )
        .reset_index()
        .sort_values(["total_weight", "edge_count"], ascending=[False, False])
    )

dna_summary_path = save(dna_summary, "53_aotu019_dna_type_summary.csv")

print()
print("AOTU019 INVENTORY")
print(inventory.to_string(index=False) if not inventory.empty else "NONE")

print()
print("LC10a -> AOTU019")
if lc_to_aotu.empty:
    print("NONE")
else:
    print(f"Edges: {len(lc_to_aotu):,} | total weight: {int(lc_to_aotu['weight'].sum()):,}")
    print(lc_to_aotu.head(40).to_string(index=False))

print()
print("AOTU019 -> DNa02/DNa03")
if aotu_to_dna.empty:
    print("NONE")
else:
    print(f"Edges: {len(aotu_to_dna):,} | total weight: {int(aotu_to_dna['weight'].sum()):,}")
    print(dna_summary.to_string(index=False))

print()
print("LC10a -> AOTU019 -> DNa02/DNa03 LATERALITY")
print(laterality.to_string(index=False) if not laterality.empty else "NONE")

print()
print("STRONGEST TWO-HOP PATHS")
if paths.empty:
    print("NONE")
else:
    cols = [
        "lc10a_instance","lc10a_side","aotu_instance","aotu_side",
        "dna_type","dna_instance","dna_side",
        "lc10a_to_aotu_weight","aotu_to_dna_weight","bottleneck_weight",
    ]
    print(paths[cols].head(60).to_string(index=False))

print()
print("FILES CREATED")
for p in [
    inventory_path, lc_to_aotu_path, aotu_to_dna_path,
    paths_path, laterality_path, dna_summary_path
]:
    print(p)

print()
print("STEP 53 COMPLETE.")
