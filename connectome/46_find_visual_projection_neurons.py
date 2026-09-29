import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_neurons


# ============================================================
# STEP 46 - MALECNS VISUAL PROJECTION NEURON DISCOVERY
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

print()
print("=" * 110)
print("STEP 46 - MALECNS VISUAL PROJECTION NEURON DISCOVERY")
print("=" * 110)
print("Dataset: male-cns:v1.0")
print("Selection rule: superclass == visual_projection")
print()

# 1) Verify live superclass taxonomy.
superclass_query = """
MATCH (n:Neuron)
WHERE n.superclass IS NOT NULL
RETURN n.superclass AS superclass,
       count(*) AS neuron_count
ORDER BY neuron_count DESC
"""

superclass_df = client.fetch_custom(superclass_query)
superclass_path = OUTPUT_DIR / "46_superclass_counts.csv"
superclass_df.to_csv(superclass_path, index=False)

print("Superclass inventory:")
print(superclass_df.to_string(index=False))
print()

# 2) Fetch all visual projection neurons.
visual_df, visual_roi_df = fetch_neurons(
    NC(superclass="visual_projection"),
    client=client,
)

if visual_df.empty:
    raise RuntimeError(
        "No neurons matched superclass='visual_projection'."
    )

print(f"Visual projection neurons found: {len(visual_df):,}")

preferred_columns = [
    "bodyId",
    "type",
    "instance",
    "superclass",
    "class",
    "subclass",
    "systematicType",
    "description",
    "somaSide",
    "rootSide",
    "predictedNt",
    "consensusNt",
    "pre",
    "post",
    "inputRois",
    "outputRois",
    "hemibrainType",
    "flywireType",
]

available_columns = [
    c for c in preferred_columns if c in visual_df.columns
]

visual_export = visual_df[available_columns].copy()
sort_cols = [
    c for c in ["type", "somaSide", "bodyId"]
    if c in visual_export.columns
]
if sort_cols:
    visual_export = visual_export.sort_values(
        by=sort_cols,
        na_position="last",
    )

visual_path = OUTPUT_DIR / "46_visual_projection_neurons.csv"
visual_export.to_csv(visual_path, index=False)

# 3) Type summary.
working = visual_df.copy()
working["type_clean"] = working["type"].fillna("<untyped>").astype(str)

if "somaSide" in working.columns:
    working["somaSide_clean"] = (
        working["somaSide"].fillna("<unknown>").astype(str)
    )
else:
    working["somaSide_clean"] = "<unavailable>"

if "predictedNt" in working.columns:
    working["predictedNt_clean"] = (
        working["predictedNt"].fillna("<unknown>").astype(str)
    )
else:
    working["predictedNt_clean"] = "<unavailable>"

agg = {
    "neuron_count": ("bodyId", "count"),
    "left_count": (
        "somaSide_clean",
        lambda s: int(sum(str(x).lower().startswith("l") for x in s)),
    ),
    "right_count": (
        "somaSide_clean",
        lambda s: int(sum(str(x).lower().startswith("r") for x in s)),
    ),
    "predicted_nt": (
        "predictedNt_clean",
        lambda s: "|".join(sorted(set(str(x) for x in s))),
    ),
}

if "pre" in working.columns:
    agg["total_pre"] = ("pre", "sum")
if "post" in working.columns:
    agg["total_post"] = ("post", "sum")

type_summary = (
    working.groupby("type_clean", dropna=False)
    .agg(**agg)
    .reset_index()
    .rename(columns={"type_clean": "type"})
    .sort_values(
        by=["neuron_count", "type"],
        ascending=[False, True],
    )
)

type_path = OUTPUT_DIR / "46_visual_projection_type_summary.csv"
type_summary.to_csv(type_path, index=False)

# 4) Coarse prefix/family summary.
def type_family(type_name):
    if not type_name or type_name == "<untyped>":
        return "<untyped>"
    match = re.match(r"^([A-Za-z]+)", str(type_name))
    return match.group(1) if match else "<other>"

working["type_family"] = working["type_clean"].map(type_family)

family_summary = (
    working.groupby("type_family")
    .agg(
        neuron_count=("bodyId", "count"),
        type_count=("type_clean", "nunique"),
    )
    .reset_index()
    .sort_values(
        by=["neuron_count", "type_family"],
        ascending=[False, True],
    )
)

family_path = OUTPUT_DIR / "46_visual_projection_family_summary.csv"
family_summary.to_csv(family_path, index=False)

# 5) Side summary.
side_summary = (
    working.groupby("somaSide_clean")
    .agg(
        neuron_count=("bodyId", "count"),
        type_count=("type_clean", "nunique"),
    )
    .reset_index()
    .rename(columns={"somaSide_clean": "somaSide"})
    .sort_values(by="neuron_count", ascending=False)
)

side_path = OUTPUT_DIR / "46_visual_projection_side_summary.csv"
side_summary.to_csv(side_path, index=False)

print()
print("=" * 110)
print("TOP VISUAL PROJECTION TYPES")
print("=" * 110)
print(type_summary.head(40).to_string(index=False))

print()
print("=" * 110)
print("TYPE-PREFIX / FAMILY SUMMARY")
print("=" * 110)
print(family_summary.head(40).to_string(index=False))

print()
print("=" * 110)
print("SIDE SUMMARY")
print("=" * 110)
print(side_summary.to_string(index=False))

print()
print("=" * 110)
print("FILES CREATED")
print("=" * 110)

for path in [
    superclass_path,
    visual_path,
    type_path,
    family_path,
    side_path,
]:
    print(path)

print()
print("STEP 46 COMPLETE.")
print(
    "NEXT: trace selected visual-projection neurons toward "
    "the existing FlyNav navigation network using measured "
    "MaleCNS connectivity."
)

if __name__ == "__main__":
    pass
