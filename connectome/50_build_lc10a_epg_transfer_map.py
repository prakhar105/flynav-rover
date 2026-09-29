import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 50 - LC10a -> TuBu -> ER -> EPG STRUCTURAL TRANSFER MAP
#
# Purpose:
#   Use the measured MaleCNS edge tables from Steps 47/48 to
#   test whether LEFT- and RIGHT-side LC10a populations produce
#   distinguishable structural influence profiles across the
#   existing 16 EPG heading labels.
#
# Why this comes before motor control:
#   We do NOT want to invent:
#
#       camera-left -> turn-left
#
#   without first checking what the measured visual pathway
#   actually does to the EPG population.
#
# Biological data:
#   - LC10a, TuBu, ER, EPG neuron IDs
#   - MaleCNS anatomical edge weights
#
# Engineering analysis:
#   - Row-normalised matrix propagation:
#
#       LC10a -> TuBu -> ER -> EPG
#
#   - The resulting values are STRUCTURAL influence proxies,
#     not firing rates and not biological gains.
#
# Inputs:
#   connectome/output/48_lc10a_to_tubu_edges.csv
#   connectome/output/47_tubu_to_er_edges.csv
#   connectome/output/47_er_to_epg_edges.csv
#
# Outputs:
#   50_lc10a_neuron_epg_profile.csv
#   50_lc10a_neuron_heading_profile.csv
#   50_lc10a_side_heading_profile.csv
#   50_lc10a_side_separability.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "connectome" / "output"

W1_PATH = OUTPUT_DIR / "48_lc10a_to_tubu_edges.csv"
W2_PATH = OUTPUT_DIR / "47_tubu_to_er_edges.csv"
W3_PATH = OUTPUT_DIR / "47_er_to_epg_edges.csv"

HEADING_ORDER = [
    "L1", "L2", "L3", "L4",
    "L5", "L6", "L7", "L8",
    "R8", "R7", "R6", "R5",
    "R4", "R3", "R2", "R1",
]


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step 47/48 output is missing:\n{path}"
        )


def row_normalise(matrix):
    matrix = matrix.astype(float, copy=True)
    sums = matrix.sum(axis=1, keepdims=True)
    nonzero = sums[:, 0] > 0
    matrix[nonzero] /= sums[nonzero]
    return matrix


def safe_cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def heading_from_instance(instance):
    if pd.isna(instance):
        return None

    match = re.search(
        r"_(L[1-8]|R[1-8])$",
        str(instance),
    )

    if match:
        return match.group(1)

    return None


for required in [W1_PATH, W2_PATH, W3_PATH]:
    require(required)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

w1 = pd.read_csv(W1_PATH)
w2 = pd.read_csv(W2_PATH)
w3 = pd.read_csv(W3_PATH)

print()
print("=" * 116)
print("STEP 50 - LC10a -> TuBu -> ER -> EPG STRUCTURAL TRANSFER MAP")
print("=" * 116)
print()
print("Inputs:")
print(f"  {W1_PATH.name}: {len(w1):,} edges")
print(f"  {W2_PATH.name}: {len(w2):,} edges")
print(f"  {W3_PATH.name}: {len(w3):,} edges")
print()
print(
    "Analysis = row-normalised anatomical weight propagation. "
    "These scores are structural proxies, NOT neural firing rates."
)
print()


# ------------------------------------------------------------
# Build aligned neuron sets.
# ------------------------------------------------------------

lc_ids = sorted(
    int(x)
    for x in w1["source_bodyId"].dropna().unique()
)

tubu_ids = sorted(
    set(
        int(x)
        for x in w1["target_bodyId"].dropna().unique()
    )
    | set(
        int(x)
        for x in w2["source_bodyId"].dropna().unique()
    )
)

er_ids = sorted(
    set(
        int(x)
        for x in w2["target_bodyId"].dropna().unique()
    )
    | set(
        int(x)
        for x in w3["source_bodyId"].dropna().unique()
    )
)

epg_ids = sorted(
    int(x)
    for x in w3["target_bodyId"].dropna().unique()
)

lc_index = {body_id: i for i, body_id in enumerate(lc_ids)}
tb_index = {body_id: i for i, body_id in enumerate(tubu_ids)}
er_index = {body_id: i for i, body_id in enumerate(er_ids)}
epg_index = {body_id: i for i, body_id in enumerate(epg_ids)}

A = np.zeros(
    (len(lc_ids), len(tubu_ids)),
    dtype=float,
)

B = np.zeros(
    (len(tubu_ids), len(er_ids)),
    dtype=float,
)

C = np.zeros(
    (len(er_ids), len(epg_ids)),
    dtype=float,
)


for row in w1.itertuples(index=False):
    s = int(row.source_bodyId)
    t = int(row.target_bodyId)
    A[lc_index[s], tb_index[t]] += float(row.weight)

for row in w2.itertuples(index=False):
    s = int(row.source_bodyId)
    t = int(row.target_bodyId)
    B[tb_index[s], er_index[t]] += float(row.weight)

for row in w3.itertuples(index=False):
    s = int(row.source_bodyId)
    t = int(row.target_bodyId)
    C[er_index[s], epg_index[t]] += float(row.weight)


# ------------------------------------------------------------
# Structural propagation.
#
# Row-normalising each stage prevents large absolute synapse
# totals from trivially dominating the comparison.
# ------------------------------------------------------------

A_n = row_normalise(A)
B_n = row_normalise(B)
C_n = row_normalise(C)

P = A_n @ B_n @ C_n

# Re-normalise final profile per LC10a neuron.
P = row_normalise(P)


# ------------------------------------------------------------
# Biological metadata maps.
# ------------------------------------------------------------

lc_meta = (
    w1[
        [
            "source_bodyId",
            "source_instance",
            "source_side",
        ]
    ]
    .drop_duplicates("source_bodyId")
    .copy()
)

lc_meta["source_bodyId"] = (
    lc_meta["source_bodyId"].astype(int)
)

lc_side = dict(
    zip(
        lc_meta["source_bodyId"],
        lc_meta["source_side"],
    )
)

lc_instance = dict(
    zip(
        lc_meta["source_bodyId"],
        lc_meta["source_instance"],
    )
)

epg_meta = (
    w3[
        [
            "target_bodyId",
            "target_instance",
            "target_side",
        ]
    ]
    .drop_duplicates("target_bodyId")
    .copy()
)

epg_meta["target_bodyId"] = (
    epg_meta["target_bodyId"].astype(int)
)

epg_instance = dict(
    zip(
        epg_meta["target_bodyId"],
        epg_meta["target_instance"],
    )
)

epg_side = dict(
    zip(
        epg_meta["target_bodyId"],
        epg_meta["target_side"],
    )
)

epg_heading = {
    body_id: heading_from_instance(
        epg_instance.get(body_id)
    )
    for body_id in epg_ids
}


# ------------------------------------------------------------
# Export LC10a-neuron -> individual EPG profile.
# ------------------------------------------------------------

neuron_epg_rows = []

for i, lc_id in enumerate(lc_ids):
    for j, epg_id in enumerate(epg_ids):
        neuron_epg_rows.append(
            {
                "lc10a_bodyId": lc_id,
                "lc10a_instance": lc_instance.get(lc_id),
                "lc10a_side": lc_side.get(lc_id),
                "epg_bodyId": epg_id,
                "epg_instance": epg_instance.get(epg_id),
                "epg_side": epg_side.get(epg_id),
                "epg_heading": epg_heading.get(epg_id),
                "structural_influence": float(P[i, j]),
            }
        )

neuron_epg_df = pd.DataFrame(neuron_epg_rows)

neuron_epg_path = (
    OUTPUT_DIR
    / "50_lc10a_neuron_epg_profile.csv"
)

neuron_epg_df.to_csv(
    neuron_epg_path,
    index=False,
)


# ------------------------------------------------------------
# Collapse 46 EPG neurons into the 16 heading labels already
# used by the frozen FlyNav navigation system.
# ------------------------------------------------------------

heading_index = {
    heading: idx
    for idx, heading in enumerate(HEADING_ORDER)
}

H = np.zeros(
    (len(lc_ids), len(HEADING_ORDER)),
    dtype=float,
)

for j, epg_id in enumerate(epg_ids):
    heading = epg_heading.get(epg_id)

    if heading not in heading_index:
        continue

    H[:, heading_index[heading]] += P[:, j]

H = row_normalise(H)


neuron_heading_rows = []

for i, lc_id in enumerate(lc_ids):
    for heading in HEADING_ORDER:
        neuron_heading_rows.append(
            {
                "lc10a_bodyId": lc_id,
                "lc10a_instance": lc_instance.get(lc_id),
                "lc10a_side": lc_side.get(lc_id),
                "epg_heading": heading,
                "structural_influence": float(
                    H[i, heading_index[heading]]
                ),
            }
        )

neuron_heading_df = pd.DataFrame(
    neuron_heading_rows
)

neuron_heading_path = (
    OUTPUT_DIR
    / "50_lc10a_neuron_heading_profile.csv"
)

neuron_heading_df.to_csv(
    neuron_heading_path,
    index=False,
)


# ------------------------------------------------------------
# Aggregate LC10a source populations by anatomical soma side.
# ------------------------------------------------------------

side_profiles = {}

for side in ["L", "R"]:
    rows = [
        i
        for i, lc_id in enumerate(lc_ids)
        if str(lc_side.get(lc_id)).upper() == side
    ]

    if not rows:
        side_profiles[side] = np.zeros(
            len(HEADING_ORDER),
            dtype=float,
        )
        continue

    profile = H[rows].mean(axis=0)

    total = profile.sum()

    if total > 0:
        profile = profile / total

    side_profiles[side] = profile


side_rows = []

for heading in HEADING_ORDER:
    idx = heading_index[heading]

    side_rows.append(
        {
            "epg_heading": heading,
            "LC10a_L_profile": float(
                side_profiles["L"][idx]
            ),
            "LC10a_R_profile": float(
                side_profiles["R"][idx]
            ),
            "L_minus_R": float(
                side_profiles["L"][idx]
                - side_profiles["R"][idx]
            ),
        }
    )

side_df = pd.DataFrame(side_rows)

side_profile_path = (
    OUTPUT_DIR
    / "50_lc10a_side_heading_profile.csv"
)

side_df.to_csv(
    side_profile_path,
    index=False,
)


# ------------------------------------------------------------
# Separability metrics.
#
# total_variation:
#   0 = identical distributions
#   1 = non-overlapping distributions
#
# cosine_similarity:
#   1 = same direction/profile
#
# These are analysis metrics only.
# ------------------------------------------------------------

left_profile = side_profiles["L"]
right_profile = side_profiles["R"]

cosine = safe_cosine(
    left_profile,
    right_profile,
)

total_variation = float(
    0.5
    * np.abs(
        left_profile - right_profile
    ).sum()
)

l1_distance = float(
    np.abs(
        left_profile - right_profile
    ).sum()
)

left_peak = (
    HEADING_ORDER[
        int(np.argmax(left_profile))
    ]
    if left_profile.sum() > 0
    else None
)

right_peak = (
    HEADING_ORDER[
        int(np.argmax(right_profile))
    ]
    if right_profile.sum() > 0
    else None
)

# Circular phase on the 16-label EPG ring.
# This is an index-space phase, NOT world-angle degrees.
angles = np.linspace(
    0.0,
    2.0 * math.pi,
    len(HEADING_ORDER),
    endpoint=False,
)


def circular_phase(profile):
    x = float(
        np.sum(
            profile * np.cos(angles)
        )
    )

    y = float(
        np.sum(
            profile * np.sin(angles)
        )
    )

    strength = math.hypot(x, y)

    if strength <= 1e-12:
        return float("nan"), strength

    phase = math.atan2(y, x)

    if phase < 0:
        phase += 2.0 * math.pi

    phase_index = (
        phase
        / (2.0 * math.pi)
        * len(HEADING_ORDER)
    )

    return phase_index, strength


left_phase, left_vector_strength = circular_phase(
    left_profile
)

right_phase, right_vector_strength = circular_phase(
    right_profile
)


separability_df = pd.DataFrame(
    [
        {
            "connected_lc10a_neurons": len(lc_ids),
            "connected_lc10a_L": int(
                sum(
                    str(lc_side.get(x)).upper() == "L"
                    for x in lc_ids
                )
            ),
            "connected_lc10a_R": int(
                sum(
                    str(lc_side.get(x)).upper() == "R"
                    for x in lc_ids
                )
            ),
            "epg_neurons": len(epg_ids),
            "cosine_similarity_L_vs_R": cosine,
            "total_variation_L_vs_R": total_variation,
            "l1_distance_L_vs_R": l1_distance,
            "left_peak_heading": left_peak,
            "right_peak_heading": right_peak,
            "left_circular_phase_index": left_phase,
            "right_circular_phase_index": right_phase,
            "left_circular_vector_strength": left_vector_strength,
            "right_circular_vector_strength": right_vector_strength,
        }
    ]
)

separability_path = (
    OUTPUT_DIR
    / "50_lc10a_side_separability.csv"
)

separability_df.to_csv(
    separability_path,
    index=False,
)


# ------------------------------------------------------------
# Console report.
# ------------------------------------------------------------

print("=" * 116)
print("NETWORK DIMENSIONS")
print("=" * 116)
print(
    f"Connected LC10a neurons: {len(lc_ids)}"
)
print(
    f"TuBu neurons in aligned matrix: {len(tubu_ids)}"
)
print(
    f"ER neurons in aligned matrix: {len(er_ids)}"
)
print(
    f"EPG neurons: {len(epg_ids)}"
)

print()
print("=" * 116)
print("LC10a SIDE -> EPG HEADING PROFILE")
print("=" * 116)

display_df = side_df.copy()

for col in [
    "LC10a_L_profile",
    "LC10a_R_profile",
    "L_minus_R",
]:
    display_df[col] = display_df[col].map(
        lambda x: f"{x:.6f}"
    )

print(display_df.to_string(index=False))

print()
print("=" * 116)
print("LEFT / RIGHT STRUCTURAL SEPARABILITY")
print("=" * 116)
print(
    f"Cosine similarity: "
    f"{cosine:.6f} "
    f"(1.0 = identical profile)"
)
print(
    f"Total variation:   "
    f"{total_variation:.6f} "
    f"(0.0 = identical, 1.0 = non-overlap)"
)
print(
    f"L1 distance:       "
    f"{l1_distance:.6f}"
)
print(
    f"Left peak heading:  {left_peak}"
)
print(
    f"Right peak heading: {right_peak}"
)
print(
    f"Left phase index:   "
    f"{left_phase:.3f} / 16 "
    f"(vector strength={left_vector_strength:.4f})"
)
print(
    f"Right phase index:  "
    f"{right_phase:.3f} / 16 "
    f"(vector strength={right_vector_strength:.4f})"
)

print()
print("=" * 116)
print("INTERPRETATION RULE")
print("=" * 116)
print(
    "Do NOT map camera LEFT/RIGHT directly to steering yet."
)
print(
    "First inspect whether LC10a_L and LC10a_R produce "
    "meaningfully different EPG structural profiles."
)
print(
    "If they are nearly identical, Step 51 must use finer "
    "visual/retinotopic channels (for example TuBu subtype "
    "or individual LC10a connectivity), not merely soma side."
)

print()
print("=" * 116)
print("FILES CREATED")
print("=" * 116)

for path in [
    neuron_epg_path,
    neuron_heading_path,
    side_profile_path,
    separability_path,
]:
    print(path)

print()
print("STEP 50 COMPLETE.")
