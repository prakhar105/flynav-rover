import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 51 - DISCOVER FINE RETINOTOPIC VISUAL CHANNELS
#
# Motivation:
#   Step 50 showed LC10a soma side is too coarse:
#       cosine(L,R) ~ 0.984
#       total variation ~ 0.068
#
#   Therefore we now inspect finer channel structure using the
#   actual LC10a -> TuBu -> ER -> EPG connectivity.
#
# Goal:
#   Find whether TuBu subtypes / individual LC10a neurons
#   produce distinguishable EPG structural profiles that can
#   later serve as visual-bearing templates.
#
# Biological data:
#   - MaleCNS neuron IDs/types
#   - MaleCNS edge weights
#
# Engineering analysis:
#   - row-normalised structural propagation
#   - cosine distance, total variation, entropy, circular phase
#
# IMPORTANT:
#   This step still does NOT map camera LEFT/CENTER/RIGHT to
#   specific TuBu/EPG channels. That mapping, if used later,
#   will be explicitly marked as an engineered calibration.
#
# Inputs:
#   48_lc10a_to_tubu_edges.csv
#   47_tubu_to_er_edges.csv
#   47_er_to_epg_edges.csv
#
# Outputs:
#   51_lc10a_fine_channel_metrics.csv
#   51_tubu_type_epg_profiles.csv
#   51_tubu_type_metrics.csv
#   51_tubu_type_pairwise_separability.csv
#   51_tubu_candidate_channels.csv
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
            f"Required file missing:\n{path}"
        )


def row_normalise(matrix):
    matrix = matrix.astype(float, copy=True)
    sums = matrix.sum(axis=1, keepdims=True)
    mask = sums[:, 0] > 0
    matrix[mask] /= sums[mask]
    return matrix


def safe_cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def entropy_bits(profile):
    p = np.asarray(profile, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return float("nan")
    return float(-(p * np.log2(p)).sum())


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


def circular_metrics(profile):
    profile = np.asarray(profile, dtype=float)

    if profile.sum() <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    profile = profile / profile.sum()

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        len(profile),
        endpoint=False,
    )

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
        return (
            float("nan"),
            strength,
        )

    phase = math.atan2(y, x)

    if phase < 0:
        phase += 2.0 * math.pi

    phase_index = (
        phase
        / (2.0 * math.pi)
        * len(profile)
    )

    return (
        float(phase_index),
        float(strength),
    )


for path in [W1_PATH, W2_PATH, W3_PATH]:
    require(path)

w1 = pd.read_csv(W1_PATH)
w2 = pd.read_csv(W2_PATH)
w3 = pd.read_csv(W3_PATH)

print()
print("=" * 118)
print("STEP 51 - DISCOVER FINE RETINOTOPIC VISUAL CHANNELS")
print("=" * 118)
print()
print("Step 50 conclusion:")
print(
    "  LC10a soma-side grouping was nearly identical, "
    "so LEFT/RIGHT soma side is NOT sufficient."
)
print()
print("Now testing finer structure:")
print("  individual LC10a neurons")
print("  TuBu target subtypes")
print()


# ------------------------------------------------------------
# Build aligned matrices.
# ------------------------------------------------------------

lc_ids = sorted(
    int(x)
    for x in w1["source_bodyId"].dropna().unique()
)

tb_ids = sorted(
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

lc_index = {
    body_id: i
    for i, body_id in enumerate(lc_ids)
}

tb_index = {
    body_id: i
    for i, body_id in enumerate(tb_ids)
}

er_index = {
    body_id: i
    for i, body_id in enumerate(er_ids)
}

epg_index = {
    body_id: i
    for i, body_id in enumerate(epg_ids)
}

A = np.zeros(
    (len(lc_ids), len(tb_ids)),
    dtype=float,
)

B = np.zeros(
    (len(tb_ids), len(er_ids)),
    dtype=float,
)

C = np.zeros(
    (len(er_ids), len(epg_ids)),
    dtype=float,
)


for row in w1.itertuples(index=False):
    A[
        lc_index[int(row.source_bodyId)],
        tb_index[int(row.target_bodyId)],
    ] += float(row.weight)

for row in w2.itertuples(index=False):
    B[
        tb_index[int(row.source_bodyId)],
        er_index[int(row.target_bodyId)],
    ] += float(row.weight)

for row in w3.itertuples(index=False):
    C[
        er_index[int(row.source_bodyId)],
        epg_index[int(row.target_bodyId)],
    ] += float(row.weight)


A_n = row_normalise(A)
B_n = row_normalise(B)
C_n = row_normalise(C)

P_lc_epg = row_normalise(
    A_n @ B_n @ C_n
)


# ------------------------------------------------------------
# Collapse EPG neurons into 16 existing FlyNav heading labels.
# ------------------------------------------------------------

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

epg_heading = {
    body_id: heading_from_instance(
        epg_instance.get(body_id)
    )
    for body_id in epg_ids
}

heading_index = {
    heading: i
    for i, heading in enumerate(HEADING_ORDER)
}

H_lc = np.zeros(
    (len(lc_ids), len(HEADING_ORDER)),
    dtype=float,
)

for j, epg_id in enumerate(epg_ids):
    heading = epg_heading.get(epg_id)

    if heading in heading_index:
        H_lc[:, heading_index[heading]] += (
            P_lc_epg[:, j]
        )

H_lc = row_normalise(H_lc)


# ------------------------------------------------------------
# LC10a metadata and dominant TuBu target type.
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

lc_instance = dict(
    zip(
        lc_meta["source_bodyId"],
        lc_meta["source_instance"],
    )
)

lc_side = dict(
    zip(
        lc_meta["source_bodyId"],
        lc_meta["source_side"],
    )
)

dominant_tubu = (
    w1.groupby(
        [
            "source_bodyId",
            "target_type",
        ],
        dropna=False,
    )["weight"]
    .sum()
    .reset_index()
    .sort_values(
        [
            "source_bodyId",
            "weight",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop_duplicates(
        "source_bodyId"
    )
)

dominant_tubu["source_bodyId"] = (
    dominant_tubu[
        "source_bodyId"
    ].astype(int)
)

dominant_tubu_map = dict(
    zip(
        dominant_tubu["source_bodyId"],
        dominant_tubu["target_type"],
    )
)


# ------------------------------------------------------------
# Per-LC10a fine-channel metrics.
# ------------------------------------------------------------

lc_metric_rows = []

for i, lc_id in enumerate(lc_ids):
    profile = H_lc[i]

    phase, vector_strength = (
        circular_metrics(profile)
    )

    peak_idx = int(
        np.argmax(profile)
    )

    lc_metric_rows.append(
        {
            "lc10a_bodyId": lc_id,
            "lc10a_instance":
                lc_instance.get(lc_id),
            "lc10a_side":
                lc_side.get(lc_id),
            "dominant_tubu_type":
                dominant_tubu_map.get(lc_id),
            "peak_heading":
                HEADING_ORDER[peak_idx],
            "peak_value":
                float(profile[peak_idx]),
            "circular_phase_index":
                phase,
            "circular_vector_strength":
                vector_strength,
            "entropy_bits":
                entropy_bits(profile),
        }
    )

lc_metrics = pd.DataFrame(
    lc_metric_rows
).sort_values(
    [
        "circular_vector_strength",
        "peak_value",
    ],
    ascending=[
        False,
        False,
    ],
)

lc_metrics_path = (
    OUTPUT_DIR
    / "51_lc10a_fine_channel_metrics.csv"
)

lc_metrics.to_csv(
    lc_metrics_path,
    index=False,
)


# ------------------------------------------------------------
# Build TuBu type -> EPG profiles.
#
# We use LC10a anatomical input as the initial distribution
# over TuBu neurons belonging to each TuBu type, then propagate
# through TuBu -> ER -> EPG.
# ------------------------------------------------------------

tb_type_by_id = (
    w1[
        [
            "target_bodyId",
            "target_type",
        ]
    ]
    .drop_duplicates("target_bodyId")
    .copy()
)

tb_type_by_id["target_bodyId"] = (
    tb_type_by_id[
        "target_bodyId"
    ].astype(int)
)

tb_type_map = dict(
    zip(
        tb_type_by_id["target_bodyId"],
        tb_type_by_id["target_type"],
    )
)

tubu_types = sorted(
    x
    for x in tb_type_by_id[
        "target_type"
    ].dropna().unique()
)

tubu_profiles = {}
tubu_input_weight = {}
tubu_input_edges = {}
tubu_input_lc10a = {}


# Total LC10a input into each individual TuBu neuron.
tb_lc_input = (
    w1.groupby(
        "target_bodyId"
    )["weight"]
    .sum()
    .to_dict()
)

for tubu_type in tubu_types:
    start = np.zeros(
        len(tb_ids),
        dtype=float,
    )

    member_tb_ids = [
        tb_id
        for tb_id in tb_ids
        if tb_type_map.get(tb_id) == tubu_type
    ]

    for tb_id in member_tb_ids:
        start[
            tb_index[tb_id]
        ] = float(
            tb_lc_input.get(
                tb_id,
                0.0,
            )
        )

    if start.sum() <= 0:
        continue

    start /= start.sum()

    epg_profile = (
        start
        @ B_n
        @ C_n
    )

    if epg_profile.sum() > 0:
        epg_profile = (
            epg_profile
            / epg_profile.sum()
        )

    heading_profile = np.zeros(
        len(HEADING_ORDER),
        dtype=float,
    )

    for j, epg_id in enumerate(epg_ids):
        heading = epg_heading.get(epg_id)

        if heading in heading_index:
            heading_profile[
                heading_index[heading]
            ] += epg_profile[j]

    if heading_profile.sum() > 0:
        heading_profile /= (
            heading_profile.sum()
        )

    tubu_profiles[tubu_type] = (
        heading_profile
    )

    subset = w1[
        w1["target_type"]
        == tubu_type
    ]

    tubu_input_weight[tubu_type] = float(
        subset["weight"].sum()
    )

    tubu_input_edges[tubu_type] = int(
        len(subset)
    )

    tubu_input_lc10a[tubu_type] = int(
        subset["source_bodyId"].nunique()
    )


# ------------------------------------------------------------
# Export TuBu-type heading profiles + metrics.
# ------------------------------------------------------------

profile_rows = []
metric_rows = []

for tubu_type, profile in tubu_profiles.items():
    phase, vector_strength = (
        circular_metrics(profile)
    )

    peak_idx = int(
        np.argmax(profile)
    )

    for heading in HEADING_ORDER:
        profile_rows.append(
            {
                "tubu_type": tubu_type,
                "epg_heading": heading,
                "structural_influence": float(
                    profile[
                        heading_index[heading]
                    ]
                ),
            }
        )

    metric_rows.append(
        {
            "tubu_type": tubu_type,
            "lc10a_input_weight":
                tubu_input_weight[
                    tubu_type
                ],
            "lc10a_input_edges":
                tubu_input_edges[
                    tubu_type
                ],
            "lc10a_source_neurons":
                tubu_input_lc10a[
                    tubu_type
                ],
            "peak_heading":
                HEADING_ORDER[peak_idx],
            "peak_value":
                float(profile[peak_idx]),
            "circular_phase_index":
                phase,
            "circular_vector_strength":
                vector_strength,
            "entropy_bits":
                entropy_bits(profile),
        }
    )

tubu_profile_df = pd.DataFrame(
    profile_rows
)

tubu_metrics_df = pd.DataFrame(
    metric_rows
).sort_values(
    [
        "circular_vector_strength",
        "lc10a_input_weight",
    ],
    ascending=[
        False,
        False,
    ],
)

tubu_profile_path = (
    OUTPUT_DIR
    / "51_tubu_type_epg_profiles.csv"
)

tubu_metrics_path = (
    OUTPUT_DIR
    / "51_tubu_type_metrics.csv"
)

tubu_profile_df.to_csv(
    tubu_profile_path,
    index=False,
)

tubu_metrics_df.to_csv(
    tubu_metrics_path,
    index=False,
)


# ------------------------------------------------------------
# Pairwise separability among TuBu-type EPG profiles.
# ------------------------------------------------------------

pair_rows = []

for a, b in combinations(
    sorted(tubu_profiles),
    2,
):
    pa = tubu_profiles[a]
    pb = tubu_profiles[b]

    cosine = safe_cosine(
        pa,
        pb,
    )

    tv = float(
        0.5
        * np.abs(
            pa - pb
        ).sum()
    )

    l1 = float(
        np.abs(
            pa - pb
        ).sum()
    )

    phase_a, strength_a = (
        circular_metrics(pa)
    )

    phase_b, strength_b = (
        circular_metrics(pb)
    )

    if (
        math.isnan(phase_a)
        or math.isnan(phase_b)
    ):
        phase_gap = float("nan")
    else:
        raw_gap = abs(
            phase_a - phase_b
        )

        phase_gap = min(
            raw_gap,
            len(HEADING_ORDER)
            - raw_gap,
        )

    pair_rows.append(
        {
            "tubu_type_a": a,
            "tubu_type_b": b,
            "cosine_similarity":
                cosine,
            "total_variation":
                tv,
            "l1_distance":
                l1,
            "phase_gap_bins":
                phase_gap,
            "vector_strength_a":
                strength_a,
            "vector_strength_b":
                strength_b,
        }
    )

pair_df = pd.DataFrame(
    pair_rows
).sort_values(
    [
        "total_variation",
        "phase_gap_bins",
    ],
    ascending=[
        False,
        False,
    ],
)

pairwise_path = (
    OUTPUT_DIR
    / "51_tubu_type_pairwise_separability.csv"
)

pair_df.to_csv(
    pairwise_path,
    index=False,
)


# ------------------------------------------------------------
# Candidate channel ranking.
#
# No hard biological claim: this simply prefers TuBu types that
#  - receive substantial LC10a input,
#  - have some directional concentration in EPG index space,
#  - are not excessively diffuse.
#
# We rank for experiment selection only.
# ------------------------------------------------------------

candidate_df = (
    tubu_metrics_df.copy()
)

if not candidate_df.empty:
    weight_max = max(
        candidate_df[
            "lc10a_input_weight"
        ].max(),
        1.0,
    )

    candidate_df[
        "input_weight_norm"
    ] = (
        candidate_df[
            "lc10a_input_weight"
        ]
        / weight_max
    )

    candidate_df[
        "candidate_score"
    ] = (
        0.55
        * candidate_df[
            "circular_vector_strength"
        ].fillna(0.0)
        +
        0.30
        * candidate_df[
            "input_weight_norm"
        ]
        +
        0.15
        * candidate_df[
            "peak_value"
        ].fillna(0.0)
    )

    candidate_df = (
        candidate_df.sort_values(
            [
                "candidate_score",
                "lc10a_input_weight",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

candidate_path = (
    OUTPUT_DIR
    / "51_tubu_candidate_channels.csv"
)

candidate_df.to_csv(
    candidate_path,
    index=False,
)


# ------------------------------------------------------------
# Console report.
# ------------------------------------------------------------

print("=" * 118)
print("INDIVIDUAL LC10a STRUCTURAL DIVERSITY")
print("=" * 118)

print(
    f"Connected LC10a neurons analysed: "
    f"{len(lc_metrics)}"
)

if not lc_metrics.empty:
    print()
    print(
        lc_metrics.head(25).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("TuBu TYPE -> EPG CHANNEL METRICS")
print("=" * 118)

if tubu_metrics_df.empty:
    print("No TuBu subtype profiles were produced.")
else:
    print(
        tubu_metrics_df.to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("MOST DISTINCT TuBu TYPE PAIRS")
print("=" * 118)

if pair_df.empty:
    print("No pairwise comparison available.")
else:
    print(
        pair_df.head(30).to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("EXPERIMENTAL CANDIDATE CHANNELS")
print("=" * 118)

if candidate_df.empty:
    print("No candidate channels available.")
else:
    display_cols = [
        "tubu_type",
        "lc10a_input_weight",
        "lc10a_source_neurons",
        "peak_heading",
        "peak_value",
        "circular_phase_index",
        "circular_vector_strength",
        "entropy_bits",
        "candidate_score",
    ]

    print(
        candidate_df[
            display_cols
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

print()
print("=" * 118)
print("INTERPRETATION")
print("=" * 118)
print(
    "If TuBu subtypes show meaningfully different EPG "
    "profiles, Step 52 can build a 3-channel visual-bearing "
    "interface from those connectome-derived templates."
)
print(
    "If the profiles are still nearly identical, we should "
    "move one level finer and use individual LC10a/TuBu "
    "neurons rather than subtype averages."
)
print()
print(
    "Camera LEFT/CENTER/RIGHT will NOT be assigned to a "
    "biological channel in this script."
)

print()
print("=" * 118)
print("FILES CREATED")
print("=" * 118)

for path in [
    lc_metrics_path,
    tubu_profile_path,
    tubu_metrics_path,
    pairwise_path,
    candidate_path,
]:
    print(path)

print()
print("STEP 51 COMPLETE.")
