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
    / "steering_core_edges.csv"
)

STEERING_MANIFOLD_FILE = (
    OUTPUT_DIR
    / "fc2_goal_manifold.csv"
)

PROJECTION_MATRIX_FILE = (
    OUTPUT_DIR
    / "fc2_projection_matrix.csv"
)

MANIFOLD_FILE = (
    OUTPUT_DIR
    / "fc2_projection_manifold.csv"
)

RECONSTRUCTION_FILE = (
    OUTPUT_DIR
    / "fc2_projection_reconstruction.csv"
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
# Identify FC2 neurons
# =========================================================

fc2 = neurons[
    neurons[
        "type"
    ].isin(
        [
            "FC2A",
            "FC2B",
            "FC2C",
        ]
    )
].copy()


fc2[
    "fc2_column"
] = fc2[
    "instance"
].map(
    parse_fc2_column
)


fc2 = fc2[
    fc2[
        "fc2_column"
    ].notna()
].copy()


fc2[
    "fc2_column"
] = fc2[
    "fc2_column"
].astype(int)


if len(fc2) != 92:

    raise RuntimeError(
        f"Expected 92 FC2 neurons, "
        f"got {len(fc2)}."
    )


print("=" * 120)
print("FC2 -> PFL ANATOMICAL PROJECTION MANIFOLD")
print("=" * 120)

print()

print(
    f"FC2 neurons: "
    f"{len(fc2)}"
)


print()

print(
    "FC2 neurons by column:"
)

print(
    fc2[
        "fc2_column"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# PFL target neurons
# =========================================================

pfl = neurons[
    neurons[
        "type"
    ].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
].copy()


pfl = (
    pfl
    .sort_values(
        [
            "type",
            "somaSide",
            "bodyId",
        ]
    )
    .reset_index(
        drop=True
    )
)


print()

print(
    f"PFL target neurons: "
    f"{len(pfl)}"
)


print(
    f"PFL2: "
    f"{int((pfl['type'] == 'PFL2').sum())}"
)


print(
    f"PFL3: "
    f"{int((pfl['type'] == 'PFL3').sum())}"
)


# =========================================================
# Source lookup
# =========================================================

source_to_column = {
    int(row["bodyId"]):
        int(row["fc2_column"])

    for _, row
    in fc2.iterrows()
}


pfl_ids = set(
    pfl[
        "bodyId"
    ].astype(int)
)


# =========================================================
# FC2 -> PFL anatomical connections
# =========================================================

projection_edges = edges[
    edges[
        "source_bodyId"
    ].isin(
        source_to_column.keys()
    )
    &
    edges[
        "target_bodyId"
    ].isin(
        pfl_ids
    )
].copy()


projection_edges[
    "fc2_column"
] = projection_edges[
    "source_bodyId"
].map(
    source_to_column
)


print()

print(
    f"FC2 -> PFL edges: "
    f"{len(projection_edges)}"
)

print(
    f"Total raw FC2 -> PFL weight: "
    f"{projection_edges['weight'].sum():.0f}"
)


# =========================================================
# Build anatomical projection matrix
#
# Rows:
#     FC2 C1..C9
#
# Columns:
#     individual PFL2/PFL3 neurons
#
# Cell:
#     total raw connectome weight from every FC2 neuron
#     in that C-column to that particular PFL neuron.
# =========================================================

target_order = (
    pfl[
        "bodyId"
    ]
    .astype(int)
    .tolist()
)


projection = (
    projection_edges
    .groupby(
        [
            "fc2_column",
            "target_bodyId",
        ]
    )[
        "weight"
    ]
    .sum()
    .unstack(
        fill_value=0.0
    )
)


projection = (
    projection
    .reindex(
        index=range(
            1,
            10,
        ),
        fill_value=0.0,
    )
    .reindex(
        columns=target_order,
        fill_value=0.0,
    )
)


# =========================================================
# Human-readable column names
# =========================================================

target_name_lookup = {}


for _, row in pfl.iterrows():

    body_id = int(
        row[
            "bodyId"
        ]
    )

    neuron_type = str(
        row[
            "type"
        ]
    )

    side = str(
        row.get(
            "somaSide",
            "",
        )
    )


    target_name_lookup[
        body_id
    ] = (
        f"{neuron_type}_"
        f"{side}_"
        f"{body_id}"
    )


projection_named = (
    projection.copy()
)


projection_named.columns = [
    target_name_lookup[
        int(column)
    ]
    for column
    in projection_named.columns
]


projection_named.index = [
    f"C{x}"
    for x
    in projection_named.index
]


projection_named.to_csv(
    PROJECTION_MATRIX_FILE
)


# =========================================================
# Column totals
# =========================================================

raw_matrix = (
    projection
    .to_numpy(
        dtype=float
    )
)


raw_totals = (
    raw_matrix.sum(
        axis=1
    )
)


nonzero_targets = (
    (
        raw_matrix
        > 0
    )
    .sum(
        axis=1
    )
)


print()
print("=" * 120)
print("FC2 COLUMN PROJECTION TOTALS")
print("=" * 120)
print()


for goal in range(
    1,
    10,
):

    index = (
        goal
        - 1
    )

    print(
        f"C{goal}: "
        f"weight={raw_totals[index]:.0f} "
        f"| nonzero PFL targets="
        f"{nonzero_targets[index]}"
    )


# =========================================================
# L1 normalization
#
# Raw magnitude varies partly because each C-column
# contains a different number of FC2 neurons.
#
# For manifold geometry we therefore analyse:
#
# 1. RAW projections
# 2. L1-normalized projection SHAPE
#
# The normalized version preserves relative anatomical
# targeting while reducing population-size effects.
# =========================================================

safe_totals = np.where(
    raw_totals > 0,
    raw_totals,
    1.0,
)


normalized_matrix = (
    raw_matrix
    /
    safe_totals[
        :,
        None,
    ]
)


# =========================================================
# PCA/SVD analysis utility
# =========================================================

def analyse_matrix(
    name,
    data,
):

    mean_vector = (
        data.mean(
            axis=0,
            keepdims=True,
        )
    )


    centered = (
        data
        -
        mean_vector
    )


    (
        U,
        singular_values,
        Vt,
    ) = np.linalg.svd(
        centered,
        full_matrices=False,
    )


    variance = (
        singular_values
        ** 2
    )


    total_variance = float(
        variance.sum()
    )


    if total_variance > 0:

        explained = (
            variance
            /
            total_variance
        )

    else:

        explained = np.zeros_like(
            variance
        )


    cumulative = np.cumsum(
        explained
    )


    scores = (
        U
        *
        singular_values
    )


    pc1 = (
        scores[
            :,
            0,
        ]
        if scores.shape[1] >= 1
        else np.zeros(
            len(data)
        )
    )


    pc2 = (
        scores[
            :,
            1,
        ]
        if scores.shape[1] >= 2
        else np.zeros(
            len(data)
        )
    )


    radius = np.sqrt(
        pc1 ** 2
        +
        pc2 ** 2
    )


    angle = (
        np.degrees(
            np.arctan2(
                pc2,
                pc1,
            )
        )
        % 360.0
    )


    # -----------------------------------------------------
    # 2D reconstruction
    # -----------------------------------------------------

    k = min(
        2,
        len(
            singular_values
        ),
    )


    centered_2d = (
        U[
            :,
            :k
        ]
        @
        np.diag(
            singular_values[
                :k
            ]
        )
        @
        Vt[
            :k,
            :
        ]
    )


    reconstructed = (
        centered_2d
        +
        mean_vector
    )


    # -----------------------------------------------------
    # Overall reconstruction
    # -----------------------------------------------------

    residual = (
        data
        -
        reconstructed
    )


    ss_res = float(
        np.sum(
            residual ** 2
        )
    )


    ss_tot = float(
        np.sum(
            (
                data
                -
                data.mean()
            )
            ** 2
        )
    )


    full_r2 = (
        1.0
        -
        ss_res
        /
        ss_tot

        if ss_tot > 0

        else 0.0
    )


    mae = float(
        np.mean(
            np.abs(
                residual
            )
        )
    )


    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )


    # -----------------------------------------------------
    # Per-column reconstruction
    # -----------------------------------------------------

    per_goal_rows = []


    for goal_index in range(
        len(data)
    ):

        truth = (
            data[
                goal_index
            ]
        )


        prediction = (
            reconstructed[
                goal_index
            ]
        )


        row_residual = (
            truth
            -
            prediction
        )


        correlation = 0.0


        if (
            np.std(
                truth
            ) > 1e-12

            and

            np.std(
                prediction
            ) > 1e-12
        ):

            correlation = float(
                np.corrcoef(
                    truth,
                    prediction,
                )[0, 1]
            )


        denominator = (
            np.linalg.norm(
                truth
            )
            *
            np.linalg.norm(
                prediction
            )
        )


        cosine = (
            float(
                np.dot(
                    truth,
                    prediction,
                )
                /
                denominator
            )

            if denominator > 1e-12

            else 0.0
        )


        per_goal_rows.append(
            {
                "analysis":
                    name,

                "goal_column":
                    goal_index
                    + 1,

                "correlation":
                    correlation,

                "cosine_similarity":
                    cosine,

                "MAE":
                    float(
                        np.mean(
                            np.abs(
                                row_residual
                            )
                        )
                    ),

                "RMSE":
                    float(
                        np.sqrt(
                            np.mean(
                                row_residual
                                ** 2
                            )
                        )
                    ),
            }
        )


    per_goal = pd.DataFrame(
        per_goal_rows
    )


    # -----------------------------------------------------
    # Angular coverage
    # -----------------------------------------------------

    sorted_angles = np.sort(
        angle
    )


    circular_angles = np.concatenate(
        [
            sorted_angles,
            [
                sorted_angles[0]
                + 360.0
            ],
        ]
    )


    gaps = np.diff(
        circular_angles
    )


    largest_gap = float(
        gaps.max()
    )


    covered_arc = (
        360.0
        -
        largest_gap
    )


    radius_mean = float(
        radius.mean()
    )


    radius_cv = (
        float(
            radius.std()
            /
            radius_mean
        )

        if radius_mean > 1e-12

        else np.inf
    )


    return {
        "name":
            name,

        "explained":
            explained,

        "cumulative":
            cumulative,

        "scores":
            scores,

        "pc1":
            pc1,

        "pc2":
            pc2,

        "radius":
            radius,

        "angle":
            angle,

        "reconstructed":
            reconstructed,

        "full_r2":
            full_r2,

        "mae":
            mae,

        "rmse":
            rmse,

        "per_goal":
            per_goal,

        "covered_arc":
            covered_arc,

        "largest_gap":
            largest_gap,

        "radius_cv":
            radius_cv,
    }


# =========================================================
# Analyse raw + normalized anatomy
# =========================================================

raw_result = analyse_matrix(
    "raw",
    raw_matrix,
)


norm_result = analyse_matrix(
    "l1_normalized",
    normalized_matrix,
)


# =========================================================
# Print explained variance
# =========================================================

for result in [
    raw_result,
    norm_result,
]:

    print()
    print("=" * 130)
    print(
        f"{result['name'].upper()} "
        f"ANATOMICAL EXPLAINED VARIANCE"
    )
    print("=" * 130)
    print()


    for component in range(
        len(
            result[
                "explained"
            ]
        )
    ):

        print(
            f"PC{component + 1}: "
            f"{result['explained'][component]:.4f} "
            f"| cumulative="
            f"{result['cumulative'][component]:.4f}"
        )


# =========================================================
# Normalized 2D coordinates
#
# This is our main anatomical manifold.
# =========================================================

anatomical_manifold = pd.DataFrame(
    {
        "goal_column":
            list(
                range(
                    1,
                    10,
                )
            ),

        "raw_total_weight":
            raw_totals,

        "nonzero_pfl_targets":
            nonzero_targets,

        "PC1":
            norm_result[
                "pc1"
            ],

        "PC2":
            norm_result[
                "pc2"
            ],

        "radius_2d":
            norm_result[
                "radius"
            ],

        "latent_angle_deg":
            norm_result[
                "angle"
            ],
    }
)


print()
print("=" * 140)
print("L1-NORMALIZED ANATOMICAL FC2 COORDINATES")
print("=" * 140)
print()


print(
    anatomical_manifold
    .sort_values(
        "latent_angle_deg"
    )
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.4f}",
    )
)


# =========================================================
# Main 2D anatomical diagnostic
# =========================================================

print()
print("=" * 120)
print("ANATOMICAL 2D MANIFOLD DIAGNOSTIC")
print("=" * 120)
print()


print(
    "RAW projections:"
)

print(
    f"  first 2 PCs variance: "
    f"{raw_result['cumulative'][1]:.4f}"
)

print(
    f"  full projection R2: "
    f"{raw_result['full_r2']:.4f}"
)

print(
    f"  mean per-column correlation: "
    f"{raw_result['per_goal']['correlation'].mean():.4f}"
)


print()

print(
    "L1-normalized projection shape:"
)

print(
    f"  first 2 PCs variance: "
    f"{norm_result['cumulative'][1]:.4f}"
)

print(
    f"  full projection R2: "
    f"{norm_result['full_r2']:.4f}"
)

print(
    f"  reconstruction MAE: "
    f"{norm_result['mae']:.6f}"
)

print(
    f"  reconstruction RMSE: "
    f"{norm_result['rmse']:.6f}"
)

print(
    f"  mean per-column correlation: "
    f"{norm_result['per_goal']['correlation'].mean():.4f}"
)

print(
    f"  mean per-column cosine similarity: "
    f"{norm_result['per_goal']['cosine_similarity'].mean():.4f}"
)

print(
    f"  angular covered arc: "
    f"{norm_result['covered_arc']:.2f}°"
)

print(
    f"  largest angular gap: "
    f"{norm_result['largest_gap']:.2f}°"
)

print(
    f"  radius coefficient of variation: "
    f"{norm_result['radius_cv']:.4f}"
)


# =========================================================
# Per-column anatomical reconstruction
# =========================================================

print()
print("=" * 140)
print("2D ANATOMICAL RECONSTRUCTION BY FC2 COLUMN")
print("=" * 140)
print()


print(
    norm_result[
        "per_goal"
    ][
        [
            "goal_column",
            "correlation",
            "cosine_similarity",
            "MAE",
            "RMSE",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}",
    )
)


# =========================================================
# Compare anatomical manifold to steering-output manifold
#
# PCA axes themselves are arbitrary, so do NOT directly
# compare PC1 or PC2 signs.
#
# Instead compare pairwise distances between goals.
# This is invariant to rotation/reflection.
# =========================================================

def pairwise_distance_vector(
    coordinates,
):

    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )


    distances = []


    for i in range(
        len(
            coordinates
        )
    ):

        for j in range(
            i + 1,
            len(
                coordinates
            )
        ):

            distances.append(
                float(
                    np.linalg.norm(
                        coordinates[
                            i
                        ]
                        -
                        coordinates[
                            j
                        ]
                    )
                )
            )


    return np.asarray(
        distances,
        dtype=float,
    )


if STEERING_MANIFOLD_FILE.exists():

    steering_manifold = pd.read_csv(
        STEERING_MANIFOLD_FILE
    )


    steering_manifold = (
        steering_manifold
        .sort_values(
            "goal_column"
        )
        .reset_index(
            drop=True
        )
    )


    anatomical_sorted = (
        anatomical_manifold
        .sort_values(
            "goal_column"
        )
        .reset_index(
            drop=True
        )
    )


    steering_coordinates = (
        steering_manifold[
            [
                "PC1",
                "PC2",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )


    anatomical_coordinates = (
        anatomical_sorted[
            [
                "PC1",
                "PC2",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )


    steering_distances = (
        pairwise_distance_vector(
            steering_coordinates
        )
    )


    anatomical_distances = (
        pairwise_distance_vector(
            anatomical_coordinates
        )
    )


    geometry_correlation = 0.0


    if (
        np.std(
            steering_distances
        ) > 1e-12

        and

        np.std(
            anatomical_distances
        ) > 1e-12
    ):

        geometry_correlation = float(
            np.corrcoef(
                steering_distances,
                anatomical_distances,
            )[0, 1]
        )


    # -----------------------------------------------------
    # Nearest-neighbour agreement
    # -----------------------------------------------------

    nearest_matches = 0


    for index in range(
        9
    ):

        steering_d = np.linalg.norm(
            steering_coordinates
            -
            steering_coordinates[
                index
            ],
            axis=1,
        )


        anatomical_d = np.linalg.norm(
            anatomical_coordinates
            -
            anatomical_coordinates[
                index
            ],
            axis=1,
        )


        steering_d[
            index
        ] = np.inf

        anatomical_d[
            index
        ] = np.inf


        steering_neighbour = int(
            np.argmin(
                steering_d
            )
        )


        anatomical_neighbour = int(
            np.argmin(
                anatomical_d
            )
        )


        if (
            steering_neighbour
            ==
            anatomical_neighbour
        ):

            nearest_matches += 1


    print()
    print("=" * 120)
    print("ANATOMY VS STEERING MANIFOLD")
    print("=" * 120)
    print()


    print(
        f"Pairwise 2D geometry correlation: "
        f"{geometry_correlation:.4f}"
    )


    print(
        f"Nearest-neighbour agreement: "
        f"{nearest_matches}/9"
    )


else:

    print()
    print(
        "Steering manifold file not found; "
        "geometry comparison skipped."
    )


# =========================================================
# Save
# =========================================================

anatomical_manifold.to_csv(
    MANIFOLD_FILE,
    index=False,
)


norm_result[
    "per_goal"
].to_csv(
    RECONSTRUCTION_FILE,
    index=False,
)


print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)

print(
    PROJECTION_MATRIX_FILE
)

print(
    MANIFOLD_FILE
)

print(
    RECONSTRUCTION_FILE
)