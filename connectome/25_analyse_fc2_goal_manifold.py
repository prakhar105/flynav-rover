from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = (
    OUTPUT_DIR
    / "dna_decoder_comparison.csv"
)

MANIFOLD_FILE = (
    OUTPUT_DIR
    / "fc2_goal_manifold.csv"
)

RECONSTRUCTION_FILE = (
    OUTPUT_DIR
    / "fc2_manifold_reconstruction.csv"
)


# =========================================================
# Settings
# =========================================================

DECODER = "soft_activity_wide"


EPG_ORDER = [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "R8",
    "R7",
    "R6",
    "R5",
    "R4",
    "R3",
    "R2",
    "R1",
]


# =========================================================
# Load
# =========================================================

df = pd.read_csv(
    INPUT_FILE
)


required = [
    "heading_column",
    "goal_column",
    DECODER,
]


missing = [
    column
    for column in required
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# =========================================================
# Build steering matrix
#
# Rows:
#     EPG heading populations
#
# Columns:
#     FC2 goal populations
# =========================================================

matrix = (
    df
    .pivot(
        index="heading_column",
        columns="goal_column",
        values=DECODER,
    )
    .reindex(
        EPG_ORDER
    )
)


goal_columns = sorted(
    matrix.columns
)


matrix = matrix[
    goal_columns
]


print("=" * 120)
print("FC2 GOAL MANIFOLD ANALYSIS")
print("=" * 120)

print()

print(
    f"Decoder: {DECODER}"
)

print(
    f"Heading dimensions: "
    f"{matrix.shape[0]}"
)

print(
    f"FC2 goal populations: "
    f"{matrix.shape[1]}"
)


# =========================================================
# Goal profiles
#
# Each FC2 column becomes one 16-dimensional vector.
# =========================================================

raw_profiles = (
    matrix
    .to_numpy(
        dtype=float
    )
    .T
)


# =========================================================
# Remove common heading-dependent response
#
# We want PCA/SVD to describe differences caused by
# changing FC2 goal population, rather than the mean
# steering profile common to every goal.
# =========================================================

mean_profile = (
    raw_profiles.mean(
        axis=0,
        keepdims=True,
    )
)


X = (
    raw_profiles
    -
    mean_profile
)


# =========================================================
# SVD / PCA
# =========================================================

U, singular_values, Vt = np.linalg.svd(
    X,
    full_matrices=False,
)


variance = (
    singular_values ** 2
)


total_variance = float(
    variance.sum()
)


if total_variance > 0:

    explained_ratio = (
        variance
        /
        total_variance
    )

else:

    explained_ratio = np.zeros_like(
        variance
    )


cumulative_ratio = np.cumsum(
    explained_ratio
)


print()
print("=" * 120)
print("EXPLAINED VARIANCE")
print("=" * 120)
print()


for component in range(
    len(
        singular_values
    )
):

    print(
        f"PC{component + 1}: "
        f"{explained_ratio[component]:.4f} "
        f"| cumulative="
        f"{cumulative_ratio[component]:.4f}"
    )


# =========================================================
# Goal coordinates
#
# Scores = U * S
# =========================================================

scores = (
    U
    *
    singular_values
)


pc1 = (
    scores[:, 0]
    if scores.shape[1] >= 1
    else np.zeros(
        len(goal_columns)
    )
)


pc2 = (
    scores[:, 1]
    if scores.shape[1] >= 2
    else np.zeros(
        len(goal_columns)
    )
)


pc3 = (
    scores[:, 2]
    if scores.shape[1] >= 3
    else np.zeros(
        len(goal_columns)
    )
)


# =========================================================
# 2D polar representation
#
# IMPORTANT:
#
# This angle is a data-derived latent angle.
# It is NOT yet a world heading angle.
# =========================================================

radius = np.sqrt(
    pc1 ** 2
    +
    pc2 ** 2
)


latent_angle_rad = np.arctan2(
    pc2,
    pc1,
)


latent_angle_deg = (
    np.degrees(
        latent_angle_rad
    )
    % 360.0
)


manifold = pd.DataFrame(
    {
        "goal_column":
            goal_columns,

        "PC1":
            pc1,

        "PC2":
            pc2,

        "PC3":
            pc3,

        "radius_2d":
            radius,

        "latent_angle_deg":
            latent_angle_deg,
    }
)


# =========================================================
# Circular coverage
#
# If the FC2 populations form a useful circular code,
# their 2D angles should occupy a substantial portion
# of the circle rather than collapsing into one sector.
# =========================================================

sorted_angles = np.sort(
    latent_angle_deg
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


angular_gaps = np.diff(
    circular_angles
)


largest_gap = float(
    angular_gaps.max()
)


covered_arc = (
    360.0
    -
    largest_gap
)


mean_radius = float(
    radius.mean()
)


radius_std = float(
    radius.std()
)


radius_cv = (
    radius_std
    /
    mean_radius

    if mean_radius > 1e-12

    else np.inf
)


print()
print("=" * 130)
print("2D FC2 GOAL COORDINATES")
print("=" * 130)
print()


print(
    manifold
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
# Reconstruction tests
#
# How many latent dimensions are actually required to
# reconstruct the connectome-derived steering surface?
# =========================================================

reconstruction_rows = []


original = (
    raw_profiles
)


original_mean = float(
    original.mean()
)


ss_total = float(
    np.sum(
        (
            original
            -
            original_mean
        )
        ** 2
    )
)


max_components = min(
    8,
    len(
        singular_values
    ),
)


for k in range(
    1,
    max_components + 1,
):

    X_reconstructed = (
        U[:, :k]
        @
        np.diag(
            singular_values[:k]
        )
        @
        Vt[:k, :]
    )


    reconstructed = (
        X_reconstructed
        +
        mean_profile
    )


    residual = (
        original
        -
        reconstructed
    )


    ss_residual = float(
        np.sum(
            residual ** 2
        )
    )


    r_squared = (
        1.0
        -
        ss_residual
        /
        ss_total

        if ss_total > 0

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


    max_error = float(
        np.max(
            np.abs(
                residual
            )
        )
    )


    reconstruction_rows.append(
        {
            "components":
                k,

            "cumulative_centered_variance":
                float(
                    cumulative_ratio[
                        k - 1
                    ]
                ),

            "full_surface_R2":
                r_squared,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "max_abs_error":
                max_error,
        }
    )


reconstruction_summary = pd.DataFrame(
    reconstruction_rows
)


print()
print("=" * 140)
print("MANIFOLD RECONSTRUCTION QUALITY")
print("=" * 140)
print()


print(
    reconstruction_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Per-goal 2D reconstruction
# =========================================================

k = min(
    2,
    len(
        singular_values
    ),
)


X_2d = (
    U[:, :k]
    @
    np.diag(
        singular_values[:k]
    )
    @
    Vt[:k, :]
)


reconstructed_2d = (
    X_2d
    +
    mean_profile
)


per_goal_rows = []


for index, goal in enumerate(
    goal_columns
):

    truth = (
        original[
            index
        ]
    )


    prediction = (
        reconstructed_2d[
            index
        ]
    )


    residual = (
        truth
        -
        prediction
    )


    correlation = 0.0


    if (
        np.std(
            truth
        )
        > 1e-12

        and

        np.std(
            prediction
        )
        > 1e-12
    ):

        correlation = float(
            np.corrcoef(
                truth,
                prediction,
            )[0, 1]
        )


    meaningful = (
        np.abs(
            truth
        )
        >= 0.05
    )


    sign_match = (
        np.sign(
            truth
        )
        ==
        np.sign(
            prediction
        )
    )


    if meaningful.sum() > 0:

        sign_accuracy = float(
            sign_match[
                meaningful
            ].mean()
        )

    else:

        sign_accuracy = np.nan


    per_goal_rows.append(
        {
            "goal_column":
                goal,

            "correlation":
                correlation,

            "MAE":
                float(
                    np.mean(
                        np.abs(
                            residual
                        )
                    )
                ),

            "RMSE":
                float(
                    np.sqrt(
                        np.mean(
                            residual ** 2
                        )
                    )
                ),

            "sign_accuracy":
                sign_accuracy,
        }
    )


per_goal = pd.DataFrame(
    per_goal_rows
)


print()
print("=" * 130)
print("2D RECONSTRUCTION BY FC2 GOAL")
print("=" * 130)
print()


print(
    per_goal.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Circular ordering by PCA angle
# =========================================================

angle_order = (
    manifold
    .sort_values(
        "latent_angle_deg"
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 120)
print("FC2 MANIFOLD ORDER")
print("=" * 120)
print()


for index, row in angle_order.iterrows():

    print(
        f"{index + 1:2d}. "
        f"C{int(row['goal_column'])} "
        f"| angle="
        f"{row['latent_angle_deg']:.1f}° "
        f"| radius="
        f"{row['radius_2d']:.4f}"
    )


# =========================================================
# Global 2D diagnostic
# =========================================================

two_component_variance = float(
    cumulative_ratio[
        min(
            1,
            len(
                cumulative_ratio
            )
            - 1
        )
    ]
)


print()
print("=" * 120)
print("2D MANIFOLD DIAGNOSTIC")
print("=" * 120)
print()


print(
    f"Variance explained by first 2 PCs: "
    f"{two_component_variance:.4f}"
)


print(
    f"2D angular covered arc: "
    f"{covered_arc:.2f}°"
)


print(
    f"Largest empty angular gap: "
    f"{largest_gap:.2f}°"
)


print(
    f"Mean 2D radius: "
    f"{mean_radius:.4f}"
)


print(
    f"Radius coefficient of variation: "
    f"{radius_cv:.4f}"
)


print(
    f"Mean per-goal 2D correlation: "
    f"{per_goal['correlation'].mean():.4f}"
)


print(
    f"Mean per-goal 2D sign accuracy: "
    f"{per_goal['sign_accuracy'].dropna().mean():.4f}"
)


# =========================================================
# Save
# =========================================================

manifold.to_csv(
    MANIFOLD_FILE,
    index=False,
)


reconstruction_summary.to_csv(
    RECONSTRUCTION_FILE,
    index=False,
)


print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)

print(
    MANIFOLD_FILE
)

print(
    RECONSTRUCTION_FILE
)