from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MAP_FILE = (
    OUTPUT_DIR
    / "dna_decoder_comparison.csv"
)

PHASE_FILE = (
    OUTPUT_DIR
    / "fc2_latent_phase.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "relative_phase_code.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "relative_phase_summary.csv"
)


# =========================================================
# Settings
# =========================================================

DECODER = "soft_activity_wide"

N_PHASE_BINS = 16


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
    MAP_FILE
)

phase_df = pd.read_csv(
    PHASE_FILE
)


required_map = [
    "heading_column",
    "goal_column",
    DECODER,
]


required_phase = [
    "goal_column",
    "phase_bin",
]


for column in required_map:

    if column not in df.columns:

        raise RuntimeError(
            f"Missing map column: {column}"
        )


for column in required_phase:

    if column not in phase_df.columns:

        raise RuntimeError(
            f"Missing phase column: {column}"
        )


# =========================================================
# Heading anatomical phase
# =========================================================

heading_phase = {
    heading: index
    for index, heading
    in enumerate(
        EPG_ORDER
    )
}


df[
    "heading_phase"
] = df[
    "heading_column"
].map(
    heading_phase
)


if df[
    "heading_phase"
].isna().any():

    raise RuntimeError(
        "Unknown EPG heading column."
    )


# =========================================================
# FC2 latent phase
# =========================================================

goal_phase = {
    int(row["goal_column"]):
        int(row["phase_bin"])

    for _, row
    in phase_df.iterrows()
}


df[
    "goal_phase"
] = df[
    "goal_column"
].map(
    goal_phase
)


if df[
    "goal_phase"
].isna().any():

    raise RuntimeError(
        "Missing FC2 latent phase."
    )


# =========================================================
# Circular relative phase
#
# Positive/negative sign is currently arbitrary.
#
# We are testing structural collapse only.
# =========================================================

def circular_difference(
    value,
    modulus=N_PHASE_BINS,
):

    return (
        (
            value
            + modulus / 2
        )
        % modulus
        -
        modulus / 2
    )


df[
    "relative_phase"
] = [
    int(
        circular_difference(
            heading
            -
            goal
        )
    )

    for heading, goal
    in zip(
        df[
            "heading_phase"
        ],
        df[
            "goal_phase"
        ],
    )
]


# =========================================================
# Empirical relative-phase curve
# =========================================================

phase_summary = (
    df
    .groupby(
        "relative_phase"
    )
    .agg(
        count=(
            DECODER,
            "size",
        ),

        mean_steering=(
            DECODER,
            "mean",
        ),

        median_steering=(
            DECODER,
            "median",
        ),

        std_steering=(
            DECODER,
            "std",
        ),

        min_steering=(
            DECODER,
            "min",
        ),

        max_steering=(
            DECODER,
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        "relative_phase"
    )
)


phase_summary[
    "std_steering"
] = (
    phase_summary[
        "std_steering"
    ]
    .fillna(0.0)
)


print("=" * 120)
print("RELATIVE PHASE CODE TEST")
print("=" * 120)

print()

print(
    f"Conditions: {len(df)}"
)

print(
    f"Decoder: {DECODER}"
)


print()
print("=" * 140)
print("RELATIVE PHASE STEERING CURVE")
print("=" * 140)
print()


print(
    phase_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.4f}",
    )
)


# =========================================================
# Predict each condition using only relative phase
# =========================================================

phase_lookup = {
    int(row["relative_phase"]):
        float(row["mean_steering"])

    for _, row
    in phase_summary.iterrows()
}


df[
    "phase_only_prediction"
] = df[
    "relative_phase"
].map(
    phase_lookup
)


df[
    "phase_only_error"
] = (
    df[
        DECODER
    ]
    -
    df[
        "phase_only_prediction"
    ]
)


# =========================================================
# In-sample collapse quality
# =========================================================

y = df[
    DECODER
].to_numpy(
    dtype=float
)


prediction = df[
    "phase_only_prediction"
].to_numpy(
    dtype=float
)


residual = (
    y
    -
    prediction
)


ss_res = float(
    np.sum(
        residual ** 2
    )
)


ss_tot = float(
    np.sum(
        (
            y
            -
            np.mean(y)
        )
        ** 2
    )
)


r_squared = (
    1.0
    -
    ss_res
    / ss_tot

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


# =========================================================
# Leave-one-goal-column-out test
#
# Important:
#
# Build the relative-phase steering lookup without one
# FC2 goal column, then predict that entire held-out goal.
#
# This tells us whether the relative-phase rule
# generalizes across FC2 populations.
# =========================================================

loo_rows = []


for held_out_goal in sorted(
    df[
        "goal_column"
    ].unique()
):

    train = df[
        df[
            "goal_column"
        ]
        != held_out_goal
    ].copy()


    test = df[
        df[
            "goal_column"
        ]
        == held_out_goal
    ].copy()


    train_lookup = (
        train
        .groupby(
            "relative_phase"
        )[
            DECODER
        ]
        .mean()
        .to_dict()
    )


    global_mean = float(
        train[
            DECODER
        ].mean()
    )


    predictions = []


    for phase in test[
        "relative_phase"
    ]:

        phase = int(
            phase
        )


        if phase in train_lookup:

            predictions.append(
                float(
                    train_lookup[
                        phase
                    ]
                )
            )

            continue


        # -----------------------------------------------
        # Fallback:
        #
        # Find nearest represented circular phase.
        # -----------------------------------------------

        available = list(
            train_lookup.keys()
        )


        if not available:

            predictions.append(
                global_mean
            )

            continue


        distances = [
            abs(
                circular_difference(
                    phase
                    -
                    candidate
                )
            )

            for candidate
            in available
        ]


        nearest_index = int(
            np.argmin(
                distances
            )
        )


        nearest_phase = available[
            nearest_index
        ]


        predictions.append(
            float(
                train_lookup[
                    nearest_phase
                ]
            )
        )


    test_y = test[
        DECODER
    ].to_numpy(
        dtype=float
    )


    test_prediction = np.asarray(
        predictions,
        dtype=float,
    )


    test_residual = (
        test_y
        -
        test_prediction
    )


    goal_mae = float(
        np.mean(
            np.abs(
                test_residual
            )
        )
    )


    goal_rmse = float(
        np.sqrt(
            np.mean(
                test_residual ** 2
            )
        )
    )


    correlation = 0.0


    if (
        np.std(
            test_y
        ) > 1e-12

        and

        np.std(
            test_prediction
        ) > 1e-12
    ):

        correlation = float(
            np.corrcoef(
                test_y,
                test_prediction,
            )[0, 1]
        )


    sign_match = np.sign(
        test_y
    ) == np.sign(
        test_prediction
    )


    # Ignore tiny true outputs.
    meaningful = (
        np.abs(
            test_y
        )
        >= 0.05
    )


    if meaningful.sum() > 0:

        sign_accuracy = float(
            sign_match[
                meaningful
            ].mean()
        )

    else:

        sign_accuracy = np.nan


    loo_rows.append(
        {
            "held_out_goal":
                int(
                    held_out_goal
                ),

            "mae":
                goal_mae,

            "rmse":
                goal_rmse,

            "correlation":
                correlation,

            "sign_accuracy":
                sign_accuracy,

            "mean_true_abs":
                float(
                    np.mean(
                        np.abs(
                            test_y
                        )
                    )
                ),

            "mean_pred_abs":
                float(
                    np.mean(
                        np.abs(
                            test_prediction
                        )
                    )
                ),
        }
    )


loo = pd.DataFrame(
    loo_rows
)


# =========================================================
# Overall leave-one-goal summary
# =========================================================

print()
print("=" * 120)
print("RELATIVE-PHASE COLLAPSE QUALITY")
print("=" * 120)
print()


print(
    f"In-sample R^2: "
    f"{r_squared:.4f}"
)


print(
    f"In-sample MAE: "
    f"{mae:.4f}"
)


print(
    f"In-sample RMSE: "
    f"{rmse:.4f}"
)


print()

print(
    f"Mean within-phase steering std: "
    f"{phase_summary['std_steering'].mean():.4f}"
)


print(
    f"Median within-phase steering std: "
    f"{phase_summary['std_steering'].median():.4f}"
)


# =========================================================
# Held-out goal results
# =========================================================

print()
print("=" * 140)
print("LEAVE-ONE-GOAL-OUT RELATIVE-PHASE TEST")
print("=" * 140)
print()


print(
    loo.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


print()
print("=" * 120)
print("GENERALIZATION SUMMARY")
print("=" * 120)
print()


print(
    f"Mean held-out MAE: "
    f"{loo['mae'].mean():.4f}"
)


print(
    f"Median held-out MAE: "
    f"{loo['mae'].median():.4f}"
)


print(
    f"Mean held-out correlation: "
    f"{loo['correlation'].mean():.4f}"
)


print(
    f"Median held-out correlation: "
    f"{loo['correlation'].median():.4f}"
)


valid_sign = loo[
    "sign_accuracy"
].dropna()


if len(
    valid_sign
) > 0:

    print(
        f"Mean held-out sign accuracy: "
        f"{valid_sign.mean():.4f}"
    )


# =========================================================
# Worst individual conditions
# =========================================================

print()
print("=" * 140)
print("LARGEST RELATIVE-PHASE RESIDUALS")
print("=" * 140)
print()


print(
    df[
        [
            "heading_column",
            "goal_column",
            "heading_phase",
            "goal_phase",
            "relative_phase",
            DECODER,
            "phase_only_prediction",
            "phase_only_error",
        ]
    ]
    .assign(
        abs_error=lambda x:
        x[
            "phase_only_error"
        ].abs()
    )
    .sort_values(
        "abs_error",
        ascending=False,
    )
    .head(20)
    .drop(
        columns=[
            "abs_error",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.4f}",
    )
)


# =========================================================
# Save
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


phase_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)