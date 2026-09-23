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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fc2_latent_phase.csv"
)

PAIR_OUTPUT_FILE = (
    OUTPUT_DIR
    / "fc2_latent_phase_pair_residuals.csv"
)


# =========================================================
# Settings
# =========================================================

DECODER = "soft_activity_wide"

N_EPG = 16

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
        f"Missing required columns: {missing}"
    )


# =========================================================
# Build 16 x 9 steering matrix
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
print("FC2 LATENT PHASE FIT")
print("=" * 120)

print()

print(
    f"Decoder: {DECODER}"
)

print(
    f"EPG positions: {N_EPG}"
)

print(
    f"FC2 columns: {len(goal_columns)}"
)


# =========================================================
# Correlation
# =========================================================

def safe_correlation(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )


    if (
        np.std(a) < 1e-12
        or
        np.std(b) < 1e-12
    ):
        return 0.0


    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


# =========================================================
# Best circular alignment
#
# shift means:
#
# roll SECOND by shift positions
# so it aligns with FIRST.
#
# Therefore:
#
# observed_shift(i,j)
# ≈
# phase_i - phase_j
#
# modulo 16.
# =========================================================

def best_circular_alignment(
    first,
    second,
):

    best_shift = None
    best_corr = -np.inf


    for shift in range(
        -8,
        8,
    ):

        shifted = np.roll(
            second,
            shift,
        )


        corr = safe_correlation(
            first,
            shifted,
        )


        if corr > best_corr:

            best_corr = corr
            best_shift = shift


    return (
        int(best_shift),
        float(best_corr),
    )


# =========================================================
# Circular difference
#
# Return signed shortest distance:
#
# -8 ... +7
# =========================================================

def circular_difference(
    value,
    modulus=N_EPG,
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


# =========================================================
# Build all pairwise observations
# =========================================================

pair_rows = []


for i, goal_i in enumerate(
    goal_columns
):

    profile_i = (
        matrix[
            goal_i
        ]
        .to_numpy(
            dtype=float
        )
    )


    for j, goal_j in enumerate(
        goal_columns
    ):

        if j <= i:
            continue


        profile_j = (
            matrix[
                goal_j
            ]
            .to_numpy(
                dtype=float
            )
        )


        (
            shift,
            correlation,
        ) = best_circular_alignment(
            profile_i,
            profile_j,
        )


        pair_rows.append(
            {
                "goal_i":
                    int(goal_i),

                "goal_j":
                    int(goal_j),

                "observed_shift":
                    int(shift),

                "correlation":
                    float(correlation),
            }
        )


pairs = pd.DataFrame(
    pair_rows
)


# =========================================================
# Correlation-derived weights
#
# Stronger pairwise matches receive more influence.
#
# This is NOT a biological weight.
# It is only confidence in the alignment estimate.
# =========================================================

pairs[
    "fit_weight"
] = np.clip(
    pairs[
        "correlation"
    ],
    0.0,
    1.0,
) ** 2


print()
print("=" * 120)
print("PAIRWISE SHIFT OBSERVATIONS")
print("=" * 120)
print()


print(
    pairs.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Fit latent FC2 positions
#
# Position of C1 is fixed at 0 because absolute phase is
# arbitrary at this stage.
#
# We fit integer locations 0..15.
#
# Objective:
#
# predicted shift(i,j)
#     =
# phase_i - phase_j
#
# should agree with the observed pairwise alignment shifts.
#
# Coordinate descent is enough because there are only
# nine FC2 columns and sixteen possible positions each.
# =========================================================

goal_to_index = {
    goal: index
    for index, goal
    in enumerate(
        goal_columns
    )
}


# ---------------------------------------------------------
# Initial positions
#
# Use C1 pairwise shifts as the initial estimate.
#
# If:
#
# observed_shift(C1, Cj)
# ≈ phase_C1 - phase_Cj
#
# and phase_C1 = 0,
#
# then:
#
# phase_Cj ≈ -observed_shift
# ---------------------------------------------------------

positions = np.zeros(
    len(goal_columns),
    dtype=int,
)


reference_goal = goal_columns[
    0
]


for goal in goal_columns[
    1:
]:

    matching = pairs[
        (
            pairs[
                "goal_i"
            ]
            == reference_goal
        )
        &
        (
            pairs[
                "goal_j"
            ]
            == goal
        )
    ]


    if len(matching) == 1:

        shift = int(
            matching.iloc[0][
                "observed_shift"
            ]
        )

        positions[
            goal_to_index[
                goal
            ]
        ] = (
            -shift
        ) % N_EPG


# =========================================================
# Objective
# =========================================================

def fit_error(
    candidate_positions,
):

    total_error = 0.0
    total_weight = 0.0


    for _, row in pairs.iterrows():

        goal_i = int(
            row[
                "goal_i"
            ]
        )

        goal_j = int(
            row[
                "goal_j"
            ]
        )

        observed = float(
            row[
                "observed_shift"
            ]
        )

        weight = float(
            row[
                "fit_weight"
            ]
        )


        phase_i = int(
            candidate_positions[
                goal_to_index[
                    goal_i
                ]
            ]
        )

        phase_j = int(
            candidate_positions[
                goal_to_index[
                    goal_j
                ]
            ]
        )


        predicted = (
            phase_i
            -
            phase_j
        )


        residual = circular_difference(
            predicted
            -
            observed
        )


        total_error += (
            weight
            * residual
            * residual
        )


        total_weight += (
            weight
        )


    if total_weight <= 0:

        return np.inf


    return (
        total_error
        /
        total_weight
    )


# =========================================================
# Coordinate descent
# =========================================================

initial_error = fit_error(
    positions
)


print()
print("=" * 120)
print("INITIAL LATENT POSITIONS")
print("=" * 120)
print()


for goal, position in zip(
    goal_columns,
    positions,
):

    print(
        f"C{goal}: "
        f"{position}"
    )


print()

print(
    f"Initial weighted error: "
    f"{initial_error:.4f}"
)


# ---------------------------------------------------------
# C1 stays fixed at zero.
# ---------------------------------------------------------

MAX_ITERATIONS = 100


for iteration in range(
    MAX_ITERATIONS
):

    changed = False


    for goal in goal_columns[
        1:
    ]:

        index = goal_to_index[
            goal
        ]


        current_position = (
            positions[
                index
            ]
        )


        best_position = (
            current_position
        )

        best_error = fit_error(
            positions
        )


        for candidate in range(
            N_EPG
        ):

            if candidate == current_position:
                continue


            trial = (
                positions.copy()
            )


            trial[
                index
            ] = candidate


            error = fit_error(
                trial
            )


            if (
                error
                <
                best_error
                - 1e-12
            ):

                best_error = error
                best_position = candidate


        if (
            best_position
            != current_position
        ):

            positions[
                index
            ] = best_position

            changed = True


    if not changed:
        break


final_error = fit_error(
    positions
)


# =========================================================
# Convert to signed offsets around C1
# =========================================================

signed_positions = []


for position in positions:

    signed = circular_difference(
        position
    )


    signed_positions.append(
        int(signed)
    )


phase_table = pd.DataFrame(
    {
        "goal_column":
            goal_columns,

        "phase_bin":
            positions,

        "signed_phase_from_C1":
            signed_positions,
    }
)


print()
print("=" * 120)
print("FITTED FC2 LATENT PHASE")
print("=" * 120)
print()


print(
    phase_table.to_string(
        index=False
    )
)


print()

print(
    f"Final weighted error: "
    f"{final_error:.4f}"
)

print(
    f"Coordinate-descent iterations: "
    f"{iteration + 1}"
)


# =========================================================
# Pairwise residual analysis
# =========================================================

residual_rows = []


for _, row in pairs.iterrows():

    goal_i = int(
        row[
            "goal_i"
        ]
    )

    goal_j = int(
        row[
            "goal_j"
        ]
    )


    observed_shift = int(
        row[
            "observed_shift"
        ]
    )


    phase_i = int(
        positions[
            goal_to_index[
                goal_i
            ]
        ]
    )


    phase_j = int(
        positions[
            goal_to_index[
                goal_j
            ]
        ]
    )


    predicted_shift = (
        phase_i
        -
        phase_j
    )


    predicted_shift = int(
        circular_difference(
            predicted_shift
        )
    )


    residual = float(
        circular_difference(
            predicted_shift
            -
            observed_shift
        )
    )


    residual_rows.append(
        {
            "goal_i":
                goal_i,

            "goal_j":
                goal_j,

            "correlation":
                float(
                    row[
                        "correlation"
                    ]
                ),

            "observed_shift":
                observed_shift,

            "predicted_shift":
                predicted_shift,

            "residual_bins":
                residual,

            "absolute_residual_bins":
                abs(
                    residual
                ),
        }
    )


residuals = pd.DataFrame(
    residual_rows
)


# =========================================================
# Residual summary
# =========================================================

mean_abs_residual = float(
    residuals[
        "absolute_residual_bins"
    ].mean()
)


median_abs_residual = float(
    residuals[
        "absolute_residual_bins"
    ].median()
)


max_abs_residual = float(
    residuals[
        "absolute_residual_bins"
    ].max()
)


within_one = int(
    (
        residuals[
            "absolute_residual_bins"
        ]
        <= 1.0
    ).sum()
)


within_two = int(
    (
        residuals[
            "absolute_residual_bins"
        ]
        <= 2.0
    ).sum()
)


print()
print("=" * 120)
print("LATENT-PHASE FIT QUALITY")
print("=" * 120)
print()


print(
    f"Pairwise observations: "
    f"{len(residuals)}"
)


print(
    f"Mean absolute residual: "
    f"{mean_abs_residual:.3f} EPG bins"
)


print(
    f"Median absolute residual: "
    f"{median_abs_residual:.3f} EPG bins"
)


print(
    f"Maximum absolute residual: "
    f"{max_abs_residual:.3f} EPG bins"
)


print(
    f"Pairs within ±1 EPG bin: "
    f"{within_one}/{len(residuals)}"
)


print(
    f"Pairs within ±2 EPG bins: "
    f"{within_two}/{len(residuals)}"
)


# =========================================================
# High-confidence residuals
# =========================================================

high_confidence = residuals[
    residuals[
        "correlation"
    ]
    >= 0.70
].copy()


print()
print("=" * 140)
print("HIGH-CONFIDENCE PAIRS")
print("=" * 140)
print()


if len(
    high_confidence
) > 0:

    print(
        high_confidence
        .sort_values(
            [
                "absolute_residual_bins",
                "correlation",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

else:

    print(
        "No pairwise correlations >= 0.70."
    )


# =========================================================
# Adjacent fitted movements
#
# This shows directly whether C1..C9 are equally spaced.
# =========================================================

adjacent_rows = []


for index in range(
    len(goal_columns) - 1
):

    goal_a = (
        goal_columns[
            index
        ]
    )

    goal_b = (
        goal_columns[
            index + 1
        ]
    )


    phase_a = int(
        positions[
            goal_to_index[
                goal_a
            ]
        ]
    )


    phase_b = int(
        positions[
            goal_to_index[
                goal_b
            ]
        ]
    )


    movement = circular_difference(
        phase_b
        -
        phase_a
    )


    adjacent_rows.append(
        {
            "goal_a":
                goal_a,

            "goal_b":
                goal_b,

            "latent_step_bins":
                int(
                    movement
                ),
        }
    )


adjacent_table = pd.DataFrame(
    adjacent_rows
)


print()
print("=" * 120)
print("FITTED ADJACENT FC2 STEPS")
print("=" * 120)
print()


print(
    adjacent_table.to_string(
        index=False
    )
)


# =========================================================
# Save
# =========================================================

phase_table.to_csv(
    OUTPUT_FILE,
    index=False,
)


residuals.to_csv(
    PAIR_OUTPUT_FILE,
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
    PAIR_OUTPUT_FILE
)