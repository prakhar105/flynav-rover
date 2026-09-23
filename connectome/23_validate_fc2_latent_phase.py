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
    / "fc2_latent_phase_pair_residuals.csv"
)

LOO_FILE = (
    OUTPUT_DIR
    / "fc2_latent_phase_loo.csv"
)

STABILITY_FILE = (
    OUTPUT_DIR
    / "fc2_latent_phase_stability.csv"
)


# =========================================================
# Settings
# =========================================================

N_PHASE_BINS = 16

RANDOM_RESTARTS = 40

RNG_SEED = 42


# =========================================================
# Load pairwise observations
# =========================================================

pairs = pd.read_csv(
    INPUT_FILE
)


required = [
    "goal_i",
    "goal_j",
    "correlation",
    "observed_shift",
]


missing = [
    column
    for column in required
    if column not in pairs.columns
]


if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


pairs = pairs[
    required
].copy()


pairs[
    "fit_weight"
] = (
    np.clip(
        pairs[
            "correlation"
        ].astype(float),
        0.0,
        1.0,
    )
    ** 2
)


GOALS = sorted(
    set(
        pairs[
            "goal_i"
        ].astype(int)
    )
    |
    set(
        pairs[
            "goal_j"
        ].astype(int)
    )
)


goal_to_index = {
    goal: index
    for index, goal
    in enumerate(
        GOALS
    )
}


print("=" * 120)
print("FC2 LATENT PHASE CROSS-VALIDATION")
print("=" * 120)

print()

print(
    f"Goals: {GOALS}"
)

print(
    f"Pairwise observations: "
    f"{len(pairs)}"
)

print(
    f"Random restarts per fit: "
    f"{RANDOM_RESTARTS}"
)


# =========================================================
# Circular utilities
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


# =========================================================
# Convert dataframe to compact numpy arrays
# =========================================================

def prepare_arrays(
    dataframe,
):

    i_indices = np.asarray(
        [
            goal_to_index[
                int(x)
            ]
            for x in dataframe[
                "goal_i"
            ]
        ],
        dtype=int,
    )


    j_indices = np.asarray(
        [
            goal_to_index[
                int(x)
            ]
            for x in dataframe[
                "goal_j"
            ]
        ],
        dtype=int,
    )


    observed = dataframe[
        "observed_shift"
    ].to_numpy(
        dtype=float
    )


    weights = dataframe[
        "fit_weight"
    ].to_numpy(
        dtype=float
    )


    return (
        i_indices,
        j_indices,
        observed,
        weights,
    )


# =========================================================
# Objective
# =========================================================

def fit_error(
    positions,
    arrays,
):

    (
        i_indices,
        j_indices,
        observed,
        weights,
    ) = arrays


    predicted = (
        positions[
            i_indices
        ]
        -
        positions[
            j_indices
        ]
    )


    residual = (
        (
            predicted
            -
            observed
            +
            N_PHASE_BINS / 2
        )
        % N_PHASE_BINS
        -
        N_PHASE_BINS / 2
    )


    denominator = float(
        weights.sum()
    )


    if denominator <= 0:

        return np.inf


    return float(
        np.sum(
            weights
            * residual
            * residual
        )
        /
        denominator
    )


# =========================================================
# Reference-derived initialization
#
# C1 remains fixed at phase 0.
# =========================================================

def reference_initialization(
    dataframe,
):

    positions = np.zeros(
        len(GOALS),
        dtype=int,
    )


    reference_goal = GOALS[
        0
    ]


    for goal in GOALS[
        1:
    ]:

        direct = dataframe[
            (
                dataframe[
                    "goal_i"
                ]
                == reference_goal
            )
            &
            (
                dataframe[
                    "goal_j"
                ]
                == goal
            )
        ]


        reverse = dataframe[
            (
                dataframe[
                    "goal_i"
                ]
                == goal
            )
            &
            (
                dataframe[
                    "goal_j"
                ]
                == reference_goal
            )
        ]


        if len(direct) == 1:

            shift = int(
                direct.iloc[0][
                    "observed_shift"
                ]
            )


            positions[
                goal_to_index[
                    goal
                ]
            ] = (
                -shift
            ) % N_PHASE_BINS


        elif len(reverse) == 1:

            shift = int(
                reverse.iloc[0][
                    "observed_shift"
                ]
            )


            positions[
                goal_to_index[
                    goal
                ]
            ] = (
                shift
            ) % N_PHASE_BINS


    positions[
        0
    ] = 0


    return positions


# =========================================================
# Coordinate descent
# =========================================================

def coordinate_descent(
    initial_positions,
    arrays,
):

    positions = (
        initial_positions.copy()
    )


    positions[
        0
    ] = 0


    for _ in range(
        100
    ):

        changed = False


        # C1 remains our arbitrary phase anchor.
        for index in range(
            1,
            len(GOALS),
        ):

            current = int(
                positions[
                    index
                ]
            )


            best_position = current

            best_error = fit_error(
                positions,
                arrays,
            )


            for candidate in range(
                N_PHASE_BINS
            ):

                if candidate == current:

                    continue


                trial = positions.copy()

                trial[
                    index
                ] = candidate


                error = fit_error(
                    trial,
                    arrays,
                )


                if (
                    error
                    <
                    best_error
                    - 1e-12
                ):

                    best_error = error

                    best_position = (
                        candidate
                    )


            if (
                best_position
                != current
            ):

                positions[
                    index
                ] = best_position

                changed = True


        if not changed:

            break


    return (
        positions,
        fit_error(
            positions,
            arrays,
        ),
    )


# =========================================================
# Multi-start fit
#
# This reduces the chance that the previous coordinate
# descent result was merely a local minimum.
# =========================================================

def multistart_fit(
    dataframe,
    seed,
):

    arrays = prepare_arrays(
        dataframe
    )


    rng = np.random.default_rng(
        seed
    )


    starts = []


    # -----------------------------------------------------
    # Previous/reference-like initialization
    # -----------------------------------------------------

    starts.append(
        reference_initialization(
            dataframe
        )
    )


    # -----------------------------------------------------
    # Completely collapsed initialization
    # -----------------------------------------------------

    starts.append(
        np.zeros(
            len(GOALS),
            dtype=int,
        )
    )


    # -----------------------------------------------------
    # Simple sequential initialization
    # -----------------------------------------------------

    sequential = (
        np.arange(
            len(GOALS)
        )
        % N_PHASE_BINS
    )


    sequential[
        0
    ] = 0


    starts.append(
        sequential
    )


    # -----------------------------------------------------
    # Deterministic random restarts
    # -----------------------------------------------------

    for _ in range(
        RANDOM_RESTARTS
    ):

        trial = rng.integers(
            0,
            N_PHASE_BINS,
            size=len(GOALS),
            dtype=int,
        )


        trial[
            0
        ] = 0


        starts.append(
            trial
        )


    best_positions = None

    best_error = np.inf


    for start in starts:

        (
            fitted,
            error,
        ) = coordinate_descent(
            start,
            arrays,
        )


        if (
            error
            <
            best_error
            - 1e-12
        ):

            best_error = error

            best_positions = (
                fitted.copy()
            )


    return (
        best_positions,
        best_error,
    )


# =========================================================
# First:
# refit ALL pairs with multiple restarts.
#
# This checks whether Step 22 landed in a local optimum.
# =========================================================

(
    full_positions,
    full_error,
) = multistart_fit(
    pairs,
    seed=RNG_SEED,
)


print()
print("=" * 120)
print("MULTI-START FULL FIT")
print("=" * 120)
print()


for goal in GOALS:

    phase = int(
        full_positions[
            goal_to_index[
                goal
            ]
        ]
    )


    signed = int(
        circular_difference(
            phase
        )
    )


    print(
        f"C{goal}: "
        f"phase={phase:2d} | "
        f"signed={signed:+d}"
    )


print()

print(
    f"Weighted full-fit error: "
    f"{full_error:.4f}"
)


# =========================================================
# Leave-one-pair-out validation
# =========================================================

loo_rows = []

loo_position_rows = []


print()
print("=" * 120)
print("RUNNING LEAVE-ONE-PAIR-OUT")
print("=" * 120)
print()


for held_out_index in range(
    len(pairs)
):

    held_out = pairs.iloc[
        held_out_index
    ]


    training = pairs.drop(
        index=pairs.index[
            held_out_index
        ]
    ).reset_index(
        drop=True
    )


    (
        positions,
        training_error,
    ) = multistart_fit(
        training,
        seed=(
            RNG_SEED
            +
            held_out_index
            +
            1
        ),
    )


    goal_i = int(
        held_out[
            "goal_i"
        ]
    )

    goal_j = int(
        held_out[
            "goal_j"
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


    predicted_shift = int(
        circular_difference(
            phase_i
            -
            phase_j
        )
    )


    observed_shift = int(
        held_out[
            "observed_shift"
        ]
    )


    residual = float(
        circular_difference(
            predicted_shift
            -
            observed_shift
        )
    )


    loo_rows.append(
        {
            "goal_i":
                goal_i,

            "goal_j":
                goal_j,

            "correlation":
                float(
                    held_out[
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

            "training_error":
                float(
                    training_error
                ),
        }
    )


    position_record = {
        "held_out_goal_i":
            goal_i,

        "held_out_goal_j":
            goal_j,
    }


    for goal in GOALS:

        phase = int(
            positions[
                goal_to_index[
                    goal
                ]
            ]
        )


        position_record[
            f"C{goal}"
        ] = phase


    loo_position_rows.append(
        position_record
    )


loo = pd.DataFrame(
    loo_rows
)


loo_positions = pd.DataFrame(
    loo_position_rows
)


loo.to_csv(
    LOO_FILE,
    index=False,
)


# =========================================================
# LOO summary
# =========================================================

mean_abs = float(
    loo[
        "absolute_residual_bins"
    ].mean()
)


median_abs = float(
    loo[
        "absolute_residual_bins"
    ].median()
)


max_abs = float(
    loo[
        "absolute_residual_bins"
    ].max()
)


within_one = int(
    (
        loo[
            "absolute_residual_bins"
        ]
        <= 1
    ).sum()
)


within_two = int(
    (
        loo[
            "absolute_residual_bins"
        ]
        <= 2
    ).sum()
)


print()
print("=" * 120)
print("LEAVE-ONE-PAIR-OUT QUALITY")
print("=" * 120)
print()


print(
    f"Held-out relationships: "
    f"{len(loo)}"
)


print(
    f"Mean absolute residual: "
    f"{mean_abs:.3f} EPG bins"
)


print(
    f"Median absolute residual: "
    f"{median_abs:.3f} EPG bins"
)


print(
    f"Maximum absolute residual: "
    f"{max_abs:.3f} EPG bins"
)


print(
    f"Held-out pairs within ±1 bin: "
    f"{within_one}/{len(loo)}"
)


print(
    f"Held-out pairs within ±2 bins: "
    f"{within_two}/{len(loo)}"
)


# =========================================================
# High-confidence held-out relationships
# =========================================================

high_confidence = loo[
    loo[
        "correlation"
    ]
    >= 0.70
].copy()


print()
print("=" * 140)
print("HIGH-CONFIDENCE HELD-OUT PAIRS")
print("=" * 140)
print()


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


# =========================================================
# Phase stability
#
# Across all leave-one-pair-out refits, how stable is the
# inferred location of every FC2 column?
# =========================================================

stability_rows = []


for goal in GOALS:

    column = (
        loo_positions[
            f"C{goal}"
        ]
        .to_numpy(
            dtype=int
        )
    )


    counts = np.bincount(
        column,
        minlength=N_PHASE_BINS,
    )


    mode_phase = int(
        np.argmax(
            counts
        )
    )


    mode_count = int(
        counts[
            mode_phase
        ]
    )


    circular_deviation = np.asarray(
        [
            abs(
                circular_difference(
                    int(value)
                    -
                    mode_phase
                )
            )
            for value in column
        ],
        dtype=float,
    )


    stability_rows.append(
        {
            "goal_column":
                goal,

            "mode_phase_bin":
                mode_phase,

            "mode_fraction":
                mode_count
                /
                len(column),

            "mean_abs_deviation_from_mode":
                float(
                    circular_deviation.mean()
                ),

            "max_abs_deviation_from_mode":
                float(
                    circular_deviation.max()
                ),

            "unique_phase_bins_seen":
                int(
                    np.unique(
                        column
                    ).size
                ),
        }
    )


stability = pd.DataFrame(
    stability_rows
)


stability.to_csv(
    STABILITY_FILE,
    index=False,
)


print()
print("=" * 140)
print("LATENT PHASE STABILITY")
print("=" * 140)
print()


print(
    stability.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Worst held-out cases
# =========================================================

print()
print("=" * 140)
print("WORST HELD-OUT RELATIONSHIPS")
print("=" * 140)
print()


print(
    loo
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
    .head(10)
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Files
# =========================================================

print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)

print(
    LOO_FILE
)

print(
    STABILITY_FILE
)