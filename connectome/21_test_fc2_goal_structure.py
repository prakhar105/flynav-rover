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

PAIR_FILE = (
    OUTPUT_DIR
    / "fc2_goal_shift_pairs.csv"
)

GOAL_FILE = (
    OUTPUT_DIR
    / "fc2_goal_structure_summary.csv"
)


# =========================================================
# Decoder selected from Step 20
# =========================================================

DECODER = "soft_activity_wide"

NEUTRAL_THRESHOLD = 0.05


# =========================================================
# Anatomical EPG ring order
#
# This is anatomical ordering only.
# We still do NOT assign world angles here.
# =========================================================

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
# Build 16 x 9 matrix
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


matrix = matrix[
    sorted(
        matrix.columns
    )
]


print("=" * 120)
print("FC2 GOAL-STRUCTURE TEST")
print("=" * 120)

print()

print(
    f"Decoder: {DECODER}"
)

print(
    f"Heading columns: {len(matrix)}"
)

print(
    f"FC2 columns: {len(matrix.columns)}"
)


print()
print("=" * 120)
print("INPUT STEERING MATRIX")
print("=" * 120)
print()


print(
    matrix.to_string(
        float_format=lambda x:
        f"{x:+.3f}"
    )
)


# =========================================================
# Sign utility
# =========================================================

def sign_symbol(value):

    if value > NEUTRAL_THRESHOLD:
        return "R"

    if value < -NEUTRAL_THRESHOLD:
        return "L"

    return "."


# =========================================================
# Circular sign-transition count
#
# Ignore neutral elements before counting transitions.
# =========================================================

def sign_transition_count(
    values,
):

    signs = []


    for value in values:

        symbol = sign_symbol(
            value
        )

        if symbol != ".":
            signs.append(
                symbol
            )


    if len(signs) <= 1:
        return 0


    transitions = 0


    for index in range(
        len(signs)
    ):

        current = signs[
            index
        ]

        following = signs[
            (
                index + 1
            )
            % len(signs)
        ]


        if current != following:

            transitions += 1


    return transitions


# =========================================================
# Circular adjacent jumps
# =========================================================

def adjacent_jumps(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    return np.abs(
        np.roll(
            values,
            -1,
        )
        -
        values
    )


# =========================================================
# Per-goal topology
# =========================================================

goal_rows = []


for goal in matrix.columns:

    values = (
        matrix[
            goal
        ]
        .to_numpy(
            dtype=float
        )
    )


    abs_values = np.abs(
        values
    )


    min_index = int(
        np.argmin(
            abs_values
        )
    )


    max_right_index = int(
        np.argmax(
            values
        )
    )


    max_left_index = int(
        np.argmin(
            values
        )
    )


    jumps = adjacent_jumps(
        values
    )


    right_count = int(
        (
            values
            > NEUTRAL_THRESHOLD
        ).sum()
    )


    left_count = int(
        (
            values
            < -NEUTRAL_THRESHOLD
        ).sum()
    )


    neutral_count = int(
        len(values)
        -
        right_count
        -
        left_count
    )


    goal_rows.append(
        {
            "goal_column":
                int(goal),

            "right_count":
                right_count,

            "left_count":
                left_count,

            "neutral_count":
                neutral_count,

            "sign_transitions":
                sign_transition_count(
                    values
                ),

            "minimum_abs_heading":
                EPG_ORDER[
                    min_index
                ],

            "minimum_abs_output":
                float(
                    values[
                        min_index
                    ]
                ),

            "strongest_right_heading":
                EPG_ORDER[
                    max_right_index
                ],

            "strongest_right_output":
                float(
                    values[
                        max_right_index
                    ]
                ),

            "strongest_left_heading":
                EPG_ORDER[
                    max_left_index
                ],

            "strongest_left_output":
                float(
                    values[
                        max_left_index
                    ]
                ),

            "mean_adjacent_jump":
                float(
                    np.mean(
                        jumps
                    )
                ),

            "max_adjacent_jump":
                float(
                    np.max(
                        jumps
                    )
                ),
        }
    )


goal_summary = pd.DataFrame(
    goal_rows
)


goal_summary.to_csv(
    GOAL_FILE,
    index=False,
)


print()
print("=" * 150)
print("PER-GOAL TOPOLOGY")
print("=" * 150)
print()


print(
    goal_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Circular correlation
#
# Core question:
#
# Does FC2 C(k+1) look approximately like FC2 C(k),
# merely shifted around the anatomical EPG ring?
#
#
# If yes:
#   goal columns behave like an ordered phase code.
#
# If no:
#   a simple global angular offset is not justified.
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


def best_circular_alignment(
    first,
    second,
):

    best_correlation = -np.inf
    best_shift = None


    # 16 EPG positions.
    #
    # Search signed shifts:
    #
    # -8 ... +7
    #
    # Positive means:
    #
    # roll SECOND profile forward in EPG_ORDER
    # to align it with FIRST.
    # -----------------------------------------------------

    for shift in range(
        -8,
        8,
    ):

        shifted = np.roll(
            second,
            shift,
        )


        correlation = (
            safe_correlation(
                first,
                shifted,
            )
        )


        if (
            correlation
            > best_correlation
        ):

            best_correlation = (
                correlation
            )

            best_shift = (
                shift
            )


    return (
        best_shift,
        best_correlation,
    )


# =========================================================
# Adjacent FC2-column comparisons
# =========================================================

pair_rows = []


goal_columns = list(
    matrix.columns
)


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


    a = (
        matrix[
            goal_a
        ]
        .to_numpy(
            dtype=float
        )
    )


    b = (
        matrix[
            goal_b
        ]
        .to_numpy(
            dtype=float
        )
    )


    unshifted_corr = (
        safe_correlation(
            a,
            b,
        )
    )


    (
        best_shift,
        best_corr,
    ) = best_circular_alignment(
        a,
        b,
    )


    pair_rows.append(
        {
            "goal_a":
                int(goal_a),

            "goal_b":
                int(goal_b),

            "unshifted_correlation":
                unshifted_corr,

            "best_shift":
                best_shift,

            "best_shift_correlation":
                best_corr,
        }
    )


pair_summary = pd.DataFrame(
    pair_rows
)


# =========================================================
# Optional C9 -> C1 wrap test
#
# Kept separate because we do not yet assume the
# nine FC2 columns themselves form a complete circle.
# =========================================================

goal_9 = (
    matrix[
        goal_columns[-1]
    ]
    .to_numpy(
        dtype=float
    )
)


goal_1 = (
    matrix[
        goal_columns[0]
    ]
    .to_numpy(
        dtype=float
    )
)


wrap_unshifted = (
    safe_correlation(
        goal_9,
        goal_1,
    )
)


(
    wrap_shift,
    wrap_corr,
) = best_circular_alignment(
    goal_9,
    goal_1,
)


pair_summary.to_csv(
    PAIR_FILE,
    index=False,
)


print()
print("=" * 140)
print("ADJACENT FC2 COLUMN SHIFT TEST")
print("=" * 140)
print()


print(
    pair_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# Shift consistency
# =========================================================

shifts = (
    pair_summary[
        "best_shift"
    ]
    .to_numpy(
        dtype=int
    )
)


correlations = (
    pair_summary[
        "best_shift_correlation"
    ]
    .to_numpy(
        dtype=float
    )
)


unique_shifts, counts = np.unique(
    shifts,
    return_counts=True,
)


mode_index = int(
    np.argmax(
        counts
    )
)


mode_shift = int(
    unique_shifts[
        mode_index
    ]
)


mode_count = int(
    counts[
        mode_index
    ]
)


print()
print("=" * 120)
print("ORDERED-PHASE DIAGNOSTIC")
print("=" * 120)
print()


print(
    f"Mean best adjacent correlation: "
    f"{np.mean(correlations):.4f}"
)


print(
    f"Median best adjacent correlation: "
    f"{np.median(correlations):.4f}"
)


print(
    f"Minimum best adjacent correlation: "
    f"{np.min(correlations):.4f}"
)


print(
    f"Most common adjacent shift: "
    f"{mode_shift:+d} EPG columns"
)


print(
    f"Pairs using that shift: "
    f"{mode_count}/{len(shifts)}"
)


print()

print(
    "C9 -> C1 wrap:"
)

print(
    f"  unshifted correlation: "
    f"{wrap_unshifted:.4f}"
)

print(
    f"  best shift: "
    f"{wrap_shift:+d}"
)

print(
    f"  best correlation: "
    f"{wrap_corr:.4f}"
)


# =========================================================
# Full pairwise best-correlation matrix
# =========================================================

correlation_matrix = pd.DataFrame(
    index=[
        f"C{x}"
        for x in goal_columns
    ],
    columns=[
        f"C{x}"
        for x in goal_columns
    ],
    dtype=float,
)


shift_matrix = pd.DataFrame(
    index=[
        f"C{x}"
        for x in goal_columns
    ],
    columns=[
        f"C{x}"
        for x in goal_columns
    ],
    dtype=float,
)


for goal_a in goal_columns:

    for goal_b in goal_columns:

        first = (
            matrix[
                goal_a
            ]
            .to_numpy(
                dtype=float
            )
        )

        second = (
            matrix[
                goal_b
            ]
            .to_numpy(
                dtype=float
            )
        )


        (
            shift,
            correlation,
        ) = best_circular_alignment(
            first,
            second,
        )


        correlation_matrix.loc[
            f"C{goal_a}",
            f"C{goal_b}",
        ] = correlation


        shift_matrix.loc[
            f"C{goal_a}",
            f"C{goal_b}",
        ] = shift


print()
print("=" * 120)
print("PAIRWISE BEST CORRELATION MATRIX")
print("=" * 120)
print()


print(
    correlation_matrix.to_string(
        float_format=lambda x:
        f"{x:.3f}"
    )
)


print()
print("=" * 120)
print("PAIRWISE BEST SHIFT MATRIX")
print("=" * 120)
print()


print(
    shift_matrix.to_string(
        float_format=lambda x:
        f"{x:+.0f}"
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
    GOAL_FILE
)

print(
    PAIR_FILE
)