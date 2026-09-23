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
    / "calibrated_navigation_map.csv"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "dna_decoder_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "dna_decoder_summary.csv"
)


# =========================================================
# Load existing 16 x 9 neural output
# =========================================================

df = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "heading_column",
    "goal_column",
    "DNa02_L_spikes",
    "DNa02_R_spikes",
    "DNa02_L_v",
    "DNa02_R_v",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# =========================================================
# Anatomical heading order
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


heading_rank = {
    heading: index
    for index, heading
    in enumerate(EPG_ORDER)
}


df[
    "heading_rank"
] = df[
    "heading_column"
].map(
    heading_rank
)


df = (
    df
    .sort_values(
        [
            "goal_column",
            "heading_rank",
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# Raw DNa signals
# =========================================================

left_spikes = (
    df[
        "DNa02_L_spikes"
    ].astype(float)
)

right_spikes = (
    df[
        "DNa02_R_spikes"
    ].astype(float)
)


left_v = (
    df[
        "DNa02_L_v"
    ].astype(float)
)

right_v = (
    df[
        "DNa02_R_v"
    ].astype(float)
)


membrane_difference = (
    right_v
    -
    left_v
)


spike_count_difference = (
    right_spikes
    -
    left_spikes
)


spike_total = (
    right_spikes
    +
    left_spikes
)


# =========================================================
# Decoder 1
#
# OLD decoder from previous experiments.
#
# Included only as reference.
# =========================================================

spike_ratio = np.where(
    spike_total > 0,

    spike_count_difference
    /
    spike_total,

    0.0,
)


old_decoder = np.where(
    spike_total > 0,

    (
        0.80
        * spike_ratio
        +
        0.20
        * membrane_difference
    ),

    membrane_difference,
)


# =========================================================
# Decoder 2
#
# Membrane-only.
#
# This is continuous but ignores spike recruitment.
# =========================================================

membrane_only = (
    membrane_difference
)


# =========================================================
# Decoder 3
#
# Soft spike-count difference.
#
# Unlike:
#
#     (R-L)/(R+L)
#
# a single asymmetric spike cannot instantly become ±1.
#
# The number 3 is an engineering scale, not biological.
# =========================================================

soft_spike = np.tanh(
    spike_count_difference
    / 3.0
)


# =========================================================
# Decoder 4
#
# Continuous DNa activity.
#
# Instead of treating spikes as an all-or-nothing
# direction switch, represent each DNa02 side using:
#
#     membrane state
#       +
#     softly scaled spike recruitment
#
# The spike scale is estimated from THIS neural-output
# dataset only as a normalization constant.
# =========================================================

nonzero_counts = np.concatenate(
    [
        left_spikes[
            left_spikes > 0
        ].to_numpy(),

        right_spikes[
            right_spikes > 0
        ].to_numpy(),
    ]
)


if len(
    nonzero_counts
) > 0:

    SPIKE_SCALE = max(
        float(
            np.percentile(
                nonzero_counts,
                90,
            )
        ),
        1.0,
    )

else:

    SPIKE_SCALE = 1.0


left_activity = (
    left_v
    +
    left_spikes
    / SPIKE_SCALE
)


right_activity = (
    right_v
    +
    right_spikes
    / SPIKE_SCALE
)


activity_difference = (
    right_activity
    -
    left_activity
)


# =========================================================
# Robust activity normalization
# =========================================================

abs_activity = np.abs(
    activity_difference
)


ACTIVITY_SCALE = max(
    float(
        np.percentile(
            abs_activity,
            90,
        )
    ),
    0.05,
)


soft_activity = np.tanh(
    activity_difference
    / ACTIVITY_SCALE
)


# =========================================================
# Decoder 5
#
# Softer activity decoder.
#
# Same signal, but twice the normalization scale.
# This lets us check whether the main soft-activity
# decoder is still too aggressive.
# =========================================================

soft_activity_wide = np.tanh(
    activity_difference
    /
    (
        2.0
        * ACTIVITY_SCALE
    )
)


# =========================================================
# Save per-condition outputs
# =========================================================

df[
    "membrane_difference"
] = membrane_difference


df[
    "spike_count_difference"
] = spike_count_difference


df[
    "old_80_20"
] = old_decoder


df[
    "membrane_only"
] = membrane_only


df[
    "soft_spike"
] = soft_spike


df[
    "soft_activity"
] = soft_activity


df[
    "soft_activity_wide"
] = soft_activity_wide


df.to_csv(
    RESULT_FILE,
    index=False,
)


# =========================================================
# Decoder definitions
# =========================================================

DECODERS = [
    "old_80_20",
    "membrane_only",
    "soft_spike",
    "soft_activity",
    "soft_activity_wide",
]


# =========================================================
# Circular adjacent-heading smoothness
#
# For each FC2 goal column:
#
# L1 -> L2 -> ... -> R1 -> back to L1
#
# We measure how abruptly steering changes between
# adjacent anatomical heading columns.
# =========================================================

def adjacent_jumps(
    subset,
    decoder,
):

    subset = (
        subset
        .sort_values(
            "heading_rank"
        )
    )


    values = (
        subset[
            decoder
        ]
        .to_numpy(
            dtype=float
        )
    )


    if len(values) <= 1:

        return np.asarray(
            [],
            dtype=float,
        )


    # Circular ring.
    next_values = np.roll(
        values,
        -1,
    )


    return np.abs(
        next_values
        -
        values
    )


# =========================================================
# Sign transitions
#
# This is descriptive only.
#
# We are NOT claiming what the biologically correct
# number of transitions should be yet.
# =========================================================

def sign_symbol(
    value,
    threshold=0.05,
):

    if value > threshold:
        return 1

    if value < -threshold:
        return -1

    return 0


def count_sign_transitions(
    subset,
    decoder,
):

    subset = (
        subset
        .sort_values(
            "heading_rank"
        )
    )


    signs = np.asarray(
        [
            sign_symbol(x)
            for x in subset[
                decoder
            ].to_numpy()
        ],
        dtype=int,
    )


    if len(signs) <= 1:

        return 0


    # Ignore neutral points when counting left/right
    # transitions.
    signs = signs[
        signs != 0
    ]


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


        if (
            current
            != following
        ):
            transitions += 1


    return transitions


# =========================================================
# Summaries
# =========================================================

summary_rows = []


for decoder in DECODERS:

    values = (
        df[
            decoder
        ].to_numpy(
            dtype=float
        )
    )


    all_jumps = []
    transition_counts = []


    for goal in sorted(
        df[
            "goal_column"
        ].unique()
    ):

        subset = df[
            df[
                "goal_column"
            ]
            == goal
        ]


        jumps = adjacent_jumps(
            subset,
            decoder,
        )


        all_jumps.extend(
            jumps.tolist()
        )


        transition_counts.append(
            count_sign_transitions(
                subset,
                decoder,
            )
        )


    all_jumps = np.asarray(
        all_jumps,
        dtype=float,
    )


    saturated = int(
        (
            np.abs(values)
            > 0.75
        ).sum()
    )


    strong = int(
        (
            np.abs(values)
            > 0.50
        ).sum()
    )


    neutral = int(
        (
            np.abs(values)
            < 0.05
        ).sum()
    )


    summary_rows.append(
        {
            "decoder":
                decoder,

            "mean_abs_output":
                float(
                    np.mean(
                        np.abs(values)
                    )
                ),

            "max_abs_output":
                float(
                    np.max(
                        np.abs(values)
                    )
                ),

            "saturated_conditions":
                saturated,

            "strong_conditions":
                strong,

            "neutral_conditions":
                neutral,

            "mean_adjacent_jump":
                float(
                    np.mean(
                        all_jumps
                    )
                ),

            "median_adjacent_jump":
                float(
                    np.median(
                        all_jumps
                    )
                ),

            "max_adjacent_jump":
                float(
                    np.max(
                        all_jumps
                    )
                ),

            "mean_sign_transitions_per_goal":
                float(
                    np.mean(
                        transition_counts
                    )
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# =========================================================
# Print normalization information
# =========================================================

print("=" * 120)
print("DNa DECODER COMPARISON")
print("=" * 120)

print()

print(
    f"Conditions: "
    f"{len(df)}"
)

print(
    f"Spike normalization scale: "
    f"{SPIKE_SCALE:.3f}"
)

print(
    f"Activity normalization scale: "
    f"{ACTIVITY_SCALE:.5f}"
)


# =========================================================
# Summary
# =========================================================

print()
print("=" * 150)
print("DECODER SUMMARY")
print("=" * 150)
print()


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}",
    )
)


# =========================================================
# Example matrices
# =========================================================

for decoder in DECODERS:

    matrix = (
        df
        .pivot(
            index="heading_column",
            columns="goal_column",
            values=decoder,
        )
        .reindex(
            EPG_ORDER
        )
    )


    matrix.columns = [
        f"C{x}"
        for x in matrix.columns
    ]


    print()
    print("=" * 120)
    print(
        decoder.upper()
    )
    print("=" * 120)
    print()


    print(
        matrix.to_string(
            float_format=lambda x:
            f"{x:+.3f}"
        )
    )


# =========================================================
# Largest old-decoder jumps
# =========================================================

comparison = df[
    [
        "heading_column",
        "goal_column",
        "DNa02_L_spikes",
        "DNa02_R_spikes",
        "DNa02_L_v",
        "DNa02_R_v",
        "old_80_20",
        "soft_activity",
        "soft_activity_wide",
    ]
].copy()


comparison[
    "old_abs"
] = (
    comparison[
        "old_80_20"
    ].abs()
)


print()
print("=" * 150)
print("CONDITIONS WITH STRONGEST OLD DECODER OUTPUT")
print("=" * 150)
print()


print(
    comparison
    .sort_values(
        "old_abs",
        ascending=False,
    )
    .head(20)
    .drop(
        columns=[
            "old_abs",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x:
        f"{x:+.5f}",
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
    RESULT_FILE
)

print(
    SUMMARY_FILE
)