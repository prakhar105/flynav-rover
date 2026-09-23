import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

STEERING_NEURON_FILE = (
    OUTPUT_DIR
    / "steering_core_neurons.csv"
)

STEERING_EDGE_FILE = (
    OUTPUT_DIR
    / "steering_core_edges.csv"
)

DELTA7_NEURON_FILE = (
    OUTPUT_DIR
    / "delta7_candidates.csv"
)

DELTA7_EDGE_FILE = (
    OUTPUT_DIR
    / "delta7_bridge_edges.csv"
)

STEP31_RESULT_FILE = (
    OUTPUT_DIR
    / "fc2_anatomically_balanced_recruitment.csv"
)

STEP33_TRANSITION_FILE = (
    OUTPUT_DIR
    / "lif_discontinuity_transition_metrics.csv"
)


CONDITION_FILE = (
    OUTPUT_DIR
    / "fc2_projection_continuity_conditions.csv"
)

TRANSITION_FILE = (
    OUTPUT_DIR
    / "fc2_projection_continuity_transitions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "fc2_projection_continuity_summary.csv"
)


# =========================================================
# Load inputs
# =========================================================

for required_file in [
    STEERING_NEURON_FILE,
    STEERING_EDGE_FILE,
    DELTA7_NEURON_FILE,
    DELTA7_EDGE_FILE,
    STEP31_RESULT_FILE,
    STEP33_TRANSITION_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


steering_neurons = pd.read_csv(
    STEERING_NEURON_FILE
)

steering_edges = pd.read_csv(
    STEERING_EDGE_FILE
)

delta7_neurons = pd.read_csv(
    DELTA7_NEURON_FILE
)

delta7_edges = pd.read_csv(
    DELTA7_EDGE_FILE
)

step31 = pd.read_csv(
    STEP31_RESULT_FILE
)

step33 = pd.read_csv(
    STEP33_TRANSITION_FILE
)


# =========================================================
# Validate Step 31
# =========================================================

required_step31_columns = [
    "heading_column",
    "column_a",
    "column_b",
    "alpha",
    "a_body_ids",
    "b_body_ids",
    "steering",
]


missing = [
    column
    for column in required_step31_columns
    if column not in step31.columns
]


if missing:

    raise RuntimeError(
        "Step 31 file is missing columns: "
        f"{missing}"
    )


if len(step31) != 396:

    raise RuntimeError(
        f"Expected 396 Step 31 conditions, "
        f"got {len(step31)}."
    )


# =========================================================
# Normalize data
# =========================================================

for dataframe in [
    steering_neurons,
    delta7_neurons,
]:

    dataframe["bodyId"] = (
        dataframe["bodyId"]
        .astype("int64")
    )


for dataframe in [
    steering_edges,
    delta7_edges,
]:

    dataframe["source_bodyId"] = (
        dataframe["source_bodyId"]
        .astype("int64")
    )

    dataframe["target_bodyId"] = (
        dataframe["target_bodyId"]
        .astype("int64")
    )

    dataframe["weight"] = pd.to_numeric(
        dataframe["weight"],
        errors="coerce",
    ).fillna(0.0)


# =========================================================
# Build same neuron table used by frozen network
# =========================================================

NETWORK_TYPES = [
    "EPG",
    "Delta7",
    "FC2A",
    "FC2B",
    "FC2C",
    "PFL2",
    "PFL3",
    "DNa03",
    "DNa02",
]


neurons = pd.concat(
    [
        steering_neurons,
        delta7_neurons,
    ],
    ignore_index=True,
)


neurons = (
    neurons
    .drop_duplicates(
        subset=[
            "bodyId",
        ]
    )
)


neurons = (
    neurons[
        neurons["type"]
        .isin(
            NETWORK_TYPES
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(neurons) != 220:

    raise RuntimeError(
        f"Expected 220 neurons, "
        f"got {len(neurons)}."
    )


body_ids = set(
    neurons[
        "bodyId"
    ]
)


# =========================================================
# Combine / deduplicate edges exactly like frozen network
# =========================================================

edges = pd.concat(
    [
        steering_edges,
        delta7_edges,
    ],
    ignore_index=True,
)


edges = edges[
    edges[
        "source_bodyId"
    ].isin(
        body_ids
    )
    &
    edges[
        "target_bodyId"
    ].isin(
        body_ids
    )
].copy()


edges = (
    edges
    .sort_values(
        "weight",
        ascending=False,
    )
    .drop_duplicates(
        subset=[
            "source_bodyId",
            "target_bodyId",
        ],
        keep="first",
    )
)


# =========================================================
# FC2 parser
# =========================================================

def parse_fc2_column(
    instance,
):

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
        match.group(
            1
        )
    )


    if not (
        1 <= column <= 9
    ):

        return None


    return column


# =========================================================
# FC2 body IDs by biological column
# =========================================================

fc2_body_ids_by_column = {
    column: []
    for column
    in range(
        1,
        10,
    )
}


fc2_body_to_column = {}


for _, row in neurons.iterrows():

    if row[
        "type"
    ] not in [
        "FC2A",
        "FC2B",
        "FC2C",
    ]:

        continue


    column = parse_fc2_column(
        row[
            "instance"
        ]
    )


    if column is None:

        continue


    body_id = int(
        row[
            "bodyId"
        ]
    )


    fc2_body_ids_by_column[
        column
    ].append(
        body_id
    )


    fc2_body_to_column[
        body_id
    ] = column


for column in fc2_body_ids_by_column:

    fc2_body_ids_by_column[
        column
    ] = sorted(
        fc2_body_ids_by_column[
            column
        ]
    )


fc2_count = sum(
    len(
        values
    )
    for values
    in fc2_body_ids_by_column.values()
)


if fc2_count != 92:

    raise RuntimeError(
        f"Expected 92 FC2 neurons, "
        f"got {fc2_count}."
    )


# =========================================================
# PFL target inventory
# =========================================================

pfl_rows = neurons[
    neurons[
        "type"
    ].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
].copy()


pfl_body_ids = [
    int(
        body_id
    )
    for body_id
    in pfl_rows[
        "bodyId"
    ]
]


pfl_body_to_position = {
    body_id:
        position
    for position, body_id
    in enumerate(
        pfl_body_ids
    )
}


pfl2_positions = []
pfl3_positions = []


for position, (_, row) in enumerate(
    pfl_rows.iterrows()
):

    if row[
        "type"
    ] == "PFL2":

        pfl2_positions.append(
            position
        )


    elif row[
        "type"
    ] == "PFL3":

        pfl3_positions.append(
            position
        )


pfl2_positions = np.asarray(
    pfl2_positions,
    dtype=int,
)

pfl3_positions = np.asarray(
    pfl3_positions,
    dtype=int,
)


print("=" * 120)
print(
    "STEP 35 - FC2 -> PFL PROJECTION CONTINUITY"
)
print("=" * 120)

print()

print(
    f"FC2 neurons: "
    f"{fc2_count}"
)

print(
    f"PFL targets: "
    f"{len(pfl_body_ids)}"
)

print(
    f"PFL2 targets: "
    f"{len(pfl2_positions)}"
)

print(
    f"PFL3 targets: "
    f"{len(pfl3_positions)}"
)


# =========================================================
# Extract only FC2 -> PFL anatomical edges
# =========================================================

fc2_pfl_edges = edges[
    edges[
        "source_type"
    ].isin(
        [
            "FC2A",
            "FC2B",
            "FC2C",
        ]
    )
    &
    edges[
        "target_type"
    ].isin(
        [
            "PFL2",
            "PFL3",
        ]
    )
].copy()


print(
    f"FC2 -> PFL anatomical edges: "
    f"{len(fc2_pfl_edges)}"
)


# =========================================================
# Raw projection vector for every individual FC2 neuron
# =========================================================

projection_by_fc2_body = {}


for body_id in fc2_body_to_column:

    vector = np.zeros(
        len(
            pfl_body_ids
        ),
        dtype=float,
    )


    outgoing = fc2_pfl_edges[
        fc2_pfl_edges[
            "source_bodyId"
        ]
        ==
        body_id
    ]


    for _, row in outgoing.iterrows():

        target_body_id = int(
            row[
                "target_bodyId"
            ]
        )


        if (
            target_body_id
            not in
            pfl_body_to_position
        ):

            continue


        position = (
            pfl_body_to_position[
                target_body_id
            ]
        )


        vector[
            position
        ] += float(
            row[
                "weight"
            ]
        )


    projection_by_fc2_body[
        body_id
    ] = vector


# =========================================================
# Full projection for each C-column
# =========================================================

full_column_projection = {}


for column in range(
    1,
    10,
):

    vectors = [
        projection_by_fc2_body[
            body_id
        ]
        for body_id
        in fc2_body_ids_by_column[
            column
        ]
    ]


    full_column_projection[
        column
    ] = np.sum(
        np.stack(
            vectors
        ),
        axis=0,
    )


# =========================================================
# Helpers
# =========================================================

def parse_body_ids(
    value,
):

    if pd.isna(
        value
    ):

        return []


    text = str(
        value
    ).strip()


    if not text:

        return []


    output = []


    for piece in text.split(
        ","
    ):

        piece = piece.strip()


        if not piece:

            continue


        output.append(
            int(
                piece
            )
        )


    return output


def projection_for_body_ids(
    body_ids_list,
):

    if not body_ids_list:

        return np.zeros(
            len(
                pfl_body_ids
            ),
            dtype=float,
        )


    return np.sum(
        np.stack(
            [
                projection_by_fc2_body[
                    body_id
                ]
                for body_id
                in body_ids_list
            ]
        ),
        axis=0,
    )


def normalized_error(
    actual,
    expected,
):

    denominator = max(
        float(
            np.linalg.norm(
                expected
            )
        ),
        1e-12,
    )


    return (
        float(
            np.linalg.norm(
                actual
                -
                expected
            )
        )
        /
        denominator
    )


# =========================================================
# Build condition-level anatomical diagnostics
# =========================================================

condition_rows = []

condition_vectors = []


for _, source_row in step31.iterrows():

    heading = str(
        source_row[
            "heading_column"
        ]
    )


    column_a = int(
        source_row[
            "column_a"
        ]
    )


    column_b = int(
        source_row[
            "column_b"
        ]
    )


    alpha = float(
        source_row[
            "alpha"
        ]
    )


    a_body_ids = parse_body_ids(
        source_row[
            "a_body_ids"
        ]
    )


    b_body_ids = parse_body_ids(
        source_row[
            "b_body_ids"
        ]
    )


    active_body_ids = sorted(
        set(
            a_body_ids
            +
            b_body_ids
        )
    )


    actual_projection = (
        projection_for_body_ids(
            active_body_ids
        )
    )


    # -----------------------------------------------------
    # Ideal continuous anatomical blend
    #
    # This is NOT a claim that alpha is an angle.
    #
    # It only represents the projection we would expect
    # from a mathematically smooth interpolation between
    # the two complete FC2 populations.
    # -----------------------------------------------------

    ideal_projection = (

        (
            1.0
            -
            alpha
        )
        *
        full_column_projection[
            column_a
        ]

        +

        alpha
        *
        full_column_projection[
            column_b
        ]
    )


    actual_pfl2 = (
        actual_projection[
            pfl2_positions
        ]
    )

    actual_pfl3 = (
        actual_projection[
            pfl3_positions
        ]
    )


    ideal_pfl2 = (
        ideal_projection[
            pfl2_positions
        ]
    )

    ideal_pfl3 = (
        ideal_projection[
            pfl3_positions
        ]
    )


    condition_id = len(
        condition_rows
    )


    condition_rows.append(
        {

            "condition_id":
                condition_id,

            "heading_column":
                heading,

            "column_a":
                column_a,

            "column_b":
                column_b,

            "alpha":
                alpha,

            "active_count":
                len(
                    active_body_ids
                ),

            "a_count":
                len(
                    a_body_ids
                ),

            "b_count":
                len(
                    b_body_ids
                ),

            "active_body_ids":
                ",".join(
                    str(
                        body_id
                    )
                    for body_id
                    in active_body_ids
                ),

            "actual_projection_norm":
                float(
                    np.linalg.norm(
                        actual_projection
                    )
                ),

            "ideal_projection_norm":
                float(
                    np.linalg.norm(
                        ideal_projection
                    )
                ),

            "actual_vs_ideal_error_all":
                normalized_error(
                    actual_projection,
                    ideal_projection,
                ),

            "actual_vs_ideal_error_PFL2":
                normalized_error(
                    actual_pfl2,
                    ideal_pfl2,
                ),

            "actual_vs_ideal_error_PFL3":
                normalized_error(
                    actual_pfl3,
                    ideal_pfl3,
                ),

            "steering":
                float(
                    source_row[
                        "steering"
                    ]
                ),
        }
    )


    condition_vectors.append(
        {
            "active_set":
                set(
                    active_body_ids
                ),

            "actual_all":
                actual_projection,

            "ideal_all":
                ideal_projection,

            "actual_PFL2":
                actual_pfl2,

            "ideal_PFL2":
                ideal_pfl2,

            "actual_PFL3":
                actual_pfl3,

            "ideal_PFL3":
                ideal_pfl3,
        }
    )


conditions = pd.DataFrame(
    condition_rows
)


conditions.to_csv(
    CONDITION_FILE,
    index=False,
)


# =========================================================
# Characteristic anatomical scales
#
# p95 norm across condition vectors.
# =========================================================

projection_scales = {}


for vector_name in [
    "actual_all",
    "actual_PFL2",
    "actual_PFL3",
]:

    norms = np.asarray(
        [
            np.linalg.norm(
                vectors[
                    vector_name
                ]
            )
            for vectors
            in condition_vectors
        ],
        dtype=float,
    )


    scale = float(
        np.percentile(
            norms,
            95.0,
        )
    )


    if scale < 1e-12:

        scale = float(
            norms.max()
        )


    if scale < 1e-12:

        scale = 1.0


    projection_scales[
        vector_name
    ] = scale


# =========================================================
# Step 33 lookup
# =========================================================

step33_lookup = {}


for _, row in step33.iterrows():

    key = (
        str(
            row[
                "heading_column"
            ]
        ),
        int(
            row[
                "column_a"
            ]
        ),
        int(
            row[
                "column_b"
            ]
        ),
        round(
            float(
                row[
                    "alpha_from"
                ]
            ),
            6,
        ),
        round(
            float(
                row[
                    "alpha_to"
                ]
            ),
            6,
        ),
    )


    step33_lookup[
        key
    ] = row


# =========================================================
# Build adjacent-alpha transitions
# =========================================================

transition_rows = []


for (
    heading,
    column_a,
    column_b,
), group in conditions.groupby(
    [
        "heading_column",
        "column_a",
        "column_b",
    ]
):

    group = (
        group
        .sort_values(
            "alpha"
        )
        .reset_index(
            drop=True
        )
    )


    for index in range(
        len(
            group
        )
        -
        1
    ):

        first = group.iloc[
            index
        ]

        second = group.iloc[
            index + 1
        ]


        first_id = int(
            first[
                "condition_id"
            ]
        )

        second_id = int(
            second[
                "condition_id"
            ]
        )


        first_vectors = (
            condition_vectors[
                first_id
            ]
        )

        second_vectors = (
            condition_vectors[
                second_id
            ]
        )


        first_set = (
            first_vectors[
                "active_set"
            ]
        )

        second_set = (
            second_vectors[
                "active_set"
            ]
        )


        removed = (
            first_set
            -
            second_set
        )


        added = (
            second_set
            -
            first_set
        )


        symmetric_difference = (
            first_set
            ^
            second_set
        )


        union = (
            first_set
            |
            second_set
        )


        intersection = (
            first_set
            &
            second_set
        )


        jaccard = (

            len(
                intersection
            )
            /
            len(
                union
            )

            if len(
                union
            ) > 0

            else 1.0
        )


        record = {

            "heading_column":
                heading,

            "column_a":
                int(
                    column_a
                ),

            "column_b":
                int(
                    column_b
                ),

            "alpha_from":
                float(
                    first[
                        "alpha"
                    ]
                ),

            "alpha_to":
                float(
                    second[
                        "alpha"
                    ]
                ),

            "count_from":
                int(
                    first[
                        "active_count"
                    ]
                ),

            "count_to":
                int(
                    second[
                        "active_count"
                    ]
                ),

            "count_change":
                int(
                    second[
                        "active_count"
                    ]
                    -
                    first[
                        "active_count"
                    ]
                ),

            "removed_neurons":
                len(
                    removed
                ),

            "added_neurons":
                len(
                    added
                ),

            "membership_churn":
                len(
                    symmetric_difference
                ),

            "jaccard_similarity":
                float(
                    jaccard
                ),

            "removed_body_ids":
                ",".join(
                    str(
                        body_id
                    )
                    for body_id
                    in sorted(
                        removed
                    )
                ),

            "added_body_ids":
                ",".join(
                    str(
                        body_id
                    )
                    for body_id
                    in sorted(
                        added
                    )
                ),

            "steering_from":
                float(
                    first[
                        "steering"
                    ]
                ),

            "steering_to":
                float(
                    second[
                        "steering"
                    ]
                ),

            "steering_abs_step":
                abs(
                    float(
                        second[
                            "steering"
                        ]
                    )
                    -
                    float(
                        first[
                            "steering"
                        ]
                    )
                ),
        }


        # =================================================
        # Anatomical projection jumps
        # =================================================

        for label in [
            "all",
            "PFL2",
            "PFL3",
        ]:

            actual_name = (
                "actual_all"
                if label == "all"
                else f"actual_{label}"
            )


            ideal_name = (
                "ideal_all"
                if label == "all"
                else f"ideal_{label}"
            )


            actual_step = float(
                np.linalg.norm(
                    second_vectors[
                        actual_name
                    ]
                    -
                    first_vectors[
                        actual_name
                    ]
                )
            )


            ideal_step = float(
                np.linalg.norm(
                    second_vectors[
                        ideal_name
                    ]
                    -
                    first_vectors[
                        ideal_name
                    ]
                )
            )


            scale_key = (
                "actual_all"
                if label == "all"
                else f"actual_{label}"
            )


            normalized_actual_step = (
                actual_step
                /
                projection_scales[
                    scale_key
                ]
            )


            normalized_ideal_step = (
                ideal_step
                /
                projection_scales[
                    scale_key
                ]
            )


            excess_ratio = (

                actual_step
                /
                ideal_step

                if ideal_step
                >
                1e-12

                else np.nan
            )


            record[
                f"actual_{label}_projection_raw_step"
            ] = actual_step


            record[
                f"actual_{label}_projection_norm_step"
            ] = normalized_actual_step


            record[
                f"ideal_{label}_projection_raw_step"
            ] = ideal_step


            record[
                f"ideal_{label}_projection_norm_step"
            ] = normalized_ideal_step


            record[
                f"{label}_projection_excess_ratio"
            ] = excess_ratio


        # =================================================
        # Attach real Step 33 PFL dynamics
        # =================================================

        key = (
            str(
                heading
            ),
            int(
                column_a
            ),
            int(
                column_b
            ),
            round(
                float(
                    first[
                        "alpha"
                    ]
                ),
                6,
            ),
            round(
                float(
                    second[
                        "alpha"
                    ]
                ),
                6,
            ),
        )


        if key in step33_lookup:

            source = (
                step33_lookup[
                    key
                ]
            )


            for source_column in [
                "FC2_input_norm_step",
                "PFL2_rate_norm_step",
                "PFL3_rate_norm_step",
                "PFL2_voltage_norm_step",
                "PFL3_voltage_norm_step",
                "DNa03_rate_norm_step",
                "DNa02_rate_norm_step",
            ]:

                record[
                    source_column
                ] = float(
                    source[
                        source_column
                    ]
                )


        transition_rows.append(
            record
        )


transitions = pd.DataFrame(
    transition_rows
)


# =========================================================
# Mark established vs exploratory adjacency
#
# C9 -> C1 was never established as biological adjacency.
# =========================================================

transitions[
    "is_exploratory_wrap"
] = (
    (
        transitions[
            "column_a"
        ]
        ==
        9
    )
    &
    (
        transitions[
            "column_b"
        ]
        ==
        1
    )
)


transitions[
    "is_primary_adjacency"
] = (
    ~transitions[
        "is_exploratory_wrap"
    ]
)


transitions.to_csv(
    TRANSITION_FILE,
    index=False,
)


# =========================================================
# Primary analysis excludes C9 -> C1
# =========================================================

primary = transitions[
    transitions[
        "is_primary_adjacency"
    ]
].copy()


wrap = transitions[
    transitions[
        "is_exploratory_wrap"
    ]
].copy()


if len(primary) != 320:

    raise RuntimeError(
        f"Expected 320 primary transitions, "
        f"got {len(primary)}."
    )


if len(wrap) != 40:

    raise RuntimeError(
        f"Expected 40 exploratory wrap transitions, "
        f"got {len(wrap)}."
    )


# =========================================================
# Correlation helper
# =========================================================

def correlation(
    dataframe,
    column_a,
    column_b,
):

    values_a = dataframe[
        column_a
    ].to_numpy(
        dtype=float,
    )


    values_b = dataframe[
        column_b
    ].to_numpy(
        dtype=float,
    )


    mask = (
        np.isfinite(
            values_a
        )
        &
        np.isfinite(
            values_b
        )
    )


    values_a = (
        values_a[
            mask
        ]
    )

    values_b = (
        values_b[
            mask
        ]
    )


    if (
        len(
            values_a
        )
        <
        3
    ):

        return np.nan


    if (
        np.std(
            values_a
        )
        <
        1e-12
        or
        np.std(
            values_b
        )
        <
        1e-12
    ):

        return np.nan


    return float(
        np.corrcoef(
            values_a,
            values_b,
        )[
            0,
            1
        ]
    )


# =========================================================
# Global summary
# =========================================================

summary_rows = []


for label, dataframe in [
    (
        "PRIMARY_C1_TO_C9",
        primary,
    ),
    (
        "EXPLORATORY_C9_TO_C1",
        wrap,
    ),
    (
        "ALL",
        transitions,
    ),
]:

    summary_rows.append(
        {

            "analysis_group":
                label,

            "transitions":
                len(
                    dataframe
                ),

            "mean_membership_churn":
                float(
                    dataframe[
                        "membership_churn"
                    ].mean()
                ),

            "median_membership_churn":
                float(
                    dataframe[
                        "membership_churn"
                    ].median()
                ),

            "max_membership_churn":
                int(
                    dataframe[
                        "membership_churn"
                    ].max()
                ),

            "transitions_churn_gt_2":
                int(
                    (
                        dataframe[
                            "membership_churn"
                        ]
                        >
                        2
                    ).sum()
                ),

            "transitions_churn_gt_4":
                int(
                    (
                        dataframe[
                            "membership_churn"
                        ]
                        >
                        4
                    ).sum()
                ),

            "mean_jaccard":
                float(
                    dataframe[
                        "jaccard_similarity"
                    ].mean()
                ),

            "mean_actual_all_projection_step":
                float(
                    dataframe[
                        "actual_all_projection_norm_step"
                    ].mean()
                ),

            "mean_ideal_all_projection_step":
                float(
                    dataframe[
                        "ideal_all_projection_norm_step"
                    ].mean()
                ),

            "mean_all_projection_excess_ratio":
                float(
                    dataframe[
                        "all_projection_excess_ratio"
                    ].replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    ).mean()
                ),

            "mean_actual_PFL2_projection_step":
                float(
                    dataframe[
                        "actual_PFL2_projection_norm_step"
                    ].mean()
                ),

            "mean_actual_PFL3_projection_step":
                float(
                    dataframe[
                        "actual_PFL3_projection_norm_step"
                    ].mean()
                ),

            "corr_churn_vs_projection_step":
                correlation(
                    dataframe,
                    "membership_churn",
                    "actual_all_projection_norm_step",
                ),

            "corr_churn_vs_steering_jump":
                correlation(
                    dataframe,
                    "membership_churn",
                    "steering_abs_step",
                ),

            "corr_projection_vs_steering":
                correlation(
                    dataframe,
                    "actual_all_projection_norm_step",
                    "steering_abs_step",
                ),

            "corr_PFL2_projection_vs_PFL2_rate":
                correlation(
                    dataframe,
                    "actual_PFL2_projection_norm_step",
                    "PFL2_rate_norm_step",
                ),

            "corr_PFL3_projection_vs_PFL3_rate":
                correlation(
                    dataframe,
                    "actual_PFL3_projection_norm_step",
                    "PFL3_rate_norm_step",
                ),

            "corr_PFL3_projection_vs_steering":
                correlation(
                    dataframe,
                    "actual_PFL3_projection_norm_step",
                    "steering_abs_step",
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
# Print subset membership continuity
# =========================================================

print()

print("=" * 150)
print(
    "FC2 MEMBERSHIP CONTINUITY - PRIMARY ADJACENCIES"
)
print("=" * 150)

print()


print(
    f"Primary adjacent-alpha transitions: "
    f"{len(primary)}"
)

print(
    f"Mean FC2 membership churn: "
    f"{primary['membership_churn'].mean():.3f} neurons"
)

print(
    f"Median FC2 membership churn: "
    f"{primary['membership_churn'].median():.3f}"
)

print(
    f"Worst FC2 membership churn: "
    f"{int(primary['membership_churn'].max())}"
)

print(
    f"Transitions changing >2 FC2 neurons: "
    f"{int((primary['membership_churn'] > 2).sum())}/"
    f"{len(primary)}"
)

print(
    f"Transitions changing >4 FC2 neurons: "
    f"{int((primary['membership_churn'] > 4).sum())}/"
    f"{len(primary)}"
)

print(
    f"Mean active-set Jaccard similarity: "
    f"{primary['jaccard_similarity'].mean():.4f}"
)


# =========================================================
# Print anatomical projection continuity
# =========================================================

print()

print("=" * 150)
print(
    "FC2 -> PFL ANATOMICAL PROJECTION CONTINUITY"
)
print("=" * 150)

print()


print(
    "Mean normalized ACTUAL projection step:"
)

print(
    f"  all PFL: "
    f"{primary['actual_all_projection_norm_step'].mean():.4f}"
)

print(
    f"  PFL2:    "
    f"{primary['actual_PFL2_projection_norm_step'].mean():.4f}"
)

print(
    f"  PFL3:    "
    f"{primary['actual_PFL3_projection_norm_step'].mean():.4f}"
)


print()

print(
    "Mean normalized IDEAL linear projection step:"
)

print(
    f"  all PFL: "
    f"{primary['ideal_all_projection_norm_step'].mean():.4f}"
)

print(
    f"  PFL2:    "
    f"{primary['ideal_PFL2_projection_norm_step'].mean():.4f}"
)

print(
    f"  PFL3:    "
    f"{primary['ideal_PFL3_projection_norm_step'].mean():.4f}"
)


print()

print(
    "Mean actual / ideal anatomical step ratio:"
)

print(
    f"  all PFL: "
    f"{primary['all_projection_excess_ratio'].mean():.3f}x"
)

print(
    f"  PFL2:    "
    f"{primary['PFL2_projection_excess_ratio'].mean():.3f}x"
)

print(
    f"  PFL3:    "
    f"{primary['PFL3_projection_excess_ratio'].mean():.3f}x"
)


# =========================================================
# Correlations
# =========================================================

print()

print("=" * 160)
print(
    "WHAT PREDICTS THE PFL / STEERING JUMPS?"
)
print("=" * 160)

print()


diagnostic_correlations = [
    (
        "membership churn -> anatomical projection",
        correlation(
            primary,
            "membership_churn",
            "actual_all_projection_norm_step",
        ),
    ),
    (
        "membership churn -> steering jump",
        correlation(
            primary,
            "membership_churn",
            "steering_abs_step",
        ),
    ),
    (
        "FC2 anatomical projection -> steering jump",
        correlation(
            primary,
            "actual_all_projection_norm_step",
            "steering_abs_step",
        ),
    ),
    (
        "FC2->PFL2 projection -> PFL2 rate jump",
        correlation(
            primary,
            "actual_PFL2_projection_norm_step",
            "PFL2_rate_norm_step",
        ),
    ),
    (
        "FC2->PFL3 projection -> PFL3 rate jump",
        correlation(
            primary,
            "actual_PFL3_projection_norm_step",
            "PFL3_rate_norm_step",
        ),
    ),
    (
        "FC2->PFL3 projection -> steering jump",
        correlation(
            primary,
            "actual_PFL3_projection_norm_step",
            "steering_abs_step",
        ),
    ),
    (
        "PFL3 rate jump -> steering jump",
        correlation(
            primary,
            "PFL3_rate_norm_step",
            "steering_abs_step",
        ),
    ),
    (
        "DNa02 rate jump -> steering jump",
        correlation(
            primary,
            "DNa02_rate_norm_step",
            "steering_abs_step",
        ),
    ),
]


for name, value in diagnostic_correlations:

    print(
        f"{name:48s}: "
        f"{value:+.4f}"
    )


# =========================================================
# Worst membership churn cases
# =========================================================

print()

print("=" * 190)
print(
    "WORST FC2 MEMBERSHIP-CHURN TRANSITIONS"
)
print("=" * 190)

print()


churn_columns = [
    "heading_column",
    "column_a",
    "column_b",
    "alpha_from",
    "alpha_to",
    "count_from",
    "count_to",
    "removed_neurons",
    "added_neurons",
    "membership_churn",
    "jaccard_similarity",
    "actual_all_projection_norm_step",
    "ideal_all_projection_norm_step",
    "all_projection_excess_ratio",
    "PFL3_rate_norm_step",
    "DNa02_rate_norm_step",
    "steering_abs_step",
]


print(
    primary
    .sort_values(
        [
            "membership_churn",
            "actual_all_projection_norm_step",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        20
    )[
        churn_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Worst anatomical projection jumps
# =========================================================

print()

print("=" * 190)
print(
    "WORST FC2 -> PFL ANATOMICAL PROJECTION JUMPS"
)
print("=" * 190)

print()


projection_columns = [
    "heading_column",
    "column_a",
    "column_b",
    "alpha_from",
    "alpha_to",
    "membership_churn",
    "jaccard_similarity",
    "actual_all_projection_norm_step",
    "ideal_all_projection_norm_step",
    "all_projection_excess_ratio",
    "actual_PFL2_projection_norm_step",
    "actual_PFL3_projection_norm_step",
    "PFL2_rate_norm_step",
    "PFL3_rate_norm_step",
    "DNa03_rate_norm_step",
    "DNa02_rate_norm_step",
    "steering_abs_step",
]


print(
    primary
    .sort_values(
        "actual_all_projection_norm_step",
        ascending=False,
    )
    .head(
        20
    )[
        projection_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Current worst steering transition
# =========================================================

worst_steering = (
    primary
    .sort_values(
        "steering_abs_step",
        ascending=False,
    )
    .iloc[
        0
    ]
)


print()

print("=" * 170)
print(
    "CURRENT PRIMARY WORST STEERING TRANSITION"
)
print("=" * 170)

print()


print(
    f"Heading: "
    f"{worst_steering['heading_column']}"
)

print(
    f"Pair: "
    f"C{int(worst_steering['column_a'])}"
    f"->"
    f"C{int(worst_steering['column_b'])}"
)

print(
    f"Alpha: "
    f"{worst_steering['alpha_from']:.1f}"
    f" -> "
    f"{worst_steering['alpha_to']:.1f}"
)

print(
    f"FC2 count: "
    f"{int(worst_steering['count_from'])}"
    f" -> "
    f"{int(worst_steering['count_to'])}"
)

print(
    f"Removed neurons: "
    f"{int(worst_steering['removed_neurons'])}"
)

print(
    f"Added neurons: "
    f"{int(worst_steering['added_neurons'])}"
)

print(
    f"Membership churn: "
    f"{int(worst_steering['membership_churn'])}"
)

print(
    f"Jaccard similarity: "
    f"{worst_steering['jaccard_similarity']:.4f}"
)

print(
    f"Actual anatomical projection step: "
    f"{worst_steering['actual_all_projection_norm_step']:.4f}"
)

print(
    f"Ideal anatomical projection step: "
    f"{worst_steering['ideal_all_projection_norm_step']:.4f}"
)

print(
    f"Actual / ideal ratio: "
    f"{worst_steering['all_projection_excess_ratio']:.3f}x"
)

print(
    f"PFL2 rate step: "
    f"{worst_steering['PFL2_rate_norm_step']:.4f}"
)

print(
    f"PFL3 rate step: "
    f"{worst_steering['PFL3_rate_norm_step']:.4f}"
)

print(
    f"DNa02 rate step: "
    f"{worst_steering['DNa02_rate_norm_step']:.4f}"
)

print(
    f"Steering jump: "
    f"{worst_steering['steering_abs_step']:.4f}"
)

print()

print(
    "Removed body IDs:"
)

print(
    worst_steering[
        "removed_body_ids"
    ]
)

print()

print(
    "Added body IDs:"
)

print(
    worst_steering[
        "added_body_ids"
    ]
)


# =========================================================
# Actual vs ideal condition errors
# =========================================================

print()

print("=" * 150)
print(
    "ACTUAL SUBSET PROJECTION VS IDEAL LINEAR PROJECTION"
)
print("=" * 150)

print()


primary_condition_mask = (
    ~(
        (
            conditions[
                "column_a"
            ]
            ==
            9
        )
        &
        (
            conditions[
                "column_b"
            ]
            ==
            1
        )
    )
)


primary_conditions = conditions[
    primary_condition_mask
]


print(
    f"Mean all-PFL relative error: "
    f"{primary_conditions['actual_vs_ideal_error_all'].mean():.4f}"
)

print(
    f"Median all-PFL relative error: "
    f"{primary_conditions['actual_vs_ideal_error_all'].median():.4f}"
)

print(
    f"Worst all-PFL relative error: "
    f"{primary_conditions['actual_vs_ideal_error_all'].max():.4f}"
)

print()

print(
    f"Mean PFL2 relative error: "
    f"{primary_conditions['actual_vs_ideal_error_PFL2'].mean():.4f}"
)

print(
    f"Mean PFL3 relative error: "
    f"{primary_conditions['actual_vs_ideal_error_PFL3'].mean():.4f}"
)


# =========================================================
# Exploratory wrap shown separately
# =========================================================

print()

print("=" * 150)
print(
    "EXPLORATORY C9 -> C1 WRAP - REPORTED SEPARATELY"
)
print("=" * 150)

print()


print(
    f"Transitions: "
    f"{len(wrap)}"
)

print(
    f"Mean membership churn: "
    f"{wrap['membership_churn'].mean():.4f}"
)

print(
    f"Mean anatomical projection step: "
    f"{wrap['actual_all_projection_norm_step'].mean():.4f}"
)

print(
    f"Worst steering jump: "
    f"{wrap['steering_abs_step'].max():.4f}"
)


# =========================================================
# Summary
# =========================================================

print()

print("=" * 180)
print(
    "GLOBAL SUMMARY"
)
print("=" * 180)

print()


print(
    summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


# =========================================================
# Files
# =========================================================

print()

print("=" * 120)
print(
    "FILES CREATED"
)
print("=" * 120)

print(
    CONDITION_FILE
)

print(
    TRANSITION_FILE
)

print(
    SUMMARY_FILE
)