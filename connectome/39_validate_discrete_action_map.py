from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "frozen_fc2_steering_matrix.csv"

COVERAGE_FILE = OUTPUT_DIR / "discrete_goal_action_coverage.csv"
POLICY_FILE = OUTPUT_DIR / "discrete_goal_action_policy.csv"


# Engineering action levels for the later Webots controller.
# Positive/negative sign will be mapped to rover turning direction
# explicitly during Webots integration.
ACTION_TARGETS = {
    "strong_negative": -0.60,
    "mild_negative": -0.25,
    "neutral": 0.00,
    "mild_positive": +0.25,
    "strong_positive": +0.60,
}

LEFT_RIGHT_THRESHOLD = 0.05
NEUTRAL_THRESHOLD = 0.12


def load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen steering matrix not found:\n{path}\n\n"
            "Expected output from Step 27."
        )

    raw = pd.read_csv(path)

    # Long format support.
    long_required = {
        "heading_column",
        "goal_column",
        "steering",
    }

    if long_required.issubset(raw.columns):
        tmp = raw.copy()

        def goal_name(value):
            text = str(value).strip()
            if text.upper().startswith("C"):
                return text.upper()
            return f"C{int(float(text))}"

        tmp["goal_name"] = tmp["goal_column"].apply(goal_name)

        matrix = tmp.pivot_table(
            index="heading_column",
            columns="goal_name",
            values="steering",
            aggfunc="first",
        )

    else:
        # Wide format support.
        data = raw.copy()

        # Remove typical CSV index column if present.
        unnamed = [
            column
            for column in data.columns
            if str(column).lower().startswith("unnamed")
        ]

        if unnamed:
            data = data.drop(columns=unnamed)

        heading_candidates = [
            column
            for column in [
                "heading_column",
                "heading",
                "epg_column",
            ]
            if column in data.columns
        ]

        if heading_candidates:
            heading_column = heading_candidates[0]

        else:
            # Fall back to the first non-goal column.
            goal_like = {
                f"C{i}"
                for i in range(1, 10)
            }

            non_goal_columns = [
                column
                for column in data.columns
                if str(column).upper() not in goal_like
            ]

            if not non_goal_columns:
                raise RuntimeError(
                    "Could not identify heading column in "
                    "frozen steering matrix."
                )

            heading_column = non_goal_columns[0]

        data[heading_column] = (
            data[heading_column]
            .astype(str)
            .str.strip()
        )

        normalized_goal_columns = {}

        for column in data.columns:
            if column == heading_column:
                continue

            text = str(column).strip().upper()

            if text.startswith("C") and text[1:].isdigit():
                number = int(text[1:])
                if 1 <= number <= 9:
                    normalized_goal_columns[column] = f"C{number}"

            elif text.isdigit():
                number = int(text)
                if 1 <= number <= 9:
                    normalized_goal_columns[column] = f"C{number}"

        if len(normalized_goal_columns) != 9:
            raise RuntimeError(
                "Expected 9 FC2 goal columns C1..C9, "
                f"found {len(normalized_goal_columns)}:\n"
                f"{normalized_goal_columns}"
            )

        data = data.rename(
            columns=normalized_goal_columns
        )

        matrix = data.set_index(
            heading_column
        )[
            [
                f"C{i}"
                for i in range(1, 10)
            ]
        ]

    matrix = matrix[
        [
            f"C{i}"
            for i in range(1, 10)
        ]
    ]

    matrix = matrix.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if matrix.isna().any().any():
        raise RuntimeError(
            "Frozen steering matrix contains missing/non-numeric values."
        )

    if len(matrix) != 16:
        raise RuntimeError(
            f"Expected 16 EPG heading rows, got {len(matrix)}."
        )

    return matrix


matrix = load_matrix(INPUT_FILE)


print("=" * 120)
print("STEP 39 - DISCRETE NEURAL ACTION MAP AUDIT")
print("=" * 120)
print()
print(f"Input: {INPUT_FILE}")
print(f"Heading states: {len(matrix)}")
print(f"FC2 goal states per heading: {matrix.shape[1]}")
print()


coverage_rows = []
policy_rows = []


for heading, row in matrix.iterrows():
    values = row.to_numpy(dtype=float)
    goal_columns = list(row.index)

    min_index = int(np.argmin(values))
    max_index = int(np.argmax(values))
    neutral_index = int(np.argmin(np.abs(values)))

    min_value = float(values[min_index])
    max_value = float(values[max_index])
    neutral_value = float(values[neutral_index])

    has_negative = bool(
        np.any(values <= -LEFT_RIGHT_THRESHOLD)
    )

    has_positive = bool(
        np.any(values >= +LEFT_RIGHT_THRESHOLD)
    )

    has_neutral = bool(
        np.any(np.abs(values) <= NEUTRAL_THRESHOLD)
    )

    coverage_rows.append(
        {
            "heading_column": heading,
            "min_steering": min_value,
            "min_goal_column": goal_columns[min_index],
            "max_steering": max_value,
            "max_goal_column": goal_columns[max_index],
            "closest_to_zero": neutral_value,
            "neutral_goal_column": goal_columns[neutral_index],
            "steering_span": max_value - min_value,
            "has_negative_action": has_negative,
            "has_positive_action": has_positive,
            "has_neutral_action": has_neutral,
        }
    )

    for action_name, target in ACTION_TARGETS.items():
        best_index = int(
            np.argmin(
                np.abs(
                    values - target
                )
            )
        )

        selected_steering = float(
            values[best_index]
        )

        policy_rows.append(
            {
                "heading_column": heading,
                "action_name": action_name,
                "target_steering": target,
                "selected_goal_column": goal_columns[best_index],
                "neural_steering": selected_steering,
                "absolute_action_error": abs(
                    selected_steering - target
                ),
            }
        )


coverage = pd.DataFrame(
    coverage_rows
)

policy = pd.DataFrame(
    policy_rows
)


coverage.to_csv(
    COVERAGE_FILE,
    index=False,
)

policy.to_csv(
    POLICY_FILE,
    index=False,
)


print("=" * 140)
print("PER-HEADING CONTROL COVERAGE")
print("=" * 140)
print()

print(
    coverage.to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.4f}",
    )
)


print()
print("=" * 140)
print("CONTROL COVERAGE SUMMARY")
print("=" * 140)
print()

full_bidirectional = int(
    (
        coverage["has_negative_action"]
        &
        coverage["has_positive_action"]
    ).sum()
)

neutral_count = int(
    coverage["has_neutral_action"].sum()
)

print(
    f"Headings with both negative and positive steering: "
    f"{full_bidirectional}/{len(coverage)}"
)

print(
    f"Headings with a near-neutral action "
    f"(|steering| <= {NEUTRAL_THRESHOLD:.2f}): "
    f"{neutral_count}/{len(coverage)}"
)

print(
    f"Mean steering span: "
    f"{coverage['steering_span'].mean():.4f}"
)

print(
    f"Worst heading steering span: "
    f"{coverage['steering_span'].min():.4f}"
)


action_summary = (
    policy
    .groupby("action_name")
    .agg(
        mean_absolute_error=(
            "absolute_action_error",
            "mean",
        ),
        max_absolute_error=(
            "absolute_action_error",
            "max",
        ),
        mean_selected_steering=(
            "neural_steering",
            "mean",
        ),
    )
    .reset_index()
)


print()
print("=" * 140)
print("DISCRETE ACTION APPROXIMATION")
print("=" * 140)
print()

print(
    action_summary.to_string(
        index=False,
        float_format=lambda value:
            f"{value:+.4f}",
    )
)


ready_for_webots = bool(
    full_bidirectional == len(coverage)
    and
    neutral_count == len(coverage)
)


print()
print("=" * 140)
print("STEP 39 DECISION")
print("=" * 140)
print()

if ready_for_webots:
    print(
        "PASS: every heading has negative, positive, and near-neutral "
        "connectome-derived steering actions."
    )
    print(
        "NEXT: build the Webots goal-navigation controller using this "
        "discrete neural action map plus motor smoothing/hysteresis."
    )
else:
    print(
        "REVIEW: some heading rows do not provide complete discrete "
        "steering coverage."
    )
    print(
        "Do not tune neural dynamics yet. Inspect the missing heading/action "
        "coverage before Webots integration."
    )


print()
print("=" * 120)
print("FILES CREATED")
print("=" * 120)
print(COVERAGE_FILE)
print(POLICY_FILE)
