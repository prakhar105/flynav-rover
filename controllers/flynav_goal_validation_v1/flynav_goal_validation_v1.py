import csv
import math
from pathlib import Path

from controller import Supervisor


# ============================================================
# STEP 41 - AUTOMATED 8-DIRECTION WEBOTS GOAL VALIDATION
#
# This controller validates the frozen connectome-derived
# discrete steering map in Webots.
#
# It automatically runs 8 goal directions around the rover:
#
#   E, NE, N, NW, W, SW, S, SE
#
# Each trial:
#   1. resets the rover to the same starting pose
#   2. places the goal at a fixed radius around that start
#   3. runs the same frozen neural action-map controller
#   4. records success/failure, distance, time, path length,
#      heading states, FC2 states and fallback usage
#
# Before running:
#   Place the E-puck roughly in the CENTER of the empty arena.
#   Keep enough free space around it for the test radius.
#
# Scientific wording:
#   connectome-derived steering map + engineered world/action
#   interface. This is not a literal simulated fly brain.
# ============================================================


TIME_STEP = 32

# ------------------------------------------------------------
# Validation geometry
# ------------------------------------------------------------

GOAL_RADIUS_M = 0.18
GOAL_TOLERANCE_M = 0.035
TRIAL_TIMEOUT_S = 20.0
SETTLE_STEPS = 8

DIRECTIONS = [
    ("E",   0.0),
    ("NE", 45.0),
    ("N",  90.0),
    ("NW", 135.0),
    ("W", 180.0),
    ("SW", 225.0),
    ("S", 270.0),
    ("SE", 315.0),
]


# ------------------------------------------------------------
# Frozen controller interface
# ------------------------------------------------------------

ACTION_STEERING = 0.25
NEUTRAL_DEADBAND_RAD = math.radians(8.0)
SIGN_MIN = 0.05

BASE_SPEED_FRACTION = 0.24
TURN_SPEED_FRACTION = 0.13
HARD_TURN_SPEED_FRACTION = 0.08

STEERING_GAIN = 0.62
MOTOR_SMOOTH_ALPHA = 0.25

# Engineering phase convention used by goal controller v1.
WORLD_HEADING_OFFSET_RAD = 0.0

HEADING_ORDER = [
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

BIN_WIDTH = 2.0 * math.pi / len(HEADING_ORDER)


# ============================================================
# Helpers
# ============================================================

def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle <= -math.pi:
        angle += 2.0 * math.pi

    return angle


def load_frozen_matrix():
    project_root = Path(__file__).resolve().parents[2]

    matrix_path = (
        project_root
        / "connectome"
        / "output"
        / "frozen_fc2_steering_matrix.csv"
    )

    if not matrix_path.exists():
        raise FileNotFoundError(
            "Frozen steering matrix not found:\n"
            f"{matrix_path}"
        )

    with matrix_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Frozen steering matrix has no header."
            )

        goal_columns = [
            f"C{i}"
            for i in range(1, 10)
        ]

        heading_field = None

        for candidate in [
            "heading_column",
            "heading",
            "epg_column",
        ]:
            if candidate in reader.fieldnames:
                heading_field = candidate
                break

        if heading_field is None:
            non_goal = [
                field
                for field in reader.fieldnames
                if field not in goal_columns
                and not field.lower().startswith("unnamed")
            ]

            if len(non_goal) != 1:
                raise RuntimeError(
                    "Could not identify heading column "
                    "in frozen steering matrix."
                )

            heading_field = non_goal[0]

        matrix = {}

        for row in reader:
            heading = str(
                row[heading_field]
            ).strip()

            if not heading:
                continue

            matrix[heading] = {
                goal: float(row[goal])
                for goal in goal_columns
            }

    missing = [
        heading
        for heading in HEADING_ORDER
        if heading not in matrix
    ]

    if missing:
        raise RuntimeError(
            "Frozen steering matrix missing headings: "
            f"{missing}"
        )

    return matrix


def quantize_heading(angle):
    relative = (
        wrap_pi(
            angle
            -
            WORLD_HEADING_OFFSET_RAD
        )
        %
        (2.0 * math.pi)
    )

    index = int(
        math.floor(
            relative
            /
            BIN_WIDTH
            +
            0.5
        )
    ) % len(HEADING_ORDER)

    return (
        index,
        HEADING_ORDER[index],
    )


def best_action_in_row(
    matrix,
    heading_label,
    desired_steering,
):
    row = matrix[
        heading_label
    ]

    best_goal = min(
        row,
        key=lambda goal:
            abs(
                row[goal]
                -
                desired_steering
            ),
    )

    return (
        best_goal,
        float(
            row[best_goal]
        ),
    )


def sign_is_usable(
    steering,
    desired,
):
    if desired < 0.0:
        return steering <= -SIGN_MIN

    if desired > 0.0:
        return steering >= +SIGN_MIN

    return abs(steering) <= 0.12


def choose_connectome_action(
    matrix,
    heading_index,
    desired_steering,
):
    current_heading = HEADING_ORDER[
        heading_index
    ]

    goal_column, neural_steering = (
        best_action_in_row(
            matrix,
            current_heading,
            desired_steering,
        )
    )

    if sign_is_usable(
        neural_steering,
        desired_steering,
    ):
        return (
            current_heading,
            goal_column,
            neural_steering,
            False,
        )

    # Generic bounded neighbour fallback.
    candidates = []

    for offset in [
        -1,
        +1,
    ]:
        candidate_index = (
            heading_index
            +
            offset
        ) % len(
            HEADING_ORDER
        )

        candidate_heading = (
            HEADING_ORDER[
                candidate_index
            ]
        )

        candidate_goal, candidate_steering = (
            best_action_in_row(
                matrix,
                candidate_heading,
                desired_steering,
            )
        )

        if sign_is_usable(
            candidate_steering,
            desired_steering,
        ):
            candidates.append(
                (
                    abs(
                        candidate_steering
                        -
                        desired_steering
                    ),
                    candidate_heading,
                    candidate_goal,
                    candidate_steering,
                )
            )

    if candidates:
        candidates.sort(
            key=lambda item:
                item[0]
        )

        _, heading, goal, steering = (
            candidates[0]
        )

        return (
            heading,
            goal,
            steering,
            True,
        )

    return (
        current_heading,
        goal_column,
        neural_steering,
        False,
    )


def get_robot_heading(
    self_node,
):
    orientation = (
        self_node.getOrientation()
    )

    forward_x = float(
        orientation[0]
    )

    forward_y = float(
        orientation[3]
    )

    return math.atan2(
        forward_y,
        forward_x,
    )


def save_results(
    output_file,
    rows,
):
    if not rows:
        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main validation
# ============================================================

def main():
    robot = Supervisor()

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 41 requires supervisor TRUE."
        )

    translation_field = (
        self_node.getField(
            "translation"
        )
    )

    rotation_field = (
        self_node.getField(
            "rotation"
        )
    )

    if (
        translation_field is None
        or
        rotation_field is None
    ):
        raise RuntimeError(
            "Could not access robot translation/rotation."
        )

    left_motor = robot.getDevice(
        "left wheel motor"
    )

    right_motor = robot.getDevice(
        "right wheel motor"
    )

    left_motor.setPosition(
        float("inf")
    )

    right_motor.setPosition(
        float("inf")
    )

    left_motor.setVelocity(
        0.0
    )

    right_motor.setVelocity(
        0.0
    )

    max_speed = min(
        float(
            left_motor.getMaxVelocity()
        ),
        float(
            right_motor.getMaxVelocity()
        ),
    )

    matrix = load_frozen_matrix()

    # --------------------------------------------------------
    # Capture the user's current pose as the fixed start pose.
    # --------------------------------------------------------

    start_translation = list(
        translation_field.getSFVec3f()
    )

    start_rotation = list(
        rotation_field.getSFRotation()
    )

    start_x = float(
        start_translation[0]
    )

    start_y = float(
        start_translation[1]
    )

    # --------------------------------------------------------
    # Optional visual GOAL node.
    # --------------------------------------------------------

    goal_node = robot.getFromDef(
        "GOAL"
    )

    goal_translation_field = None
    goal_z = 0.01

    if goal_node is not None:
        goal_translation_field = (
            goal_node.getField(
                "translation"
            )
        )

        if goal_translation_field is not None:
            existing_goal_translation = (
                goal_translation_field
                .getSFVec3f()
            )

            goal_z = float(
                existing_goal_translation[2]
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    output_file = (
        project_root
        / "connectome"
        / "output"
        / "webots_goal_validation_results.csv"
    )

    results = []

    print()
    print("=" * 110)
    print(
        "STEP 41 - AUTOMATED 8-DIRECTION "
        "WEBOTS GOAL VALIDATION"
    )
    print("=" * 110)
    print()
    print(
        f"Start position: "
        f"({start_x:+.3f}, {start_y:+.3f})"
    )
    print(
        f"Goal radius: "
        f"{GOAL_RADIUS_M:.3f} m"
    )
    print(
        f"Goal tolerance: "
        f"{GOAL_TOLERANCE_M:.3f} m"
    )
    print(
        f"Trial timeout: "
        f"{TRIAL_TIMEOUT_S:.1f} s"
    )
    print(
        f"Visual DEF GOAL available: "
        f"{goal_translation_field is not None}"
    )
    print()
    print(
        "IMPORTANT: this test assumes the captured "
        "start position has enough open space in all "
        "8 directions."
    )

    for trial_number, (
        direction_name,
        direction_degrees,
    ) in enumerate(
        DIRECTIONS,
        start=1,
    ):
        direction_rad = math.radians(
            direction_degrees
        )

        goal_x = (
            start_x
            +
            GOAL_RADIUS_M
            *
            math.cos(
                direction_rad
            )
        )

        goal_y = (
            start_y
            +
            GOAL_RADIUS_M
            *
            math.sin(
                direction_rad
            )
        )

        # ----------------------------------------------------
        # Stop and reset rover pose.
        # ----------------------------------------------------

        left_motor.setVelocity(
            0.0
        )

        right_motor.setVelocity(
            0.0
        )

        translation_field.setSFVec3f(
            start_translation
        )

        rotation_field.setSFRotation(
            start_rotation
        )

        self_node.resetPhysics()

        # Move visible goal if available.
        if goal_translation_field is not None:
            goal_translation_field.setSFVec3f(
                [
                    goal_x,
                    goal_y,
                    goal_z,
                ]
            )

            goal_node.resetPhysics()

        # Let Webots settle after teleport.
        for _ in range(
            SETTLE_STEPS
        ):
            if robot.step(
                TIME_STEP
            ) == -1:
                return

        previous_left = 0.0
        previous_right = 0.0

        elapsed_s = 0.0
        min_distance = float(
            "inf"
        )
        path_length = 0.0
        fallback_count = 0
        control_steps = 0

        epg_states = set()
        policy_epg_states = set()
        fc2_states = set()

        previous_position = (
            self_node.getPosition()
        )

        previous_x = float(
            previous_position[0]
        )

        previous_y = float(
            previous_position[1]
        )

        success = False

        print()
        print("-" * 110)
        print(
            f"TRIAL {trial_number}/8 | "
            f"{direction_name} | "
            f"goal=({goal_x:+.3f},"
            f"{goal_y:+.3f})"
        )
        print("-" * 110)

        while elapsed_s < TRIAL_TIMEOUT_S:
            if robot.step(
                TIME_STEP
            ) == -1:
                return

            elapsed_s += (
                TIME_STEP
                /
                1000.0
            )

            control_steps += 1

            position = (
                self_node.getPosition()
            )

            x = float(
                position[0]
            )

            y = float(
                position[1]
            )

            path_length += math.hypot(
                x - previous_x,
                y - previous_y,
            )

            previous_x = x
            previous_y = y

            dx = (
                goal_x
                -
                x
            )

            dy = (
                goal_y
                -
                y
            )

            distance = math.hypot(
                dx,
                dy,
            )

            min_distance = min(
                min_distance,
                distance,
            )

            if distance <= GOAL_TOLERANCE_M:
                success = True

                left_motor.setVelocity(
                    0.0
                )

                right_motor.setVelocity(
                    0.0
                )

                break

            robot_heading = (
                get_robot_heading(
                    self_node
                )
            )

            goal_bearing = math.atan2(
                dy,
                dx,
            )

            bearing_error = wrap_pi(
                goal_bearing
                -
                robot_heading
            )

            # Neural sign convention:
            # + = right, - = left.
            if (
                bearing_error
                >
                NEUTRAL_DEADBAND_RAD
            ):
                desired_neural = (
                    -ACTION_STEERING
                )

            elif (
                bearing_error
                <
                -NEUTRAL_DEADBAND_RAD
            ):
                desired_neural = (
                    +ACTION_STEERING
                )

            else:
                desired_neural = 0.0

            (
                heading_index,
                heading_label,
            ) = quantize_heading(
                robot_heading
            )

            (
                policy_heading,
                goal_column,
                neural_steering,
                used_fallback,
            ) = choose_connectome_action(
                matrix,
                heading_index,
                desired_neural,
            )

            epg_states.add(
                heading_label
            )

            policy_epg_states.add(
                policy_heading
            )

            fc2_states.add(
                goal_column
            )

            if used_fallback:
                fallback_count += 1

            abs_error = abs(
                bearing_error
            )

            if abs_error > math.radians(
                90.0
            ):
                forward_speed = (
                    HARD_TURN_SPEED_FRACTION
                    *
                    max_speed
                )

            elif abs_error > math.radians(
                35.0
            ):
                forward_speed = (
                    TURN_SPEED_FRACTION
                    *
                    max_speed
                )

            else:
                forward_speed = (
                    BASE_SPEED_FRACTION
                    *
                    max_speed
                )

            turn_speed = (
                neural_steering
                *
                STEERING_GAIN
                *
                max_speed
            )

            target_left = (
                forward_speed
                +
                turn_speed
            )

            target_right = (
                forward_speed
                -
                turn_speed
            )

            target_left = max(
                -max_speed,
                min(
                    max_speed,
                    target_left,
                ),
            )

            target_right = max(
                -max_speed,
                min(
                    max_speed,
                    target_right,
                ),
            )

            left_velocity = (
                (
                    1.0
                    -
                    MOTOR_SMOOTH_ALPHA
                )
                *
                previous_left
                +
                MOTOR_SMOOTH_ALPHA
                *
                target_left
            )

            right_velocity = (
                (
                    1.0
                    -
                    MOTOR_SMOOTH_ALPHA
                )
                *
                previous_right
                +
                MOTOR_SMOOTH_ALPHA
                *
                target_right
            )

            left_motor.setVelocity(
                left_velocity
            )

            right_motor.setVelocity(
                right_velocity
            )

            previous_left = (
                left_velocity
            )

            previous_right = (
                right_velocity
            )

            if (
                control_steps
                %
                40
                ==
                0
            ):
                print(
                    f"  t={elapsed_s:5.2f}s "
                    f"dist={distance:.3f} "
                    f"err={math.degrees(bearing_error):+6.1f}deg "
                    f"EPG={heading_label:>2} "
                    f"policy={policy_heading:>2} "
                    f"FC2={goal_column} "
                    f"steer={neural_steering:+.3f} "
                    f"fallback={used_fallback}"
                )

        left_motor.setVelocity(
            0.0
        )

        right_motor.setVelocity(
            0.0
        )

        final_position = (
            self_node.getPosition()
        )

        final_x = float(
            final_position[0]
        )

        final_y = float(
            final_position[1]
        )

        final_distance = math.hypot(
            goal_x - final_x,
            goal_y - final_y,
        )

        fallback_fraction = (
            fallback_count
            /
            control_steps
            if control_steps > 0
            else 0.0
        )

        result = {
            "trial":
                trial_number,

            "direction":
                direction_name,

            "direction_degrees":
                direction_degrees,

            "goal_x":
                round(
                    goal_x,
                    6,
                ),

            "goal_y":
                round(
                    goal_y,
                    6,
                ),

            "success":
                success,

            "elapsed_s":
                round(
                    elapsed_s,
                    4,
                ),

            "final_distance_m":
                round(
                    final_distance,
                    6,
                ),

            "min_distance_m":
                round(
                    min_distance,
                    6,
                ),

            "path_length_m":
                round(
                    path_length,
                    6,
                ),

            "path_efficiency":
                round(
                    GOAL_RADIUS_M
                    /
                    path_length
                    if path_length > 1e-9
                    else 0.0,
                    6,
                ),

            "control_steps":
                control_steps,

            "fallback_count":
                fallback_count,

            "fallback_fraction":
                round(
                    fallback_fraction,
                    6,
                ),

            "unique_epg_states":
                len(
                    epg_states
                ),

            "epg_states":
                ",".join(
                    sorted(
                        epg_states
                    )
                ),

            "unique_policy_epg_states":
                len(
                    policy_epg_states
                ),

            "policy_epg_states":
                ",".join(
                    sorted(
                        policy_epg_states
                    )
                ),

            "unique_fc2_states":
                len(
                    fc2_states
                ),

            "fc2_states":
                ",".join(
                    sorted(
                        fc2_states
                    )
                ),
        }

        results.append(
            result
        )

        save_results(
            output_file,
            results,
        )

        status = (
            "PASS"
            if success
            else
            "FAIL"
        )

        print()
        print(
            f"{status} | "
            f"{direction_name} | "
            f"time={elapsed_s:.2f}s | "
            f"final_dist={final_distance:.3f}m | "
            f"path={path_length:.3f}m | "
            f"fallback={fallback_count}/"
            f"{control_steps}"
        )

    # --------------------------------------------------------
    # Restore original start pose at end.
    # --------------------------------------------------------

    left_motor.setVelocity(
        0.0
    )

    right_motor.setVelocity(
        0.0
    )

    translation_field.setSFVec3f(
        start_translation
    )

    rotation_field.setSFRotation(
        start_rotation
    )

    self_node.resetPhysics()

    save_results(
        output_file,
        results,
    )

    passes = sum(
        1
        for row in results
        if row["success"]
    )

    print()
    print("=" * 110)
    print(
        "STEP 41 FINAL SUMMARY"
    )
    print("=" * 110)
    print()

    for row in results:
        status = (
            "PASS"
            if row["success"]
            else
            "FAIL"
        )

        print(
            f"{row['direction']:>2} : "
            f"{status:4s} | "
            f"time={row['elapsed_s']:>6.2f}s | "
            f"final={row['final_distance_m']:.3f}m | "
            f"path_eff={row['path_efficiency']:.3f} | "
            f"fallback={row['fallback_count']}"
        )

    print()
    print(
        f"Successful goal directions: "
        f"{passes}/8"
    )

    total_fallback = sum(
        int(
            row[
                "fallback_count"
            ]
        )
        for row in results
    )

    print(
        f"Total neighbour-fallback activations: "
        f"{total_fallback}"
    )

    print(
        f"Results CSV: "
        f"{output_file}"
    )

    print()

    if passes == 8:
        print(
            "PASS: all 8 world-space goal directions "
            "were reached."
        )

        print(
            "NEXT: freeze goal-navigation v1 and merge "
            "it with the frozen obstacle-avoidance controller."
        )

    else:
        print(
            "REVIEW: one or more world-space directions "
            "failed."
        )

        print(
            "Inspect only the failed direction(s), heading "
            "phase and fallback behaviour before obstacle merge."
        )


if __name__ == "__main__":
    main()
