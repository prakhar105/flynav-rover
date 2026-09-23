import csv
import math
from pathlib import Path

from controller import Supervisor


# ============================================================
# FlyNav Goal Controller v1
#
# Simulation-only Webots controller.
#
# Connectome-derived component:
#   - uses the frozen 16 x 9 steering matrix produced by Step 27
#
# Engineering interfaces:
#   - world heading -> one of 16 EPG heading labels
#   - goal-bearing error -> desired discrete turn action
#   - FC2 column chosen as the closest connectome-derived action
#   - immediate-neighbour heading fallback when the current
#     heading row lacks the requested steering direction
#   - wheel-speed smoothing
#
# This is NOT a literal simulated fly brain.
# ============================================================


TIME_STEP = 32

GOAL_FALLBACK_X = 0.30
GOAL_FALLBACK_Y = 0.30
GOAL_TOLERANCE_M = 0.035

# Existing project convention:
# positive neural steering = right turn
# negative neural steering = left turn
ACTION_STEERING = 0.25
NEUTRAL_DEADBAND_RAD = math.radians(8.0)
SIGN_MIN = 0.05

BASE_SPEED_FRACTION = 0.24
TURN_SPEED_FRACTION = 0.13
HARD_TURN_SPEED_FRACTION = 0.08

STEERING_GAIN = 0.62
MOTOR_SMOOTH_ALPHA = 0.25

# This is an engineered simulation phase convention:
# L1 is aligned to +X / 0 radians.
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
                "Steering matrix has no header."
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
                    "Could not identify heading column in "
                    "frozen steering matrix."
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
            "Frozen matrix is missing heading rows: "
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

    return index, HEADING_ORDER[index]


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
            row[
                best_goal
            ]
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

    # --------------------------------------------------------
    # Bounded generic fallback.
    #
    # Step 39 showed only L3 and L4 lack negative actions.
    # Rather than special-casing those labels, search the two
    # immediately adjacent heading rows and take the closest
    # usable connectome-derived action.
    # --------------------------------------------------------

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
            candidates[
                0
            ]
        )

        return (
            heading,
            goal,
            steering,
            True,
        )

    # Last resort: keep the current row's closest available
    # action. This preserves bounded behaviour if a future
    # matrix differs from the current Step 27 map.
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
        self_node
        .getOrientation()
    )

    forward_x = orientation[
        0
    ]

    forward_y = orientation[
        3
    ]

    return math.atan2(
        forward_y,
        forward_x,
    )


def main():
    robot = Supervisor()

    self_node = (
        robot.getSelf()
    )

    if self_node is None:
        raise RuntimeError(
            "Controller requires supervisor TRUE."
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

    matrix = (
        load_frozen_matrix()
    )

    goal_node = (
        robot.getFromDef(
            "GOAL"
        )
    )

    previous_left = 0.0
    previous_right = 0.0

    print(
        "[FlyNav Goal v1] "
        "Frozen connectome-derived action map loaded."
    )

    print(
        "[FlyNav Goal v1] "
        "World->EPG phase is engineered: "
        "L1 = +X."
    )

    if goal_node is not None:
        print(
            "[FlyNav Goal v1] "
            "Using DEF GOAL position."
        )

    else:
        print(
            "[FlyNav Goal v1] "
            f"No DEF GOAL found; using "
            f"({GOAL_FALLBACK_X:.2f}, "
            f"{GOAL_FALLBACK_Y:.2f})."
        )

    step_count = 0

    while robot.step(
        TIME_STEP
    ) != -1:
        step_count += 1

        position = (
            self_node
            .getPosition()
        )

        x = float(
            position[
                0
            ]
        )

        y = float(
            position[
                1
            ]
        )

        if goal_node is not None:
            goal_position = (
                goal_node
                .getPosition()
            )

            goal_x = float(
                goal_position[
                    0
                ]
            )

            goal_y = float(
                goal_position[
                    1
                ]
            )

        else:
            goal_x = (
                GOAL_FALLBACK_X
            )

            goal_y = (
                GOAL_FALLBACK_Y
            )

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

        if distance <= GOAL_TOLERANCE_M:
            left_motor.setVelocity(
                0.0
            )

            right_motor.setVelocity(
                0.0
            )

            if step_count % 20 == 0:
                print(
                    "[FlyNav Goal v1] "
                    f"GOAL REACHED | "
                    f"distance={distance:.3f} m"
                )

            previous_left = 0.0
            previous_right = 0.0
            continue

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

        # ----------------------------------------------------
        # Existing neural sign convention:
        #
        # positive steering = right turn
        #
        # Geometric positive bearing error is left/CCW,
        # therefore desired neural sign is inverted.
        # ----------------------------------------------------

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
            used_neighbor_fallback,
        ) = choose_connectome_action(
            matrix,
            heading_index,
            desired_neural,
        )

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

        if step_count % 10 == 0:
            print(
                "[FlyNav Goal v1] "
                f"pos=({x:+.3f},{y:+.3f}) "
                f"goal=({goal_x:+.3f},{goal_y:+.3f}) "
                f"dist={distance:.3f} "
                f"heading={math.degrees(robot_heading):+.1f}deg "
                f"bearing_err={math.degrees(bearing_error):+.1f}deg "
                f"EPG={heading_label} "
                f"policy_EPG={policy_heading} "
                f"FC2={goal_column} "
                f"steer={neural_steering:+.3f} "
                f"fallback={used_neighbor_fallback} "
                f"wheels=({left_velocity:+.2f},"
                f"{right_velocity:+.2f})"
            )


if __name__ == "__main__":
    main()
