import csv
import math
from pathlib import Path

from controller import Supervisor


# ============================================================
# STEP 42 - FLYNAV COMBINED NAVIGATION V1
#
# Frozen Goal Navigation v1
#          +
# Frozen Obstacle-Avoidance response
#          ↓
# Robotics-layer arbitration
#          ↓
# Webots E-puck
#
# IMPORTANT
# ---------
# This file does NOT retune the frozen connectome-derived
# goal steering map. The goal map is loaded directly from:
#
#   connectome/output/frozen_fc2_steering_matrix.csv
#
# The obstacle branch preserves the frozen obstacle-controller
# calibration envelope:
#
#   SENSOR_FLOOR       = 80
#   SENSOR_SATURATION  = 650
#   BASE_SPEED         = 0.28 * max wheel speed
#   STEERING_GAIN      = 0.55 * max wheel speed
#   SMOOTH_ALPHA       = 0.25
#   speed_scale        = 1 - 0.85 * avg_obstacle
#   speed_scale clamp  = [0.12, 1.0]
#
# Arbitration, hysteresis and command blending are engineering
# robotics logic. They are not biological claims.
# ============================================================

TIME_STEP = 32

GOAL_FALLBACK_X = 0.30
GOAL_FALLBACK_Y = 0.30
GOAL_TOLERANCE_M = 0.035

ACTION_STEERING = 0.25
NEUTRAL_DEADBAND_RAD = math.radians(8.0)
GOAL_SIGN_MIN = 0.05

GOAL_BASE_SPEED_FRACTION = 0.24
GOAL_TURN_SPEED_FRACTION = 0.13
GOAL_HARD_TURN_SPEED_FRACTION = 0.08
GOAL_STEERING_GAIN = 0.62

WORLD_HEADING_OFFSET_RAD = 0.0

HEADING_ORDER = [
    "L1", "L2", "L3", "L4",
    "L5", "L6", "L7", "L8",
    "R8", "R7", "R6", "R5",
    "R4", "R3", "R2", "R1",
]

BIN_WIDTH = 2.0 * math.pi / len(HEADING_ORDER)

SENSOR_FLOOR = 80.0
SENSOR_SATURATION = 650.0

OBSTACLE_BASE_SPEED_FRACTION = 0.28
OBSTACLE_STEERING_GAIN = 0.55
OBSTACLE_SMOOTH_ALPHA = 0.25

OBSTACLE_SPEED_REDUCTION = 0.85
OBSTACLE_MIN_SPEED_SCALE = 0.12

LEFT_SENSOR_NAMES = ["ps5", "ps6", "ps7"]
RIGHT_SENSOR_NAMES = ["ps0", "ps1", "ps2"]
FRONT_SENSOR_NAMES = ["ps7", "ps0"]

OBSTACLE_ENTER_RAW = 250.0
OBSTACLE_EXIT_RAW = 180.0
OBSTACLE_FULL_CONTROL_RAW = 650.0
OBSTACLE_TIE_EPS = 0.04

MOTOR_SMOOTH_ALPHA = 0.25

PRINT_EVERY_STEPS = 10
CSV_LOG_EVERY_STEPS = 5


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def normalize_sensor(raw_value):
    return clamp(
        (float(raw_value) - SENSOR_FLOOR)
        / (SENSOR_SATURATION - SENSOR_FLOOR),
        0.0,
        1.0,
    )


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

        goal_columns = [f"C{i}" for i in range(1, 10)]

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
            non_goal_fields = [
                field
                for field in reader.fieldnames
                if field not in goal_columns
                and not field.lower().startswith("unnamed")
            ]

            if len(non_goal_fields) != 1:
                raise RuntimeError(
                    "Could not identify heading column "
                    "in frozen steering matrix."
                )

            heading_field = non_goal_fields[0]

        matrix = {}

        for row in reader:
            heading = str(row[heading_field]).strip()
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
        wrap_pi(angle - WORLD_HEADING_OFFSET_RAD)
        % (2.0 * math.pi)
    )

    index = int(
        math.floor(relative / BIN_WIDTH + 0.5)
    ) % len(HEADING_ORDER)

    return index, HEADING_ORDER[index]


def best_action_in_row(
    matrix,
    heading_label,
    desired_steering,
):
    row = matrix[heading_label]

    best_goal = min(
        row,
        key=lambda goal:
            abs(row[goal] - desired_steering),
    )

    return best_goal, float(row[best_goal])


def sign_is_usable(steering, desired):
    if desired < 0.0:
        return steering <= -GOAL_SIGN_MIN

    if desired > 0.0:
        return steering >= +GOAL_SIGN_MIN

    return abs(steering) <= 0.12


def choose_connectome_goal_action(
    matrix,
    heading_index,
    desired_steering,
):
    current_heading = HEADING_ORDER[heading_index]

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

    candidates = []

    for offset in [-1, +1]:
        candidate_index = (
            heading_index + offset
        ) % len(HEADING_ORDER)

        candidate_heading = HEADING_ORDER[
            candidate_index
        ]

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
                        - desired_steering
                    ),
                    candidate_heading,
                    candidate_goal,
                    candidate_steering,
                )
            )

    if candidates:
        candidates.sort(
            key=lambda item: item[0]
        )

        _, heading, goal, steering = candidates[0]

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


def get_robot_heading(self_node):
    orientation = self_node.getOrientation()

    forward_x = float(orientation[0])
    forward_y = float(orientation[3])

    return math.atan2(
        forward_y,
        forward_x,
    )


def read_obstacle_state(proximity_sensors):
    raw = {
        name: float(
            proximity_sensors[name].getValue()
        )
        for name in proximity_sensors
    }

    normalized = {
        name: normalize_sensor(value)
        for name, value in raw.items()
    }

    left_activation = max(
        normalized[name]
        for name in LEFT_SENSOR_NAMES
    )

    right_activation = max(
        normalized[name]
        for name in RIGHT_SENSOR_NAMES
    )

    front_activation = max(
        normalized[name]
        for name in FRONT_SENSOR_NAMES
    )

    left_raw = max(
        raw[name]
        for name in LEFT_SENSOR_NAMES
    )

    right_raw = max(
        raw[name]
        for name in RIGHT_SENSOR_NAMES
    )

    front_raw = max(
        raw[name]
        for name in FRONT_SENSOR_NAMES
    )

    max_raw = max(
        left_raw,
        right_raw,
        front_raw,
    )

    # Frozen sign convention:
    # left obstacle -> positive steering -> right turn
    # right obstacle -> negative steering -> left turn
    raw_steering = (
        left_activation
        - right_activation
    )

    average_obstacle = (
        left_activation
        + right_activation
    ) / 2.0

    return {
        "raw_values": raw,
        "left_activation": left_activation,
        "right_activation": right_activation,
        "front_activation": front_activation,
        "left_raw": left_raw,
        "right_raw": right_raw,
        "front_raw": front_raw,
        "max_raw": max_raw,
        "raw_steering": clamp(
            raw_steering,
            -1.0,
            +1.0,
        ),
        "average_obstacle": clamp(
            average_obstacle,
            0.0,
            1.0,
        ),
    }


def obstacle_blend_weight(max_raw):
    return clamp(
        (
            max_raw
            - OBSTACLE_EXIT_RAW
        )
        /
        (
            OBSTACLE_FULL_CONTROL_RAW
            - OBSTACLE_EXIT_RAW
        ),
        0.0,
        1.0,
    )


def open_log():
    project_root = Path(__file__).resolve().parents[2]

    output_path = (
        project_root
        / "connectome"
        / "output"
        / "combined_navigation_v1_runtime.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle = output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    )

    fields = [
        "step",
        "time_s",
        "x",
        "y",
        "goal_x",
        "goal_y",
        "distance_m",
        "heading_deg",
        "bearing_error_deg",
        "epg_heading",
        "policy_epg",
        "fc2_column",
        "goal_steering",
        "goal_fallback",
        "left_sensor_raw",
        "right_sensor_raw",
        "front_sensor_raw",
        "max_sensor_raw",
        "left_obstacle",
        "right_obstacle",
        "obstacle_steering",
        "obstacle_blend",
        "mode",
        "forward_speed",
        "turn_speed",
        "left_wheel",
        "right_wheel",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
    )

    writer.writeheader()
    handle.flush()

    return output_path, handle, writer


def main():
    robot = Supervisor()

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Combined controller requires "
            "supervisor TRUE."
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

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    max_speed = min(
        float(left_motor.getMaxVelocity()),
        float(right_motor.getMaxVelocity()),
    )

    proximity_sensors = {}

    for index in range(8):
        name = f"ps{index}"

        sensor = robot.getDevice(name)
        sensor.enable(TIME_STEP)

        proximity_sensors[name] = sensor

    matrix = load_frozen_matrix()

    goal_node = robot.getFromDef(
        "GOAL"
    )

    previous_left = 0.0
    previous_right = 0.0

    smoothed_obstacle_steering = 0.0

    obstacle_latched = False

    last_avoid_direction = +1.0

    step_count = 0

    log_path, log_handle, log_writer = open_log()

    print()
    print("=" * 110)
    print(
        "STEP 42 - FLYNAV COMBINED NAVIGATION V1"
    )
    print("=" * 110)
    print()

    print(
        "[Combined v1] "
        "Frozen goal steering map loaded."
    )

    print(
        "[Combined v1] "
        "Frozen obstacle calibration envelope loaded."
    )

    print(
        "[Combined v1] "
        "Arbitration: goal -> blend -> obstacle."
    )

    print(
        "[Combined v1] "
        f"Obstacle enter={OBSTACLE_ENTER_RAW:.0f}, "
        f"exit={OBSTACLE_EXIT_RAW:.0f}, "
        f"full={OBSTACLE_FULL_CONTROL_RAW:.0f}."
    )

    print(
        "[Combined v1] "
        f"Runtime log: {log_path}"
    )

    if goal_node is not None:
        print(
            "[Combined v1] "
            "Using DEF GOAL position."
        )

    else:
        print(
            "[Combined v1] "
            "No DEF GOAL found; "
            f"using ({GOAL_FALLBACK_X:.2f}, "
            f"{GOAL_FALLBACK_Y:.2f})."
        )

    print()

    try:
        while robot.step(TIME_STEP) != -1:
            step_count += 1

            time_s = (
                step_count
                * TIME_STEP
                / 1000.0
            )

            position = self_node.getPosition()

            x = float(position[0])
            y = float(position[1])

            if goal_node is not None:
                goal_position = goal_node.getPosition()

                goal_x = float(goal_position[0])
                goal_y = float(goal_position[1])

            else:
                goal_x = GOAL_FALLBACK_X
                goal_y = GOAL_FALLBACK_Y

            dx = goal_x - x
            dy = goal_y - y

            distance = math.hypot(
                dx,
                dy,
            )

            if distance <= GOAL_TOLERANCE_M:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                previous_left = 0.0
                previous_right = 0.0

                if step_count % 20 == 0:
                    print(
                        "[Combined v1] "
                        "GOAL REACHED | "
                        f"distance={distance:.3f} m"
                    )

                continue

            robot_heading = get_robot_heading(
                self_node
            )

            goal_bearing = math.atan2(
                dy,
                dx,
            )

            bearing_error = wrap_pi(
                goal_bearing
                - robot_heading
            )

            if (
                bearing_error
                > NEUTRAL_DEADBAND_RAD
            ):
                desired_neural = (
                    -ACTION_STEERING
                )

            elif (
                bearing_error
                < -NEUTRAL_DEADBAND_RAD
            ):
                desired_neural = (
                    +ACTION_STEERING
                )

            else:
                desired_neural = 0.0

            heading_index, heading_label = (
                quantize_heading(
                    robot_heading
                )
            )

            (
                policy_heading,
                goal_column,
                goal_steering,
                used_goal_fallback,
            ) = choose_connectome_goal_action(
                matrix,
                heading_index,
                desired_neural,
            )

            abs_error = abs(
                bearing_error
            )

            if (
                abs_error
                > math.radians(90.0)
            ):
                goal_forward = (
                    GOAL_HARD_TURN_SPEED_FRACTION
                    * max_speed
                )

            elif (
                abs_error
                > math.radians(35.0)
            ):
                goal_forward = (
                    GOAL_TURN_SPEED_FRACTION
                    * max_speed
                )

            else:
                goal_forward = (
                    GOAL_BASE_SPEED_FRACTION
                    * max_speed
                )

            goal_turn = (
                goal_steering
                * GOAL_STEERING_GAIN
                * max_speed
            )

            obstacle = read_obstacle_state(
                proximity_sensors
            )

            max_sensor_raw = obstacle[
                "max_raw"
            ]

            if not obstacle_latched:
                if (
                    max_sensor_raw
                    >= OBSTACLE_ENTER_RAW
                ):
                    obstacle_latched = True

            else:
                if (
                    max_sensor_raw
                    <= OBSTACLE_EXIT_RAW
                ):
                    obstacle_latched = False

            raw_obstacle_steering = obstacle[
                "raw_steering"
            ]

            if (
                obstacle_latched
                and obstacle["front_raw"]
                >= OBSTACLE_ENTER_RAW
                and abs(
                    raw_obstacle_steering
                ) < OBSTACLE_TIE_EPS
            ):
                if (
                    abs(goal_steering)
                    >= GOAL_SIGN_MIN
                ):
                    raw_obstacle_steering = (
                        +1.0
                        if goal_steering > 0.0
                        else -1.0
                    )
                else:
                    raw_obstacle_steering = (
                        last_avoid_direction
                    )

            if (
                abs(raw_obstacle_steering)
                >= OBSTACLE_TIE_EPS
            ):
                last_avoid_direction = (
                    +1.0
                    if raw_obstacle_steering > 0.0
                    else -1.0
                )

            smoothed_obstacle_steering = (
                (
                    1.0
                    - OBSTACLE_SMOOTH_ALPHA
                )
                * smoothed_obstacle_steering
                +
                OBSTACLE_SMOOTH_ALPHA
                * raw_obstacle_steering
            )

            obstacle_speed_scale = clamp(
                (
                    1.0
                    - OBSTACLE_SPEED_REDUCTION
                    * obstacle[
                        "average_obstacle"
                    ]
                ),
                OBSTACLE_MIN_SPEED_SCALE,
                1.0,
            )

            obstacle_forward = (
                OBSTACLE_BASE_SPEED_FRACTION
                * max_speed
                * obstacle_speed_scale
            )

            obstacle_turn = (
                smoothed_obstacle_steering
                * OBSTACLE_STEERING_GAIN
                * max_speed
            )

            if obstacle_latched:
                blend = obstacle_blend_weight(
                    max_sensor_raw
                )

                blend = max(
                    blend,
                    0.15,
                )

            else:
                blend = 0.0

            final_turn = (
                (
                    1.0
                    - blend
                )
                * goal_turn
                +
                blend
                * obstacle_turn
            )

            final_forward = min(
                goal_forward,
                obstacle_forward,
            )

            if blend >= 0.75:
                mode = "AVOID"
            elif blend > 0.0:
                mode = "BLEND"
            else:
                mode = "GOAL"

            target_left = (
                final_forward
                + final_turn
            )

            target_right = (
                final_forward
                - final_turn
            )

            target_left = clamp(
                target_left,
                -max_speed,
                +max_speed,
            )

            target_right = clamp(
                target_right,
                -max_speed,
                +max_speed,
            )

            left_velocity = (
                (
                    1.0
                    - MOTOR_SMOOTH_ALPHA
                )
                * previous_left
                +
                MOTOR_SMOOTH_ALPHA
                * target_left
            )

            right_velocity = (
                (
                    1.0
                    - MOTOR_SMOOTH_ALPHA
                )
                * previous_right
                +
                MOTOR_SMOOTH_ALPHA
                * target_right
            )

            left_motor.setVelocity(
                left_velocity
            )

            right_motor.setVelocity(
                right_velocity
            )

            previous_left = left_velocity
            previous_right = right_velocity

            if (
                step_count
                % PRINT_EVERY_STEPS
                == 0
            ):
                print(
                    "[Combined v1] "
                    f"mode={mode:<5} "
                    f"dist={distance:.3f} "
                    f"err={math.degrees(bearing_error):+6.1f}deg "
                    f"EPG={heading_label:>2} "
                    f"policy={policy_heading:>2} "
                    f"FC2={goal_column} "
                    f"goal={goal_steering:+.3f} "
                    f"obs={smoothed_obstacle_steering:+.3f} "
                    f"prox={max_sensor_raw:7.1f} "
                    f"blend={blend:.2f} "
                    f"wheels=({left_velocity:+.2f},"
                    f"{right_velocity:+.2f})"
                )

            if (
                step_count
                % CSV_LOG_EVERY_STEPS
                == 0
            ):
                log_writer.writerow(
                    {
                        "step": step_count,
                        "time_s": f"{time_s:.4f}",
                        "x": f"{x:.6f}",
                        "y": f"{y:.6f}",
                        "goal_x": f"{goal_x:.6f}",
                        "goal_y": f"{goal_y:.6f}",
                        "distance_m": f"{distance:.6f}",
                        "heading_deg":
                            f"{math.degrees(robot_heading):.4f}",
                        "bearing_error_deg":
                            f"{math.degrees(bearing_error):.4f}",
                        "epg_heading": heading_label,
                        "policy_epg": policy_heading,
                        "fc2_column": goal_column,
                        "goal_steering":
                            f"{goal_steering:.6f}",
                        "goal_fallback":
                            int(used_goal_fallback),
                        "left_sensor_raw":
                            f"{obstacle['left_raw']:.3f}",
                        "right_sensor_raw":
                            f"{obstacle['right_raw']:.3f}",
                        "front_sensor_raw":
                            f"{obstacle['front_raw']:.3f}",
                        "max_sensor_raw":
                            f"{max_sensor_raw:.3f}",
                        "left_obstacle":
                            f"{obstacle['left_activation']:.6f}",
                        "right_obstacle":
                            f"{obstacle['right_activation']:.6f}",
                        "obstacle_steering":
                            f"{smoothed_obstacle_steering:.6f}",
                        "obstacle_blend":
                            f"{blend:.6f}",
                        "mode": mode,
                        "forward_speed":
                            f"{final_forward:.6f}",
                        "turn_speed":
                            f"{final_turn:.6f}",
                        "left_wheel":
                            f"{left_velocity:.6f}",
                        "right_wheel":
                            f"{right_velocity:.6f}",
                    }
                )

                log_handle.flush()

    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()
