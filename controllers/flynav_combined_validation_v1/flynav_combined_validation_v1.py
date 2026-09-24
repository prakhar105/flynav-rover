import csv
import importlib.util
import math
from pathlib import Path

from controller import Supervisor


# ============================================================
# STEP 43 - COMBINED NAVIGATION V1 REGRESSION SUITE
#
# IMPORTANT:
# This validator imports the already-working
# controllers/flynav_combined_v1/flynav_combined_v1.py
# and reuses its exact constants/functions. It does not retune
# either frozen neural baseline.
# ============================================================

TIME_STEP = 32
TRIAL_TIMEOUT_S = 40.0
SETTLE_STEPS = 8
GOAL_TOLERANCE_M = 0.035

OBSTACLE_NAME = "test obstacle"
OBSTACLE_SIZE_X = 0.08
OBSTACLE_SIZE_Y = 0.08
ROBOT_RADIUS_EST_M = 0.035

CASES = [
    {
        "name": "clear_path",
        "start": (-0.30, 0.00),
        "heading_deg": 0.0,
        "goal": (0.30, 0.00),
        "obstacle": (0.00, 0.35),
        "expect_obstacle": False,
    },
    {
        "name": "obstacle_direct",
        "start": (-0.30, 0.00),
        "heading_deg": 0.0,
        "goal": (0.30, 0.00),
        "obstacle": (0.00, 0.00),
        "expect_obstacle": True,
    },
    {
        "name": "obstacle_left",
        "start": (-0.30, 0.00),
        "heading_deg": 0.0,
        "goal": (0.30, 0.00),
        "obstacle": (0.00, 0.04),
        "expect_obstacle": True,
    },
    {
        "name": "obstacle_right",
        "start": (-0.30, 0.00),
        "heading_deg": 0.0,
        "goal": (0.30, 0.00),
        "obstacle": (0.00, -0.04),
        "expect_obstacle": True,
    },
    {
        "name": "near_wall_obstacle",
        "start": (-0.30, -0.31),
        "heading_deg": 0.0,
        "goal": (0.30, -0.31),
        "obstacle": (0.00, -0.31),
        "expect_obstacle": True,
    },
    {
        "name": "goal_behind_obstacle",
        "start": (-0.30, 0.20),
        "heading_deg": 0.0,
        "goal": (0.28, 0.20),
        "obstacle": (0.10, 0.20),
        "expect_obstacle": True,
    },
]


def load_combined_module(project_root):
    source = (
        project_root
        / "controllers"
        / "flynav_combined_v1"
        / "flynav_combined_v1.py"
    )

    if not source.exists():
        raise FileNotFoundError(
            "Could not find combined controller:\n"
            f"{source}"
        )

    spec = importlib.util.spec_from_file_location(
        "flynav_combined_v1_impl",
        source,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_test_obstacle(supervisor):
    node = supervisor.getFromDef("TEST_OBSTACLE")
    if node is not None:
        return node

    root = supervisor.getRoot()
    children = root.getField("children")

    for i in range(children.getCount()):
        candidate = children.getMFNode(i)
        name_field = candidate.getField("name")

        if name_field is None:
            continue

        try:
            name = name_field.getSFString()
        except Exception:
            continue

        if name == OBSTACLE_NAME:
            return candidate

    return None


def estimated_box_clearance(
    robot_x,
    robot_y,
    obstacle_x,
    obstacle_y,
):
    half_x = OBSTACLE_SIZE_X / 2.0
    half_y = OBSTACLE_SIZE_Y / 2.0

    dx = max(
        abs(robot_x - obstacle_x) - half_x,
        0.0,
    )
    dy = max(
        abs(robot_y - obstacle_y) - half_y,
        0.0,
    )

    return math.hypot(dx, dy) - ROBOT_RADIUS_EST_M


def collapse_modes(modes):
    if not modes:
        return ""

    result = [modes[0]]

    for mode in modes[1:]:
        if mode != result[-1]:
            result.append(mode)

    return "->".join(result)


def main():
    robot = Supervisor()

    project_root = Path(__file__).resolve().parents[2]
    combined = load_combined_module(project_root)

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 43 requires supervisor TRUE."
        )

    robot_translation = self_node.getField("translation")
    robot_rotation = self_node.getField("rotation")

    original_robot_translation = list(
        robot_translation.getSFVec3f()
    )
    original_robot_rotation = list(
        robot_rotation.getSFRotation()
    )
    robot_z = float(original_robot_translation[2])

    obstacle_node = find_test_obstacle(robot)

    if obstacle_node is None:
        raise RuntimeError(
            'Could not find Solid named "test obstacle".'
        )

    obstacle_translation = obstacle_node.getField(
        "translation"
    )

    original_obstacle_translation = list(
        obstacle_translation.getSFVec3f()
    )
    obstacle_z = float(original_obstacle_translation[2])

    goal_node = robot.getFromDef("GOAL")
    goal_translation = None
    original_goal_translation = None
    goal_z = 0.01

    if goal_node is not None:
        goal_translation = goal_node.getField("translation")

        if goal_translation is not None:
            original_goal_translation = list(
                goal_translation.getSFVec3f()
            )
            goal_z = float(original_goal_translation[2])

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    max_speed = min(
        float(left_motor.getMaxVelocity()),
        float(right_motor.getMaxVelocity()),
    )

    sensors = {}

    for i in range(8):
        name = f"ps{i}"
        sensor = robot.getDevice(name)
        sensor.enable(TIME_STEP)
        sensors[name] = sensor

    matrix = combined.load_frozen_matrix()

    output_file = (
        project_root
        / "connectome"
        / "output"
        / "combined_navigation_v1_regression.csv"
    )

    results = []

    print()
    print("=" * 112)
    print(
        "STEP 43 - COMBINED NAVIGATION V1 "
        "AUTOMATED REGRESSION"
    )
    print("=" * 112)
    print(f'Obstacle: "{OBSTACLE_NAME}"')
    print(f"Cases: {len(CASES)}")
    print()

    try:
        for trial, case in enumerate(CASES, start=1):
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            sx, sy = case["start"]
            gx, gy = case["goal"]
            ox, oy = case["obstacle"]

            robot_translation.setSFVec3f(
                [sx, sy, robot_z]
            )

            robot_rotation.setSFRotation(
                [
                    0.0,
                    0.0,
                    1.0,
                    math.radians(case["heading_deg"]),
                ]
            )

            self_node.resetPhysics()

            obstacle_translation.setSFVec3f(
                [ox, oy, obstacle_z]
            )
            obstacle_node.resetPhysics()

            if goal_translation is not None:
                goal_translation.setSFVec3f(
                    [gx, gy, goal_z]
                )
                goal_node.resetPhysics()

            for _ in range(SETTLE_STEPS):
                if robot.step(TIME_STEP) == -1:
                    return

            previous_left = 0.0
            previous_right = 0.0
            smoothed_obstacle_steering = 0.0
            obstacle_latched = False
            last_avoid_direction = +1.0

            elapsed_s = 0.0
            steps = 0
            path_length = 0.0
            goal_fallback_count = 0
            blend_steps = 0
            avoid_steps = 0
            max_proximity = 0.0
            max_blend = 0.0
            min_clearance_est = float("inf")
            modes = []
            success = False

            previous_position = self_node.getPosition()
            previous_x = float(previous_position[0])
            previous_y = float(previous_position[1])

            print("-" * 112)
            print(
                f"TRIAL {trial}/{len(CASES)} | "
                f"{case['name']}"
            )
            print(
                f"start=({sx:+.2f},{sy:+.2f}) "
                f"goal=({gx:+.2f},{gy:+.2f}) "
                f"obstacle=({ox:+.2f},{oy:+.2f})"
            )

            while elapsed_s < TRIAL_TIMEOUT_S:
                if robot.step(TIME_STEP) == -1:
                    return

                elapsed_s += TIME_STEP / 1000.0
                steps += 1

                position = self_node.getPosition()
                x = float(position[0])
                y = float(position[1])

                path_length += math.hypot(
                    x - previous_x,
                    y - previous_y,
                )

                previous_x = x
                previous_y = y

                dx = gx - x
                dy = gy - y
                distance = math.hypot(dx, dy)

                min_clearance_est = min(
                    min_clearance_est,
                    estimated_box_clearance(
                        x,
                        y,
                        ox,
                        oy,
                    ),
                )

                if distance <= GOAL_TOLERANCE_M:
                    success = True
                    left_motor.setVelocity(0.0)
                    right_motor.setVelocity(0.0)
                    break

                robot_heading = combined.get_robot_heading(
                    self_node
                )

                goal_bearing = math.atan2(dy, dx)

                bearing_error = combined.wrap_pi(
                    goal_bearing - robot_heading
                )

                if (
                    bearing_error
                    > combined.NEUTRAL_DEADBAND_RAD
                ):
                    desired_neural = (
                        -combined.ACTION_STEERING
                    )
                elif (
                    bearing_error
                    < -combined.NEUTRAL_DEADBAND_RAD
                ):
                    desired_neural = (
                        +combined.ACTION_STEERING
                    )
                else:
                    desired_neural = 0.0

                heading_index, heading_label = (
                    combined.quantize_heading(
                        robot_heading
                    )
                )

                (
                    policy_heading,
                    goal_column,
                    goal_steering,
                    used_goal_fallback,
                ) = combined.choose_connectome_goal_action(
                    matrix,
                    heading_index,
                    desired_neural,
                )

                if used_goal_fallback:
                    goal_fallback_count += 1

                abs_error = abs(bearing_error)

                if abs_error > math.radians(90.0):
                    goal_forward = (
                        combined.GOAL_HARD_TURN_SPEED_FRACTION
                        * max_speed
                    )
                elif abs_error > math.radians(35.0):
                    goal_forward = (
                        combined.GOAL_TURN_SPEED_FRACTION
                        * max_speed
                    )
                else:
                    goal_forward = (
                        combined.GOAL_BASE_SPEED_FRACTION
                        * max_speed
                    )

                goal_turn = (
                    goal_steering
                    * combined.GOAL_STEERING_GAIN
                    * max_speed
                )

                obstacle = combined.read_obstacle_state(
                    sensors
                )

                max_sensor_raw = obstacle["max_raw"]
                max_proximity = max(
                    max_proximity,
                    max_sensor_raw,
                )

                if not obstacle_latched:
                    if (
                        max_sensor_raw
                        >= combined.OBSTACLE_ENTER_RAW
                    ):
                        obstacle_latched = True
                else:
                    if (
                        max_sensor_raw
                        <= combined.OBSTACLE_EXIT_RAW
                    ):
                        obstacle_latched = False

                raw_obstacle_steering = obstacle[
                    "raw_steering"
                ]

                if (
                    obstacle_latched
                    and obstacle["front_raw"]
                    >= combined.OBSTACLE_ENTER_RAW
                    and abs(raw_obstacle_steering)
                    < combined.OBSTACLE_TIE_EPS
                ):
                    if (
                        abs(goal_steering)
                        >= combined.GOAL_SIGN_MIN
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
                    >= combined.OBSTACLE_TIE_EPS
                ):
                    last_avoid_direction = (
                        +1.0
                        if raw_obstacle_steering > 0.0
                        else -1.0
                    )

                smoothed_obstacle_steering = (
                    (
                        1.0
                        - combined.OBSTACLE_SMOOTH_ALPHA
                    )
                    * smoothed_obstacle_steering
                    +
                    combined.OBSTACLE_SMOOTH_ALPHA
                    * raw_obstacle_steering
                )

                obstacle_speed_scale = combined.clamp(
                    (
                        1.0
                        - combined.OBSTACLE_SPEED_REDUCTION
                        * obstacle["average_obstacle"]
                    ),
                    combined.OBSTACLE_MIN_SPEED_SCALE,
                    1.0,
                )

                obstacle_forward = (
                    combined.OBSTACLE_BASE_SPEED_FRACTION
                    * max_speed
                    * obstacle_speed_scale
                )

                obstacle_turn = (
                    smoothed_obstacle_steering
                    * combined.OBSTACLE_STEERING_GAIN
                    * max_speed
                )

                if obstacle_latched:
                    blend = combined.obstacle_blend_weight(
                        max_sensor_raw
                    )
                    blend = max(blend, 0.15)
                else:
                    blend = 0.0

                max_blend = max(max_blend, blend)

                final_turn = (
                    (1.0 - blend) * goal_turn
                    + blend * obstacle_turn
                )

                final_forward = min(
                    goal_forward,
                    obstacle_forward,
                )

                if blend >= 0.75:
                    mode = "AVOID"
                    avoid_steps += 1
                elif blend > 0.0:
                    mode = "BLEND"
                    blend_steps += 1
                else:
                    mode = "GOAL"

                modes.append(mode)

                target_left = combined.clamp(
                    final_forward + final_turn,
                    -max_speed,
                    +max_speed,
                )

                target_right = combined.clamp(
                    final_forward - final_turn,
                    -max_speed,
                    +max_speed,
                )

                left_velocity = (
                    (
                        1.0
                        - combined.MOTOR_SMOOTH_ALPHA
                    )
                    * previous_left
                    +
                    combined.MOTOR_SMOOTH_ALPHA
                    * target_left
                )

                right_velocity = (
                    (
                        1.0
                        - combined.MOTOR_SMOOTH_ALPHA
                    )
                    * previous_right
                    +
                    combined.MOTOR_SMOOTH_ALPHA
                    * target_right
                )

                left_motor.setVelocity(left_velocity)
                right_motor.setVelocity(right_velocity)

                previous_left = left_velocity
                previous_right = right_velocity

                if steps % 50 == 0:
                    print(
                        f"  t={elapsed_s:5.2f}s "
                        f"dist={distance:.3f} "
                        f"mode={mode:<5} "
                        f"prox={max_sensor_raw:7.1f} "
                        f"blend={blend:.2f} "
                        f"EPG={heading_label:>2} "
                        f"FC2={goal_column}"
                    )

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            final_position = self_node.getPosition()

            final_distance = math.hypot(
                gx - float(final_position[0]),
                gy - float(final_position[1]),
            )

            obstacle_activated = (
                blend_steps > 0
                or avoid_steps > 0
            )

            expected = bool(
                case["expect_obstacle"]
            )

            behavior_ok = (
                obstacle_activated
                if expected
                else not obstacle_activated
            )

            case_pass = bool(
                success
                and behavior_ok
            )

            row = {
                "trial": trial,
                "case": case["name"],
                "pass": case_pass,
                "goal_reached": success,
                "expect_obstacle": expected,
                "obstacle_activated": obstacle_activated,
                "elapsed_s": round(elapsed_s, 4),
                "final_distance_m": round(
                    final_distance,
                    6,
                ),
                "path_length_m": round(
                    path_length,
                    6,
                ),
                "max_proximity": round(
                    max_proximity,
                    3,
                ),
                "max_obstacle_blend": round(
                    max_blend,
                    6,
                ),
                "blend_steps": blend_steps,
                "avoid_steps": avoid_steps,
                "goal_fallback_count":
                    goal_fallback_count,
                "mode_transitions":
                    collapse_modes(modes),
                "estimated_min_clearance_m":
                    round(
                        min_clearance_est,
                        6,
                    ),
            }

            results.append(row)

            status = (
                "PASS"
                if case_pass
                else "FAIL"
            )

            print(
                f"{status} | "
                f"goal={success} | "
                f"final={final_distance:.3f}m | "
                f"prox={max_proximity:.1f} | "
                f"blend={max_blend:.2f} | "
                f"modes={row['mode_transitions']} | "
                f"clear_est={min_clearance_est:.3f}m"
            )
            print()

        with output_file.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    results[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(results)

        passed = sum(
            1
            for row in results
            if row["pass"]
        )

        print("=" * 112)
        print("STEP 43 FINAL SUMMARY")
        print("=" * 112)

        for row in results:
            status = (
                "PASS"
                if row["pass"]
                else "FAIL"
            )

            print(
                f"{row['case']:<22} "
                f"{status:<4} | "
                f"goal={str(row['goal_reached']):<5} | "
                f"prox={row['max_proximity']:>7.1f} | "
                f"blend={row['max_obstacle_blend']:.2f} | "
                f"fallback={row['goal_fallback_count']:>3} | "
                f"{row['mode_transitions']}"
            )

        print()
        print(
            f"Passed regression cases: "
            f"{passed}/{len(results)}"
        )
        print(
            f"Results CSV: {output_file}"
        )

        if passed == len(results):
            print()
            print(
                "PASS: Combined Navigation v1 passed "
                "the controlled regression suite."
            )
            print(
                "NEXT: freeze combined-nav-v1 and move "
                "to randomized robustness validation."
            )
        else:
            print()
            print(
                "REVIEW: one or more controlled cases failed."
            )
            print(
                "Inspect only the failed cases before "
                "changing arbitration parameters."
            )

    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        robot_translation.setSFVec3f(
            original_robot_translation
        )
        robot_rotation.setSFRotation(
            original_robot_rotation
        )
        self_node.resetPhysics()

        obstacle_translation.setSFVec3f(
            original_obstacle_translation
        )
        obstacle_node.resetPhysics()

        if (
            goal_translation is not None
            and original_goal_translation is not None
        ):
            goal_translation.setSFVec3f(
                original_goal_translation
            )
            goal_node.resetPhysics()


if __name__ == "__main__":
    main()
