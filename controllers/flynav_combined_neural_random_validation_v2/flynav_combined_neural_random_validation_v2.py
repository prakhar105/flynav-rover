import csv
import importlib.util
import math
import random
import statistics
from pathlib import Path

from controller import Supervisor


# ============================================================
# STEP 45D - COMBINED NEURAL V2 RANDOMIZED ROBUSTNESS
#
# Purpose:
#   Re-run the SAME 30 randomized scenario distribution and
#   SAME deterministic seed used by Step 44, but with the
#   actual FlyNavBrain obstacle branch from Combined Neural v2.
#
# Primary comparison metric:
#   goal success rate (baseline Step 44 = 30/30)
#
# Additional neural diagnostics:
#   DNa spike activity
#   max absolute neural steering
#   obstacle blend / AVOID usage
#   goal fallback usage
#
# This validator does NOT modify the frozen controllers.
# ============================================================

TIME_STEP = 32
TRIAL_TIMEOUT_S = 45.0
SETTLE_STEPS = 8

NUM_TRIALS = 30
RANDOM_SEED = 20260924

GOAL_TOLERANCE_M = 0.035

SAMPLE_LIMIT = 0.36
NEAR_WALL_COORD = 0.31

MIN_START_GOAL_DISTANCE = 0.42
MAX_START_GOAL_DISTANCE = 0.78

OBSTACLE_NAME = "test obstacle"
OBSTACLE_SIZE_X = 0.08
OBSTACLE_SIZE_Y = 0.08
ROBOT_RADIUS_EST_M = 0.035

# MUST match Step 44.
SCENARIO_WEIGHTS = [
    ("clear", 0.15),
    ("direct", 0.35),
    ("offset", 0.35),
    ("near_wall", 0.15),
]


def weighted_choice(rng):
    value = rng.random()
    cumulative = 0.0

    for name, weight in SCENARIO_WEIGHTS:
        cumulative += weight
        if value <= cumulative:
            return name

    return SCENARIO_WEIGHTS[-1][0]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def load_neural_controller(project_root):
    source = (
        project_root
        / "controllers"
        / "flynav_combined_neural_v2"
        / "flynav_combined_neural_v2.py"
    )

    if not source.exists():
        raise FileNotFoundError(
            "Could not find Combined Neural v2:\n"
            f"{source}"
        )

    spec = importlib.util.spec_from_file_location(
        "flynav_combined_neural_v2_impl",
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

    for index in range(children.getCount()):
        candidate = children.getMFNode(index)
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


def point_inside_sampling_area(x, y, margin=0.0):
    limit = SAMPLE_LIMIT - margin

    return (
        -limit <= x <= limit
        and -limit <= y <= limit
    )


def sample_start_goal(rng):
    for _ in range(5000):
        sx = rng.uniform(
            -SAMPLE_LIMIT,
            SAMPLE_LIMIT,
        )

        sy = rng.uniform(
            -SAMPLE_LIMIT,
            SAMPLE_LIMIT,
        )

        gx = rng.uniform(
            -SAMPLE_LIMIT,
            SAMPLE_LIMIT,
        )

        gy = rng.uniform(
            -SAMPLE_LIMIT,
            SAMPLE_LIMIT,
        )

        distance = math.hypot(
            gx - sx,
            gy - sy,
        )

        if (
            MIN_START_GOAL_DISTANCE
            <= distance
            <= MAX_START_GOAL_DISTANCE
        ):
            return sx, sy, gx, gy

    raise RuntimeError(
        "Could not generate valid start/goal pair."
    )


def build_random_trial(rng, trial_index):
    # This function intentionally mirrors Step 44 exactly
    # so RANDOM_SEED=20260924 reproduces the same scenarios.
    scenario = weighted_choice(rng)

    if scenario == "near_wall":
        horizontal = rng.random() < 0.5
        wall_sign = rng.choice([-1.0, +1.0])

        if horizontal:
            y = wall_sign * NEAR_WALL_COORD

            if rng.random() < 0.5:
                sx, gx = -0.32, +0.32
            else:
                sx, gx = +0.32, -0.32

            sy = y
            gy = y

        else:
            x = wall_sign * NEAR_WALL_COORD

            if rng.random() < 0.5:
                sy, gy = -0.32, +0.32
            else:
                sy, gy = +0.32, -0.32

            sx = x
            gx = x

    else:
        sx, sy, gx, gy = sample_start_goal(rng)

    dx = gx - sx
    dy = gy - sy
    distance = math.hypot(dx, dy)

    ux = dx / distance
    uy = dy / distance

    px = -uy
    py = ux

    if scenario == "clear":
        midpoint_x = (sx + gx) / 2.0
        midpoint_y = (sy + gy) / 2.0

        side = rng.choice([-1.0, +1.0])

        ox = midpoint_x + side * px * 0.24
        oy = midpoint_y + side * py * 0.24

        if not point_inside_sampling_area(
            ox,
            oy,
            margin=0.02,
        ):
            candidates = [
                (-0.30, -0.30),
                (-0.30, +0.30),
                (+0.30, -0.30),
                (+0.30, +0.30),
            ]

            candidates.sort(
                key=lambda point: min(
                    math.hypot(
                        point[0] - sx,
                        point[1] - sy,
                    ),
                    math.hypot(
                        point[0] - gx,
                        point[1] - gy,
                    ),
                ),
                reverse=True,
            )

            ox, oy = candidates[0]

        expect_obstacle = False

    else:
        t = rng.uniform(0.38, 0.62)

        base_x = sx + t * dx
        base_y = sy + t * dy

        if scenario == "direct":
            lateral_offset = rng.uniform(
                -0.015,
                +0.015,
            )

        elif scenario == "offset":
            magnitude = rng.uniform(
                0.025,
                0.065,
            )

            lateral_offset = (
                magnitude
                * rng.choice([-1.0, +1.0])
            )

        else:
            lateral_offset = rng.uniform(
                -0.025,
                +0.025,
            )

        ox = base_x + px * lateral_offset
        oy = base_y + py * lateral_offset

        ox = clamp(
            ox,
            -SAMPLE_LIMIT + 0.04,
            +SAMPLE_LIMIT - 0.04,
        )

        oy = clamp(
            oy,
            -SAMPLE_LIMIT + 0.04,
            +SAMPLE_LIMIT - 0.04,
        )

        expect_obstacle = True

    heading_deg = rng.uniform(
        -180.0,
        +180.0,
    )

    return {
        "trial": trial_index,
        "scenario": scenario,
        "start": (sx, sy),
        "goal": (gx, gy),
        "obstacle": (ox, oy),
        "heading_deg": heading_deg,
        "expect_obstacle": expect_obstacle,
    }


def collapse_modes(modes):
    if not modes:
        return ""

    result = [modes[0]]

    for mode in modes[1:]:
        if mode != result[-1]:
            result.append(mode)

    return "->".join(result)


def main():
    rng = random.Random(RANDOM_SEED)

    robot = Supervisor()

    project_root = Path(__file__).resolve().parents[2]
    neural = load_neural_controller(project_root)
    base = neural.base

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 45D requires supervisor TRUE."
        )

    robot_translation = self_node.getField(
        "translation"
    )

    robot_rotation = self_node.getField(
        "rotation"
    )

    original_robot_translation = list(
        robot_translation.getSFVec3f()
    )

    original_robot_rotation = list(
        robot_rotation.getSFRotation()
    )

    robot_z = float(
        original_robot_translation[2]
    )

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

    obstacle_z = float(
        original_obstacle_translation[2]
    )

    left_motor = robot.getDevice(
        "left wheel motor"
    )

    right_motor = robot.getDevice(
        "right wheel motor"
    )

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    max_speed = min(
        float(left_motor.getMaxVelocity()),
        float(right_motor.getMaxVelocity()),
    )

    sensors = {}

    for index in range(8):
        name = f"ps{index}"

        sensor = robot.getDevice(name)
        sensor.enable(TIME_STEP)

        sensors[name] = sensor

    matrix = base.load_frozen_matrix()

    output_file = (
        project_root
        / "connectome"
        / "output"
        / "combined_neural_v2_randomized.csv"
    )

    trials = [
        build_random_trial(
            rng,
            trial_index,
        )
        for trial_index in range(
            1,
            NUM_TRIALS + 1,
        )
    ]

    results = []

    print()
    print("=" * 122)
    print(
        "STEP 45D - COMBINED NEURAL V2 "
        "RANDOMIZED ROBUSTNESS"
    )
    print("=" * 122)
    print(
        f"Trials: {NUM_TRIALS}"
    )
    print(
        f"Seed: {RANDOM_SEED}"
    )
    print(
        f"Timeout per trial: "
        f"{TRIAL_TIMEOUT_S:.1f}s"
    )
    print(
        "Comparison target: Step 44 arithmetic-combined "
        "baseline = 30/30 goals reached."
    )
    print()

    try:
        for trial in trials:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            trial_id = trial["trial"]
            scenario = trial["scenario"]

            sx, sy = trial["start"]
            gx, gy = trial["goal"]
            ox, oy = trial["obstacle"]

            heading_deg = trial[
                "heading_deg"
            ]

            robot_translation.setSFVec3f(
                [
                    sx,
                    sy,
                    robot_z,
                ]
            )

            robot_rotation.setSFRotation(
                [
                    0.0,
                    0.0,
                    1.0,
                    math.radians(
                        heading_deg
                    ),
                ]
            )

            self_node.resetPhysics()

            obstacle_translation.setSFVec3f(
                [
                    ox,
                    oy,
                    obstacle_z,
                ]
            )

            obstacle_node.resetPhysics()

            for _ in range(SETTLE_STEPS):
                if robot.step(TIME_STEP) == -1:
                    return

            # Fresh neural state per randomized trial.
            brain = neural.FlyNavBrain(
                project_root
            )

            previous_left = 0.0
            previous_right = 0.0

            smoothed_neural_steering = 0.0
            obstacle_latched = False

            elapsed_s = 0.0
            steps = 0
            path_length = 0.0

            goal_fallback_count = 0
            blend_steps = 0
            avoid_steps = 0
            neural_active_steps = 0

            max_proximity = 0.0
            max_blend = 0.0
            max_abs_neural_steering = 0.0
            max_left_spikes = 0
            max_right_spikes = 0

            min_clearance_est = float(
                "inf"
            )

            modes = []
            success = False

            previous_position = (
                self_node.getPosition()
            )

            previous_x = float(
                previous_position[0]
            )

            previous_y = float(
                previous_position[1]
            )

            start_goal_distance = math.hypot(
                gx - sx,
                gy - sy,
            )

            print(
                f"[{trial_id:02d}/{NUM_TRIALS}] "
                f"{scenario:<9} "
                f"start=({sx:+.2f},{sy:+.2f}) "
                f"goal=({gx:+.2f},{gy:+.2f}) "
                f"obs=({ox:+.2f},{oy:+.2f}) "
                f"heading={heading_deg:+.1f}deg"
            )

            while elapsed_s < TRIAL_TIMEOUT_S:
                if robot.step(TIME_STEP) == -1:
                    return

                elapsed_s += (
                    TIME_STEP / 1000.0
                )

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

                distance = math.hypot(
                    dx,
                    dy,
                )

                min_clearance_est = min(
                    min_clearance_est,
                    estimated_box_clearance(
                        x,
                        y,
                        ox,
                        oy,
                    ),
                )

                if (
                    distance
                    <= GOAL_TOLERANCE_M
                ):
                    success = True

                    left_motor.setVelocity(0.0)
                    right_motor.setVelocity(0.0)

                    break

                # --------------------------------------------
                # Frozen goal branch
                # --------------------------------------------

                robot_heading = (
                    base.get_robot_heading(
                        self_node
                    )
                )

                goal_bearing = math.atan2(
                    dy,
                    dx,
                )

                bearing_error = base.wrap_pi(
                    goal_bearing
                    - robot_heading
                )

                if (
                    bearing_error
                    > neural.NEUTRAL_DEADBAND_RAD
                ):
                    desired_neural = (
                        -neural.ACTION_STEERING
                    )

                elif (
                    bearing_error
                    < -neural.NEUTRAL_DEADBAND_RAD
                ):
                    desired_neural = (
                        +neural.ACTION_STEERING
                    )

                else:
                    desired_neural = 0.0

                (
                    heading_index,
                    heading_label,
                ) = base.quantize_heading(
                    robot_heading
                )

                (
                    policy_heading,
                    goal_column,
                    goal_steering,
                    used_goal_fallback,
                ) = (
                    base.choose_connectome_goal_action(
                        matrix,
                        heading_index,
                        desired_neural,
                    )
                )

                if used_goal_fallback:
                    goal_fallback_count += 1

                abs_error = abs(
                    bearing_error
                )

                if (
                    abs_error
                    > math.radians(90.0)
                ):
                    goal_forward = (
                        neural.GOAL_HARD_TURN_SPEED_FRACTION
                        * max_speed
                    )

                elif (
                    abs_error
                    > math.radians(35.0)
                ):
                    goal_forward = (
                        neural.GOAL_TURN_SPEED_FRACTION
                        * max_speed
                    )

                else:
                    goal_forward = (
                        neural.GOAL_BASE_SPEED_FRACTION
                        * max_speed
                    )

                goal_turn = (
                    goal_steering
                    * neural.GOAL_STEERING_GAIN
                    * max_speed
                )

                # --------------------------------------------
                # Actual FlyNavBrain obstacle branch
                # --------------------------------------------

                obstacle = (
                    neural.read_neural_obstacle_state(
                        sensors,
                        brain,
                    )
                )

                max_sensor_raw = obstacle[
                    "max_raw"
                ]

                max_proximity = max(
                    max_proximity,
                    max_sensor_raw,
                )

                raw_neural_steering = obstacle[
                    "raw_neural_steering"
                ]

                max_abs_neural_steering = max(
                    max_abs_neural_steering,
                    abs(raw_neural_steering),
                )

                max_left_spikes = max(
                    max_left_spikes,
                    obstacle["left_spikes"],
                )

                max_right_spikes = max(
                    max_right_spikes,
                    obstacle["right_spikes"],
                )

                if (
                    abs(raw_neural_steering) > 1e-6
                    or obstacle["left_spikes"] > 0
                    or obstacle["right_spikes"] > 0
                ):
                    neural_active_steps += 1

                if not obstacle_latched:
                    if (
                        max_sensor_raw
                        >= neural.OBSTACLE_ENTER_RAW
                    ):
                        obstacle_latched = True

                else:
                    if (
                        max_sensor_raw
                        <= neural.OBSTACLE_EXIT_RAW
                    ):
                        obstacle_latched = False

                smoothed_neural_steering = (
                    (
                        1.0
                        - neural.OBSTACLE_SMOOTH_ALPHA
                    )
                    * smoothed_neural_steering
                    +
                    neural.OBSTACLE_SMOOTH_ALPHA
                    * raw_neural_steering
                )

                obstacle_speed_scale = (
                    neural.clamp(
                        (
                            1.0
                            - neural.OBSTACLE_SPEED_REDUCTION
                            * obstacle[
                                "average_obstacle"
                            ]
                        ),
                        neural.OBSTACLE_MIN_SPEED_SCALE,
                        1.0,
                    )
                )

                obstacle_forward = (
                    neural.OBSTACLE_BASE_SPEED_FRACTION
                    * max_speed
                    * obstacle_speed_scale
                )

                obstacle_turn = (
                    smoothed_neural_steering
                    * neural.OBSTACLE_STEERING_GAIN
                    * max_speed
                )

                # --------------------------------------------
                # Engineering arbitration
                # --------------------------------------------

                if obstacle_latched:
                    blend = (
                        base.obstacle_blend_weight(
                            max_sensor_raw
                        )
                    )

                    blend = max(
                        blend,
                        0.15,
                    )

                else:
                    blend = 0.0

                max_blend = max(
                    max_blend,
                    blend,
                )

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
                    avoid_steps += 1

                elif blend > 0.0:
                    mode = "BLEND"
                    blend_steps += 1

                else:
                    mode = "GOAL"

                modes.append(mode)

                target_left = (
                    neural.clamp(
                        final_forward
                        + final_turn,
                        -max_speed,
                        +max_speed,
                    )
                )

                target_right = (
                    neural.clamp(
                        final_forward
                        - final_turn,
                        -max_speed,
                        +max_speed,
                    )
                )

                left_velocity = (
                    (
                        1.0
                        - neural.MOTOR_SMOOTH_ALPHA
                    )
                    * previous_left
                    +
                    neural.MOTOR_SMOOTH_ALPHA
                    * target_left
                )

                right_velocity = (
                    (
                        1.0
                        - neural.MOTOR_SMOOTH_ALPHA
                    )
                    * previous_right
                    +
                    neural.MOTOR_SMOOTH_ALPHA
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

            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)

            final_position = self_node.getPosition()

            final_x = float(
                final_position[0]
            )

            final_y = float(
                final_position[1]
            )

            final_distance = math.hypot(
                gx - final_x,
                gy - final_y,
            )

            obstacle_activated = (
                blend_steps > 0
                or avoid_steps > 0
            )

            path_efficiency = (
                start_goal_distance
                / path_length
                if path_length > 0.0
                else 0.0
            )

            result = {
                "trial":
                    trial_id,

                "seed":
                    RANDOM_SEED,

                "scenario":
                    scenario,

                "success":
                    success,

                "timeout":
                    not success,

                "start_x":
                    round(sx, 6),

                "start_y":
                    round(sy, 6),

                "goal_x":
                    round(gx, 6),

                "goal_y":
                    round(gy, 6),

                "obstacle_x":
                    round(ox, 6),

                "obstacle_y":
                    round(oy, 6),

                "initial_heading_deg":
                    round(
                        heading_deg,
                        4,
                    ),

                "start_goal_distance_m":
                    round(
                        start_goal_distance,
                        6,
                    ),

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

                "path_length_m":
                    round(
                        path_length,
                        6,
                    ),

                "path_efficiency":
                    round(
                        path_efficiency,
                        6,
                    ),

                "expect_obstacle":
                    bool(
                        trial["expect_obstacle"]
                    ),

                "obstacle_activated":
                    obstacle_activated,

                "neural_active_steps":
                    neural_active_steps,

                "max_proximity":
                    round(
                        max_proximity,
                        3,
                    ),

                "max_obstacle_blend":
                    round(
                        max_blend,
                        6,
                    ),

                "max_abs_neural_steering":
                    round(
                        max_abs_neural_steering,
                        6,
                    ),

                "max_left_dna_spikes":
                    max_left_spikes,

                "max_right_dna_spikes":
                    max_right_spikes,

                "blend_steps":
                    blend_steps,

                "avoid_steps":
                    avoid_steps,

                "goal_fallback_count":
                    goal_fallback_count,

                "mode_transitions":
                    collapse_modes(
                        modes
                    ),

                "estimated_min_clearance_m":
                    round(
                        min_clearance_est,
                        6,
                    ),
            }

            results.append(result)

            print(
                f"   {'PASS' if success else 'FAIL'} "
                f"time={elapsed_s:5.2f}s "
                f"final={final_distance:.3f}m "
                f"prox={max_proximity:7.1f} "
                f"blend={max_blend:.2f} "
                f"nmax={max_abs_neural_steering:.3f} "
                f"DNa=({max_left_spikes},{max_right_spikes}) "
                f"fallback={goal_fallback_count}"
            )

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
                    results[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(results)

        successes = [
            row
            for row in results
            if row["success"]
        ]

        failures = [
            row
            for row in results
            if not row["success"]
        ]

        success_rate = (
            100.0
            * len(successes)
            / len(results)
        )

        successful_times = [
            float(row["elapsed_s"])
            for row in successes
        ]

        successful_efficiencies = [
            float(row["path_efficiency"])
            for row in successes
        ]

        obstacle_trials = [
            row
            for row in results
            if row["expect_obstacle"]
        ]

        obstacle_neural_activated = sum(
            1
            for row in obstacle_trials
            if (
                row["obstacle_activated"]
                and row["neural_active_steps"] > 0
            )
        )

        print()
        print("=" * 122)
        print(
            "STEP 45D FINAL SUMMARY"
        )
        print("=" * 122)

        print(
            f"Success: "
            f"{len(successes)}/{len(results)} "
            f"({success_rate:.1f}%)"
        )

        if successful_times:
            print(
                f"Successful trial time: "
                f"mean={statistics.mean(successful_times):.2f}s "
                f"median={statistics.median(successful_times):.2f}s "
                f"max={max(successful_times):.2f}s"
            )

        if successful_efficiencies:
            print(
                f"Path efficiency: "
                f"mean={statistics.mean(successful_efficiencies):.3f} "
                f"median={statistics.median(successful_efficiencies):.3f}"
            )

        print(
            f"Expected-obstacle trials with both neural "
            f"activity and arbitration activation: "
            f"{obstacle_neural_activated}/"
            f"{len(obstacle_trials)}"
        )

        print()

        for scenario_name, _ in SCENARIO_WEIGHTS:
            subset = [
                row
                for row in results
                if row["scenario"] == scenario_name
            ]

            subset_success = sum(
                1
                for row in subset
                if row["success"]
            )

            if subset:
                rate = (
                    100.0
                    * subset_success
                    / len(subset)
                )

                print(
                    f"{scenario_name:<10} "
                    f"{subset_success:>2}/{len(subset):<2} "
                    f"({rate:5.1f}%)"
                )

        print()
        print(
            f"Results CSV: "
            f"{output_file}"
        )

        print()
        print(
            "BASELINE COMPARISON:"
        )
        print(
            "  Step 44 arithmetic-combined: "
            "30/30 goals reached"
        )
        print(
            f"  Step 45D neural-combined:    "
            f"{len(successes)}/{len(results)} goals reached"
        )

        if failures:
            print()
            print(
                "FAILED TRIALS:"
            )

            for row in failures:
                print(
                    f"  trial={row['trial']:02d} "
                    f"scenario={row['scenario']} "
                    f"final={row['final_distance_m']:.3f}m "
                    f"prox={row['max_proximity']:.1f} "
                    f"blend={row['max_obstacle_blend']:.2f} "
                    f"nmax={row['max_abs_neural_steering']:.3f} "
                    f"DNa=({row['max_left_dna_spikes']},"
                    f"{row['max_right_dna_spikes']}) "
                    f"modes={row['mode_transitions']}"
                )

            print()
            print(
                "Do not tune immediately. Because this uses "
                "the same deterministic seed as Step 44, "
                "compare each failed trial directly with the "
                "arithmetic-combined baseline."
            )

        elif len(successes) == NUM_TRIALS:
            print()
            print(
                "PASS: Combined Neural v2 preserved the "
                "30/30 randomized goal-success baseline."
            )
            print(
                "NEXT: freeze Combined Neural v2 and document "
                "the neural-vs-engineering architecture."
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


if __name__ == "__main__":
    main()
