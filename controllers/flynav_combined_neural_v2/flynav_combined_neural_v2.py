import csv
import importlib.util
import math
import sys
from pathlib import Path

from controller import Supervisor


# ============================================================
# STEP 45 - FLYNAV COMBINED NAVIGATION NEURAL V2
#
# Purpose
# -------
# Preserve the frozen Goal Navigation v1 behavior while
# replacing the arithmetic obstacle steering used by
# flynav_combined_v1 with the ACTUAL frozen FlyNavBrain
# obstacle computation used by flynav_neural_v1.
#
# Biological / connectome-derived obstacle path:
#
#   Webots proximity
#        ↓
#   frozen sensor encoding
#        ↓
#   FlyNavBrain.step(...)
#        ↓
#   DNa-derived neural steering
#        ↓
#   frozen obstacle smoothing/gain
#        ↓
#   engineering arbitration with frozen goal branch
#
# The arbitration, hysteresis, blending, goal interface and
# final motor smoothing remain engineering robotics logic.
#
# IMPORTANT:
#   - flynav_neural_v1 is NOT modified.
#   - flynav_combined_v1 is NOT modified.
#   - frozen FC2 goal map is NOT modified.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flynav_brain import FlyNavBrain


# ============================================================
# Load the frozen Combined Navigation v1 module.
#
# We reuse its already-validated goal-navigation helpers and
# constants instead of copying/re-implementing them.
# ============================================================

FROZEN_COMBINED_PATH = (
    PROJECT_ROOT
    / "controllers"
    / "flynav_combined_v1"
    / "flynav_combined_v1.py"
)


def load_frozen_combined_module():
    if not FROZEN_COMBINED_PATH.exists():
        raise FileNotFoundError(
            "Frozen combined controller not found:\n"
            f"{FROZEN_COMBINED_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "flynav_frozen_combined_v1",
        FROZEN_COMBINED_PATH,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


base = load_frozen_combined_module()


# ============================================================
# Frozen constants inherited from Combined Navigation v1
# ============================================================

TIME_STEP = base.TIME_STEP

GOAL_FALLBACK_X = base.GOAL_FALLBACK_X
GOAL_FALLBACK_Y = base.GOAL_FALLBACK_Y
GOAL_TOLERANCE_M = base.GOAL_TOLERANCE_M

ACTION_STEERING = base.ACTION_STEERING
NEUTRAL_DEADBAND_RAD = base.NEUTRAL_DEADBAND_RAD
GOAL_SIGN_MIN = base.GOAL_SIGN_MIN

GOAL_BASE_SPEED_FRACTION = base.GOAL_BASE_SPEED_FRACTION
GOAL_TURN_SPEED_FRACTION = base.GOAL_TURN_SPEED_FRACTION
GOAL_HARD_TURN_SPEED_FRACTION = (
    base.GOAL_HARD_TURN_SPEED_FRACTION
)
GOAL_STEERING_GAIN = base.GOAL_STEERING_GAIN

SENSOR_FLOOR = base.SENSOR_FLOOR
SENSOR_SATURATION = base.SENSOR_SATURATION

OBSTACLE_BASE_SPEED_FRACTION = (
    base.OBSTACLE_BASE_SPEED_FRACTION
)
OBSTACLE_STEERING_GAIN = base.OBSTACLE_STEERING_GAIN
OBSTACLE_SMOOTH_ALPHA = base.OBSTACLE_SMOOTH_ALPHA

OBSTACLE_SPEED_REDUCTION = base.OBSTACLE_SPEED_REDUCTION
OBSTACLE_MIN_SPEED_SCALE = base.OBSTACLE_MIN_SPEED_SCALE

OBSTACLE_ENTER_RAW = base.OBSTACLE_ENTER_RAW
OBSTACLE_EXIT_RAW = base.OBSTACLE_EXIT_RAW
OBSTACLE_FULL_CONTROL_RAW = base.OBSTACLE_FULL_CONTROL_RAW

MOTOR_SMOOTH_ALPHA = base.MOTOR_SMOOTH_ALPHA

PRINT_EVERY_STEPS = 10
CSV_LOG_EVERY_STEPS = 5


# ============================================================
# Frozen obstacle-controller sensor interface.
#
# This matches flynav_neural_v1 exactly:
#
#   front_left  = max(ps6, ps7)
#   front_right = max(ps0, ps1)
#
# The old combined v1 used a wider ps5/6/7 vs ps0/1/2
# arithmetic side-difference. Step 45 intentionally removes
# that bypass from the steering computation.
# ============================================================

NEURAL_LEFT_SENSOR_NAMES = ["ps6", "ps7"]
NEURAL_RIGHT_SENSOR_NAMES = ["ps0", "ps1"]


# ============================================================
# Helpers
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def encode_sensor(value):
    """
    Frozen obstacle-controller encoding:
    Webots proximity -> continuous 0..1 neural stimulus.
    """

    normalized = (
        float(value) - SENSOR_FLOOR
    ) / (
        SENSOR_SATURATION - SENSOR_FLOOR
    )

    return clamp(
        normalized,
        0.0,
        1.0,
    )


def read_neural_obstacle_state(
    proximity_sensors,
    brain,
):
    """
    Read Webots proximity sensors and pass the SAME encoded
    left/right obstacle stimulus used by flynav_neural_v1
    through the actual FlyNavBrain neural computation.
    """

    raw = {
        name: float(
            proximity_sensors[name].getValue()
        )
        for name in proximity_sensors
    }

    front_left_raw = max(
        raw[name]
        for name in NEURAL_LEFT_SENSOR_NAMES
    )

    front_right_raw = max(
        raw[name]
        for name in NEURAL_RIGHT_SENSOR_NAMES
    )

    obstacle_left = encode_sensor(
        front_left_raw
    )

    obstacle_right = encode_sensor(
        front_right_raw
    )

    neural_output = brain.step(
        obstacle_left=obstacle_left,
        obstacle_right=obstacle_right,
        duration_ms=TIME_STEP,
    )

    raw_neural_steering = float(
        neural_output["steering"]
    )

    average_obstacle = (
        obstacle_left
        + obstacle_right
    ) / 2.0

    max_raw = max(
        front_left_raw,
        front_right_raw,
    )

    return {
        "raw_values": raw,
        "front_left_raw": front_left_raw,
        "front_right_raw": front_right_raw,
        "max_raw": max_raw,
        "left_activation": obstacle_left,
        "right_activation": obstacle_right,
        "average_obstacle": average_obstacle,
        "raw_neural_steering": raw_neural_steering,
        "left_spikes": int(
            neural_output["left_spikes"]
        ),
        "right_spikes": int(
            neural_output["right_spikes"]
        ),
        "left_membrane": float(
            neural_output["left_membrane"]
        ),
        "right_membrane": float(
            neural_output["right_membrane"]
        ),
    }


def open_log():
    output_path = (
        PROJECT_ROOT
        / "connectome"
        / "output"
        / "combined_neural_v2_runtime.csv"
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
        "front_left_raw",
        "front_right_raw",
        "left_obstacle",
        "right_obstacle",
        "dna_left_spikes",
        "dna_right_spikes",
        "dna_left_membrane",
        "dna_right_membrane",
        "raw_neural_obstacle_steering",
        "smoothed_neural_obstacle_steering",
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


# ============================================================
# Main
# ============================================================

def main():
    robot = Supervisor()

    basic_time_step = int(
        robot.getBasicTimeStep()
    )

    if basic_time_step != TIME_STEP:
        raise RuntimeError(
            "Step 45 expects the frozen 32 ms runtime. "
            f"World basicTimeStep={basic_time_step}, "
            f"frozen controller TIME_STEP={TIME_STEP}."
        )

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Combined neural controller requires "
            "supervisor TRUE."
        )

    # --------------------------------------------------------
    # Motors
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Proximity sensors
    # --------------------------------------------------------

    proximity_sensors = {}

    for index in range(8):
        name = f"ps{index}"

        sensor = robot.getDevice(name)
        sensor.enable(TIME_STEP)

        proximity_sensors[name] = sensor

    # --------------------------------------------------------
    # Frozen goal map
    # --------------------------------------------------------

    matrix = base.load_frozen_matrix()

    goal_node = robot.getFromDef(
        "GOAL"
    )

    # --------------------------------------------------------
    # ACTUAL frozen FlyNav neural obstacle brain
    # --------------------------------------------------------

    print()
    print(
        "[Combined Neural v2] "
        "Loading MaleCNS FlyNav obstacle brain..."
    )

    brain = FlyNavBrain(
        PROJECT_ROOT
    )

    print(
        "[Combined Neural v2] "
        "FlyNav neural obstacle brain loaded."
    )

    # --------------------------------------------------------
    # Runtime state
    # --------------------------------------------------------

    previous_left = 0.0
    previous_right = 0.0

    # This is the same output smoothing used by the frozen
    # neural obstacle controller.
    smoothed_obstacle_steering = 0.0

    obstacle_latched = False

    step_count = 0

    log_path, log_handle, log_writer = open_log()

    print()
    print("=" * 116)
    print(
        "STEP 45 - FLYNAV COMBINED NAVIGATION NEURAL V2"
    )
    print("=" * 116)
    print()
    print(
        "[Combined Neural v2] "
        "Frozen FC2 goal steering map loaded."
    )
    print(
        "[Combined Neural v2] "
        "Obstacle steering source = FlyNavBrain.step()."
    )
    print(
        "[Combined Neural v2] "
        "Obstacle sensor interface = "
        "max(ps6,ps7) vs max(ps0,ps1)."
    )
    print(
        "[Combined Neural v2] "
        "No arithmetic left-right obstacle steering bypass."
    )
    print(
        "[Combined Neural v2] "
        "Arbitration/hysteresis/blending remain "
        "engineering robotics logic."
    )
    print(
        "[Combined Neural v2] "
        f"Obstacle enter={OBSTACLE_ENTER_RAW:.0f}, "
        f"exit={OBSTACLE_EXIT_RAW:.0f}, "
        f"full={OBSTACLE_FULL_CONTROL_RAW:.0f}."
    )
    print(
        "[Combined Neural v2] "
        f"Runtime log: {log_path}"
    )

    if goal_node is not None:
        print(
            "[Combined Neural v2] "
            "Using DEF GOAL position."
        )
    else:
        print(
            "[Combined Neural v2] "
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

            # ------------------------------------------------
            # Robot and goal geometry
            # ------------------------------------------------

            position = self_node.getPosition()

            x = float(position[0])
            y = float(position[1])

            if goal_node is not None:
                goal_position = goal_node.getPosition()

                goal_x = float(
                    goal_position[0]
                )

                goal_y = float(
                    goal_position[1]
                )

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
                        "[Combined Neural v2] "
                        "GOAL REACHED | "
                        f"distance={distance:.3f} m"
                    )

                # Continue stepping so Webots stays alive.
                # The neural obstacle state is intentionally
                # not advanced once the goal is reached.
                continue

            # ------------------------------------------------
            # Frozen connectome-derived goal branch
            # ------------------------------------------------

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
            ) = base.choose_connectome_goal_action(
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

            # ------------------------------------------------
            # Frozen ACTUAL neural obstacle branch
            # ------------------------------------------------

            obstacle = read_neural_obstacle_state(
                proximity_sensors,
                brain,
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
                "raw_neural_steering"
            ]

            # Frozen obstacle-controller smoothing.
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

            # ------------------------------------------------
            # Engineering arbitration
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Differential drive + frozen combined motor
            # smoothing
            # ------------------------------------------------

            target_left = clamp(
                final_forward
                + final_turn,
                -max_speed,
                +max_speed,
            )

            target_right = clamp(
                final_forward
                - final_turn,
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

            # ------------------------------------------------
            # Console diagnostics
            # ------------------------------------------------

            if (
                step_count
                % PRINT_EVERY_STEPS
                == 0
            ):
                print(
                    "[Combined Neural v2] "
                    f"mode={mode:<5} "
                    f"dist={distance:.3f} "
                    f"err={math.degrees(bearing_error):+6.1f}deg "
                    f"EPG={heading_label:>2} "
                    f"policy={policy_heading:>2} "
                    f"FC2={goal_column} "
                    f"goal={goal_steering:+.3f} "
                    f"FL={obstacle['front_left_raw']:7.1f} "
                    f"FR={obstacle['front_right_raw']:7.1f} "
                    f"DNa=({obstacle['left_spikes']},"
                    f"{obstacle['right_spikes']}) "
                    f"nRAW={raw_obstacle_steering:+.3f} "
                    f"nSM={smoothed_obstacle_steering:+.3f} "
                    f"blend={blend:.2f} "
                    f"wheels=({left_velocity:+.2f},"
                    f"{right_velocity:+.2f})"
                )

            # ------------------------------------------------
            # CSV diagnostics
            # ------------------------------------------------

            if (
                step_count
                % CSV_LOG_EVERY_STEPS
                == 0
            ):
                log_writer.writerow(
                    {
                        "step": step_count,
                        "time_s":
                            f"{time_s:.4f}",
                        "x":
                            f"{x:.6f}",
                        "y":
                            f"{y:.6f}",
                        "goal_x":
                            f"{goal_x:.6f}",
                        "goal_y":
                            f"{goal_y:.6f}",
                        "distance_m":
                            f"{distance:.6f}",
                        "heading_deg":
                            f"{math.degrees(robot_heading):.4f}",
                        "bearing_error_deg":
                            f"{math.degrees(bearing_error):.4f}",
                        "epg_heading":
                            heading_label,
                        "policy_epg":
                            policy_heading,
                        "fc2_column":
                            goal_column,
                        "goal_steering":
                            f"{goal_steering:.6f}",
                        "goal_fallback":
                            int(used_goal_fallback),
                        "front_left_raw":
                            f"{obstacle['front_left_raw']:.3f}",
                        "front_right_raw":
                            f"{obstacle['front_right_raw']:.3f}",
                        "left_obstacle":
                            f"{obstacle['left_activation']:.6f}",
                        "right_obstacle":
                            f"{obstacle['right_activation']:.6f}",
                        "dna_left_spikes":
                            obstacle["left_spikes"],
                        "dna_right_spikes":
                            obstacle["right_spikes"],
                        "dna_left_membrane":
                            f"{obstacle['left_membrane']:.6f}",
                        "dna_right_membrane":
                            f"{obstacle['right_membrane']:.6f}",
                        "raw_neural_obstacle_steering":
                            f"{raw_obstacle_steering:.6f}",
                        "smoothed_neural_obstacle_steering":
                            f"{smoothed_obstacle_steering:.6f}",
                        "obstacle_blend":
                            f"{blend:.6f}",
                        "mode":
                            mode,
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
