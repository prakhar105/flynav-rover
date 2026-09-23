import math
import sys
from pathlib import Path

from controller import Supervisor


# =========================================================
# Project import
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flynav_brain import FlyNavBrain


# =========================================================
# Webots
# =========================================================

robot = Supervisor()

TIME_STEP = int(robot.getBasicTimeStep())
MAX_SPEED = 6.28

# Current robot node.
# E-puck must have:
#
# supervisor TRUE
#
robot_node = robot.getSelf()


# =========================================================
# Temporary navigation goal
#
# IMPORTANT:
# The goal is NOT controlling the rover yet.
#
# For this step we only measure:
#
# current position
# current heading
# goal heading
# heading error
# distance to goal
#
# After verifying the coordinate convention,
# these values will feed EPG / FC2.
# =========================================================

GOAL_X = 0.30
GOAL_Y = 0.30


# =========================================================
# Motors
# =========================================================

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


# =========================================================
# Sensors
# =========================================================

sensors = []

for i in range(8):

    sensor = robot.getDevice(
        f"ps{i}"
    )

    sensor.enable(TIME_STEP)

    sensors.append(sensor)


# =========================================================
# Brain
# =========================================================

print(
    "Loading MaleCNS FlyNav brain..."
)

brain = FlyNavBrain(
    PROJECT_ROOT
)

print(
    "FlyNav connectome brain loaded."
)


# =========================================================
# Sensor calibration
#
# Measurements from our Webots world:
#
# Open space:
#     ~60-80
#
# Approaching obstacle:
#     ~200+
#
# Very close wall:
#     can exceed 1700
# =========================================================

SENSOR_FLOOR = 80.0
SENSOR_SATURATION = 650.0


# =========================================================
# Motor configuration
# =========================================================

BASE_SPEED = (
    0.28 * MAX_SPEED
)

STEERING_GAIN = (
    0.55 * MAX_SPEED
)

SMOOTHING_ALPHA = 0.25

smoothed_steering = 0.0


# =========================================================
# Helpers
# =========================================================

def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def encode_sensor(value):
    """
    Convert Webots proximity reading into continuous
    neural stimulus between 0 and 1.

    We preserve magnitude differences between the
    left and right sensor groups.
    """

    normalized = (
        value
        - SENSOR_FLOOR
    ) / (
        SENSOR_SATURATION
        - SENSOR_FLOOR
    )

    return clamp(
        normalized,
        0.0,
        1.0,
    )


def wrap_angle(angle):
    """
    Wrap an angle into:

        [-pi, +pi]
    """

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def get_navigation_state():
    """
    Read simulated rover pose and calculate
    direction/distance to the temporary goal.

    NOTE:
    The forward-axis convention is being tested in
    this stage. We are NOT yet using heading_error
    to control the rover.
    """

    # -----------------------------------------------------
    # Position
    # -----------------------------------------------------

    position = robot_node.getPosition()

    x = float(position[0])
    y = float(position[1])


    # -----------------------------------------------------
    # Orientation
    #
    # getOrientation() returns a flattened 3x3
    # rotation matrix.
    #
    # For this initial test we use the projected
    # local X direction as the robot heading.
    # -----------------------------------------------------

    orientation = robot_node.getOrientation()

    forward_x = orientation[0]
    forward_y = orientation[3]

    current_heading = math.atan2(
        forward_y,
        forward_x,
    )


    # -----------------------------------------------------
    # Goal direction
    # -----------------------------------------------------

    goal_dx = (
        GOAL_X
        - x
    )

    goal_dy = (
        GOAL_Y
        - y
    )

    goal_heading = math.atan2(
        goal_dy,
        goal_dx,
    )


    # -----------------------------------------------------
    # Heading error
    # -----------------------------------------------------

    heading_error = wrap_angle(
        goal_heading
        - current_heading
    )


    # -----------------------------------------------------
    # Distance to goal
    # -----------------------------------------------------

    distance_to_goal = math.sqrt(
        goal_dx ** 2
        +
        goal_dy ** 2
    )


    return {
        "x": x,
        "y": y,
        "heading": current_heading,
        "goal_heading": goal_heading,
        "heading_error": heading_error,
        "distance": distance_to_goal,
    }


# =========================================================
# Main loop
# =========================================================

counter = 0

print(
    "FlyNav neural rover controller V2 pose-test started."
)

print(
    f"Temporary goal: "
    f"X={GOAL_X:+.2f}, "
    f"Y={GOAL_Y:+.2f}"
)


while robot.step(TIME_STEP) != -1:

    # =====================================================
    # Rover navigation state
    # =====================================================

    nav = get_navigation_state()


    # =====================================================
    # Sensor readings
    # =====================================================

    values = [
        sensor.getValue()
        for sensor in sensors
    ]

    front_left = max(
        values[6],
        values[7],
    )

    front_right = max(
        values[0],
        values[1],
    )


    # =====================================================
    # Continuous sensory encoding
    # =====================================================

    obstacle_left = encode_sensor(
        front_left
    )

    obstacle_right = encode_sensor(
        front_right
    )


    # =====================================================
    # Connectome-derived neural computation
    #
    # At this stage ONLY obstacle stimuli enter the brain.
    #
    # Heading / goal signals are being measured but
    # not connected to EPG / FC2 yet.
    # =====================================================

    output = brain.step(
        obstacle_left=obstacle_left,
        obstacle_right=obstacle_right,
        duration_ms=TIME_STEP,
    )

    raw_steering = float(
        output["steering"]
    )


    # =====================================================
    # Smooth biological steering
    # =====================================================

    smoothed_steering = (
        (
            1.0
            - SMOOTHING_ALPHA
        )
        * smoothed_steering
        +
        SMOOTHING_ALPHA
        * raw_steering
    )


    # =====================================================
    # Forward-speed modulation
    # =====================================================

    average_obstacle = (
        obstacle_left
        + obstacle_right
    ) / 2.0

    speed_scale = (
        1.0
        - 0.85
        * average_obstacle
    )

    speed_scale = clamp(
        speed_scale,
        0.12,
        1.0,
    )

    forward_speed = (
        BASE_SPEED
        * speed_scale
    )


    # =====================================================
    # Neural steering -> differential drive
    #
    # Positive:
    #
    #     DNa02_R dominates
    #     RIGHT turn
    #
    # Negative:
    #
    #     DNa02_L dominates
    #     LEFT turn
    # =====================================================

    turn_component = (
        STEERING_GAIN
        * smoothed_steering
    )

    left_speed = (
        forward_speed
        + turn_component
    )

    right_speed = (
        forward_speed
        - turn_component
    )


    # =====================================================
    # Motor limits
    # =====================================================

    left_speed = clamp(
        left_speed,
        -MAX_SPEED,
        MAX_SPEED,
    )

    right_speed = clamp(
        right_speed,
        -MAX_SPEED,
        MAX_SPEED,
    )


    # =====================================================
    # Apply motor commands
    # =====================================================

    left_motor.setVelocity(
        left_speed
    )

    right_motor.setVelocity(
        right_speed
    )


    # =====================================================
    # Debug output
    # =====================================================

    if counter % 10 == 0:

        heading_deg = math.degrees(
            nav["heading"]
        )

        goal_heading_deg = math.degrees(
            nav["goal_heading"]
        )

        heading_error_deg = math.degrees(
            nav["heading_error"]
        )

        print(
            # ---------------------------------------------
            # Pose / navigation
            # ---------------------------------------------
            f"X={nav['x']:+.3f} | "
            f"Y={nav['y']:+.3f} | "
            f"HEAD={heading_deg:+6.1f}deg | "
            f"GOAL={goal_heading_deg:+6.1f}deg | "
            f"ERR={heading_error_deg:+6.1f}deg | "
            f"DIST={nav['distance']:.3f} | "

            # ---------------------------------------------
            # Proximity
            # ---------------------------------------------
            f"FL={front_left:6.1f} | "
            f"FR={front_right:6.1f} | "

            # ---------------------------------------------
            # Encoded neural stimulus
            # ---------------------------------------------
            f"SL={obstacle_left:.3f} | "
            f"SR={obstacle_right:.3f} | "

            # ---------------------------------------------
            # Biological outputs
            # ---------------------------------------------
            f"DNaL={output['left_spikes']:2d} | "
            f"DNaR={output['right_spikes']:2d} | "

            f"VL={output['left_membrane']:+.3f} | "
            f"VR={output['right_membrane']:+.3f} | "

            # ---------------------------------------------
            # Steering
            # ---------------------------------------------
            f"RAW={raw_steering:+.3f} | "
            f"SM={smoothed_steering:+.3f} | "

            # ---------------------------------------------
            # Motors
            # ---------------------------------------------
            f"ML={left_speed:+.2f} | "
            f"MR={right_speed:+.2f}"
        )

    counter += 1