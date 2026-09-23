import sys
from pathlib import Path

from controller import Robot

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

robot = Robot()

TIME_STEP = int(robot.getBasicTimeStep())
MAX_SPEED = 6.28


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
# From our measurements:
#
# open space ≈ 60–80
# approaching obstacle ≈ 200+
# close wall can exceed 1700
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
    neural stimulus in the range 0..1.

    Important:
    We deliberately preserve magnitude differences
    between the left and right sensors.
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


# =========================================================
# Main loop
# =========================================================

counter = 0

print(
    "FlyNav neural rover controller V2 started."
)


while robot.step(TIME_STEP) != -1:

    # -----------------------------------------------------
    # Sensor readings
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CONTINUOUS sensory encoding
    #
    # Example:
    #
    # FL = 300 -> ~0.39
    # FR = 270 -> ~0.33
    #
    # That difference is now preserved.
    # -----------------------------------------------------

    obstacle_left = encode_sensor(
        front_left
    )

    obstacle_right = encode_sensor(
        front_right
    )


    # -----------------------------------------------------
    # Connectome-derived neural computation
    # -----------------------------------------------------

    output = brain.step(
        obstacle_left=obstacle_left,
        obstacle_right=obstacle_right,
        duration_ms=TIME_STEP,
    )

    raw_steering = float(
        output["steering"]
    )


    # -----------------------------------------------------
    # Smooth biological output
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Forward speed
    #
    # Do NOT force forward speed completely to zero.
    #
    # Even when both sensors are active, the rover retains
    # a tiny crawl speed while the neural circuit decides
    # which direction to rotate.
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Neural steering -> differential drive
    #
    # + steering:
    # DNa02_R dominates
    # turn RIGHT
    #
    # - steering:
    # DNa02_L dominates
    # turn LEFT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Clamp
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Apply
    # -----------------------------------------------------

    left_motor.setVelocity(
        left_speed
    )

    right_motor.setVelocity(
        right_speed
    )


    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

    if counter % 10 == 0:

        print(
            f"FL={front_left:6.1f} | "
            f"FR={front_right:6.1f} | "
            f"SL={obstacle_left:.3f} | "
            f"SR={obstacle_right:.3f} | "
            f"DNaL={output['left_spikes']:2d} | "
            f"DNaR={output['right_spikes']:2d} | "
            f"VL={output['left_membrane']:+.3f} | "
            f"VR={output['right_membrane']:+.3f} | "
            f"RAW={raw_steering:+.3f} | "
            f"SM={smoothed_steering:+.3f} | "
            f"ML={left_speed:+.2f} | "
            f"MR={right_speed:+.2f}"
        )

    counter += 1