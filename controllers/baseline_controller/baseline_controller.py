from controller import Robot

robot = Robot()

TIME_STEP = int(robot.getBasicTimeStep())
MAX_SPEED = 6.28

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FORWARD_SPEED = 0.30 * MAX_SPEED
TURN_SPEED = 0.30 * MAX_SPEED
REVERSE_SPEED = 0.22 * MAX_SPEED

OBSTACLE_THRESHOLD = 250.0

# If both front sides detect an obstacle, treat it as a
# corner / blocked-front situation.
BOTH_BLOCKED_THRESHOLD = 250.0

# Timed escape manoeuvre
REVERSE_TIME = 0.40   # seconds
TURN_TIME = 0.65      # seconds

REVERSE_STEPS = max(
    1,
    int(REVERSE_TIME * 1000 / TIME_STEP)
)

TURN_STEPS = max(
    1,
    int(TURN_TIME * 1000 / TIME_STEP)
)

# ---------------------------------------------------------
# Motors
# ---------------------------------------------------------

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ---------------------------------------------------------
# Proximity sensors
# ---------------------------------------------------------

proximity_sensors = []

for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(TIME_STEP)
    proximity_sensors.append(sensor)

# ---------------------------------------------------------
# Controller state
# ---------------------------------------------------------

state = "FORWARD"
state_counter = 0

# Used to break perfect corner symmetry.
# 1 = right, -1 = left
escape_direction = 1

counter = 0

print("FlyNav classical baseline v2 started")

# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while robot.step(TIME_STEP) != -1:

    sensor_values = [
        sensor.getValue()
        for sensor in proximity_sensors
    ]

    # Front sensing
    front_left = max(
        sensor_values[6],
        sensor_values[7]
    )

    front_right = max(
        sensor_values[0],
        sensor_values[1]
    )

    max_front = max(front_left, front_right)

    # =====================================================
    # STATE: reversing from a corner
    # =====================================================

    if state == "ESCAPE_REVERSE":

        left_speed = -REVERSE_SPEED
        right_speed = -REVERSE_SPEED

        state_counter -= 1

        if state_counter <= 0:
            state = "ESCAPE_TURN"
            state_counter = TURN_STEPS

    # =====================================================
    # STATE: committed escape turn
    # =====================================================

    elif state == "ESCAPE_TURN":

        if escape_direction == 1:
            # Turn right
            left_speed = TURN_SPEED
            right_speed = -TURN_SPEED
        else:
            # Turn left
            left_speed = -TURN_SPEED
            right_speed = TURN_SPEED

        state_counter -= 1

        if state_counter <= 0:
            state = "FORWARD"

    # =====================================================
    # NORMAL REACTIVE NAVIGATION
    # =====================================================

    else:

        both_blocked = (
            front_left > BOTH_BLOCKED_THRESHOLD
            and front_right > BOTH_BLOCKED_THRESHOLD
        )

        # ---------------------------------------------
        # Corner / blocked directly ahead
        # ---------------------------------------------

        if both_blocked:

            state = "ESCAPE_REVERSE"
            state_counter = REVERSE_STEPS

            # Choose the direction with slightly more space.
            if front_left > front_right:
                escape_direction = 1   # turn right
            elif front_right > front_left:
                escape_direction = -1  # turn left
            else:
                # Exact tie: alternate
                escape_direction *= -1

            left_speed = -REVERSE_SPEED
            right_speed = -REVERSE_SPEED

        # ---------------------------------------------
        # Obstacle on left
        # ---------------------------------------------

        elif front_left > OBSTACLE_THRESHOLD:

            state = "TURN_RIGHT"

            left_speed = TURN_SPEED
            right_speed = -TURN_SPEED

        # ---------------------------------------------
        # Obstacle on right
        # ---------------------------------------------

        elif front_right > OBSTACLE_THRESHOLD:

            state = "TURN_LEFT"

            left_speed = -TURN_SPEED
            right_speed = TURN_SPEED

        # ---------------------------------------------
        # Clear
        # ---------------------------------------------

        else:

            state = "FORWARD"

            left_speed = FORWARD_SPEED
            right_speed = FORWARD_SPEED

    # ---------------------------------------------------------
    # Apply motor commands
    # ---------------------------------------------------------

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # ---------------------------------------------------------
    # Debug
    # ---------------------------------------------------------

    if counter % 10 == 0:
        print(
            f"{state:<15} | "
            f"LEFT={front_left:7.1f} | "
            f"RIGHT={front_right:7.1f} | "
            f"MAX={max_front:7.1f}"
        )

    counter += 1