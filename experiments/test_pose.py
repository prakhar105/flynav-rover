from controller import Supervisor
import math


robot = Supervisor()

TIME_STEP = int(robot.getBasicTimeStep())

node = robot.getSelf()

print("FlyNav pose test started")


while robot.step(TIME_STEP) != -1:

    position = node.getPosition()

    orientation = node.getOrientation()

    # Robot forward axis projected onto arena XY plane.
    #
    # Webots orientation is a 3x3 rotation matrix
    # flattened into a list.
    forward_x = orientation[0]
    forward_y = orientation[3]

    heading = math.atan2(
        forward_y,
        forward_x,
    )

    print(
        f"x={position[0]:+.3f} | "
        f"y={position[1]:+.3f} | "
        f"heading={math.degrees(heading):+.1f}°"
    )