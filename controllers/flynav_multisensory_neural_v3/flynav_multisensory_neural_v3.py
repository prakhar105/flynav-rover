import csv
import math
import sys
from pathlib import Path

import pandas as pd
from controller import Camera, Supervisor


# ============================================================
# STEP 60 - MULTISENSORY NEURAL NAVIGATION V3 - HEADING-ASSISTED VISUAL MEMORY
#
# Controlled fusion test:
#
#   CAMERA
#     -> engineered visual-field encoder
#     -> LC10a -> AOTU019 (GABA) -> DNa02 visual steering
#
#   PROXIMITY
#     -> frozen Webots sensor encoding
#     -> FlyNavBrain.step()
#     -> connectome-derived neural obstacle steering
#
#   visual steering + neural obstacle steering
#     -> ENGINEERED arbitration / hysteresis / blending
#     -> motors
#
# IMPORTANT SCIENTIFIC BOUNDARIES
# --------------------------------
# Biological / MaleCNS-derived:
#   - LC10a / AOTU019 / DNa02 identities and anatomical weights
#   - AOTU019 GABA annotation
#   - obstacle FlyNavBrain network and DNa decoder
#
# Engineering:
#   - camera pixel/color detection
#   - camera image side -> LC10a side interface
#   - per-side visual normalization
#   - proximity normalization
#   - arbitration, hysteresis, blending
#   - apparent red area as target-distance proxy
#
# No arithmetic obstacle left-right steering bypass is used.
#
# Four controlled cases:
#   1. visual_left_clear
#   2. conflict_left_target_left_obstacle
#   3. conflict_right_target_right_obstacle
#   4. center_target_offset_obstacle
#
# PASS requires reaching the visual target. Obstacle cases must
# also actually activate the neural obstacle branch.
#
# Step 57B deliberately avoids starting with the obstacle directly
# on the camera-to-target line. Step 57 showed that such geometry
# can fully occlude the target while the obstacle is still outside
# proximity range, producing a test-fixture deadlock before fusion
# has a chance to operate.
# ============================================================


TIME_STEP = 32

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flynav_brain import FlyNavBrain


# ============================================================
# Scene / visual target
# ============================================================

SAFE_START_X = -0.25
SAFE_START_Y = 0.0
SAFE_START_HEADING_RAD = 0.0

TARGET_DISTANCE_M = 0.50
TARGET_RADIUS_M = 0.030
TARGET_Z_OFFSET_M = 0.040

SETTLE_STEPS = 12
TRIAL_TIMEOUT_S = 45.0

CENTER_TOLERANCE = 0.12
CENTER_HOLD_FRAMES = 8

# Step 59 engineering deadlock release:
# When the target is already at the apparent-distance threshold,
# allow a wider bearing tolerance ONLY if obstacle blending is weak.
# Step 58A showed deterministic equilibria with red ~= 0.155-0.159,
# centroid ~= 0.27-0.50, and blend ~= 0.15-0.20.
#
# This changes only the arrival/arbitration layer. It does not alter
# the connectome-derived visual or obstacle pathways.
CLOSE_ARRIVAL_CENTROID_TOLERANCE = 0.55
CLOSE_ARRIVAL_MAX_BLEND = 0.25

MIN_RED = 120
MIN_RED_ADVANTAGE = 45
VISIBLE_PIXEL_FRACTION = 0.002

STOP_RED_FRACTION = 0.155
SLOW_RED_FRACTION = 0.090
CONFIDENCE_RED_FRACTION = 0.10

VISUAL_SMOOTH_ALPHA = 0.30
VISUAL_FORWARD_SPEED_FRACTION = 0.12
VISUAL_MIN_FORWARD_SPEED_FRACTION = 0.035
VISUAL_TURN_GAIN = 0.11


# ============================================================
# Frozen neural obstacle calibration
# ============================================================

SENSOR_FLOOR = 80.0
SENSOR_SATURATION = 650.0

OBSTACLE_BASE_SPEED_FRACTION = 0.28
OBSTACLE_STEERING_GAIN = 0.55
OBSTACLE_SMOOTH_ALPHA = 0.25

OBSTACLE_SPEED_REDUCTION = 0.85
OBSTACLE_MIN_SPEED_SCALE = 0.12

OBSTACLE_ENTER_RAW = 250.0
OBSTACLE_EXIT_RAW = 180.0
OBSTACLE_FULL_CONTROL_RAW = 650.0

# When the target is visually occluded but the obstacle branch
# remains active, allow only a slow avoidance crawl.
LOST_TARGET_AVOID_FORWARD_FRACTION = 0.07

# After neural obstacle avoidance turns the rover far enough that
# the visual target exits the 48-degree camera FOV, keep a bounded
# memory of the LAST SEEN target side and rotate in place toward it.
#
# This is an ENGINEERED recovery behavior. We are not claiming a
# specific MaleCNS memory/search circuit here.
REACQUIRE_TURN_SPEED_FRACTION = 0.075
REACQUIRE_MAX_TIME_S = 18.0
REACQUIRE_MIN_ABS_CENTROID = 0.03

# Step 60 engineering visual-memory policy:
#
# While the target is visible, convert camera centroid + current
# rover heading into an estimated WORLD bearing to the target.
# If vision is later lost after obstacle avoidance, rotate toward
# that remembered world bearing instead of relying only on a timed
# left/right search.
#
# This is an ENGINEERED sensor-memory layer. It is not claimed as
# a MaleCNS memory circuit.
#
# If the remembered bearing is reached but the target is still not
# visible, continue a bounded rotational scan. Proximity avoidance
# retains priority if an obstacle reappears.
REACQUIRE_HEADING_TOLERANCE_DEG = 4.0
REACQUIRE_MEMORY_MIN_RED_FRACTION = 0.002

MOTOR_SMOOTH_ALPHA = 0.25

PRINT_EVERY_STEPS = 10


# ============================================================
# Controlled multisensory cases
# ============================================================

CASES = [
    {
        "name": "visual_left_clear",
        "target_angle_deg": +10.0,
        "obstacle_xy": (0.00, +0.35),
        "expect_obstacle": False,
    },
    {
        "name": "conflict_left_target_left_obstacle",
        "target_angle_deg": +10.0,
        "obstacle_xy": (0.00, +0.09),
        "expect_obstacle": True,
    },
    {
        "name": "conflict_right_target_right_obstacle",
        "target_angle_deg": -10.0,
        "obstacle_xy": (0.00, -0.09),
        "expect_obstacle": True,
    },
    {
        "name": "center_target_offset_obstacle",
        "target_angle_deg": 0.0,
        "obstacle_xy": (0.00, +0.055),
        "expect_obstacle": True,
    },
]


# ============================================================
# Helpers
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def get_heading(self_node):
    orientation = self_node.getOrientation()

    return math.atan2(
        orientation[3],
        orientation[0],
    )


def reset_rover(self_node):
    translation = self_node.getField(
        "translation"
    )

    if translation is None:
        raise RuntimeError(
            "E-puck has no translation field."
        )

    current = translation.getSFVec3f()
    z = float(current[2])

    translation.setSFVec3f(
        [
            SAFE_START_X,
            SAFE_START_Y,
            z,
        ]
    )

    rotation = self_node.getField(
        "rotation"
    )

    if rotation is not None:
        rotation.setSFRotation(
            [
                0.0,
                0.0,
                1.0,
                SAFE_START_HEADING_RAD,
            ]
        )

    self_node.resetPhysics()


def find_top_level_node_by_name(
    robot,
    wanted_name,
):
    root = robot.getRoot()
    children = root.getField(
        "children"
    )

    if children is None:
        return None

    for i in range(
        children.getCount()
    ):
        node = children.getMFNode(i)

        if node is None:
            continue

        name_field = node.getField(
            "name"
        )

        if name_field is None:
            continue

        try:
            name = (
                name_field.getSFString()
            )
        except Exception:
            continue

        if name == wanted_name:
            return node

    return None


def move_node_xy(
    node,
    x,
    y,
):
    field = node.getField(
        "translation"
    )

    if field is None:
        raise RuntimeError(
            "Node has no translation field."
        )

    current = field.getSFVec3f()
    z = float(current[2])

    field.setSFVec3f(
        [
            float(x),
            float(y),
            z,
        ]
    )

    try:
        node.resetPhysics()
    except Exception:
        pass


def get_or_create_target(robot):
    target = robot.getFromDef(
        "VISUAL_TARGET"
    )

    if target is not None:
        return target, False

    root = robot.getRoot()
    children = root.getField(
        "children"
    )

    node_string = f"""
    DEF VISUAL_TARGET Transform {{
      translation 0 0 {TARGET_Z_OFFSET_M:.4f}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor 1 0 0
            roughness 1
            metalness 0
          }}
          geometry Sphere {{
            radius {TARGET_RADIUS_M:.4f}
          }}
        }}
      ]
    }}
    """

    children.importMFNodeFromString(
        -1,
        node_string,
    )

    target = robot.getFromDef(
        "VISUAL_TARGET"
    )

    if target is None:
        raise RuntimeError(
            "Could not create DEF VISUAL_TARGET."
        )

    return target, True


def place_target_relative(
    target_node,
    self_node,
    relative_angle_deg,
):
    position = self_node.getPosition()

    x = float(position[0])
    y = float(position[1])
    z = float(position[2])

    heading = get_heading(
        self_node
    )

    world_angle = (
        heading
        + math.radians(
            relative_angle_deg
        )
    )

    tx = (
        x
        + TARGET_DISTANCE_M
        * math.cos(world_angle)
    )

    ty = (
        y
        + TARGET_DISTANCE_M
        * math.sin(world_angle)
    )

    tz = (
        z
        + TARGET_Z_OFFSET_M
    )

    field = target_node.getField(
        "translation"
    )

    if field is None:
        raise RuntimeError(
            "VISUAL_TARGET has no translation field."
        )

    field.setSFVec3f(
        [tx, ty, tz]
    )

    return tx, ty, tz


def decode_red_target(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()

    if image is None:
        return {
            "visible": False,
            "red_fraction": 0.0,
            "centroid_x_norm": float(
                "nan"
            ),
        }

    red_pixels = 0
    x_sum = 0.0

    for y in range(height):
        for x in range(width):
            r = Camera.imageGetRed(
                image,
                width,
                x,
                y,
            )

            g = Camera.imageGetGreen(
                image,
                width,
                x,
                y,
            )

            b = Camera.imageGetBlue(
                image,
                width,
                x,
                y,
            )

            is_red = (
                r >= MIN_RED
                and (r - g)
                >= MIN_RED_ADVANTAGE
                and (r - b)
                >= MIN_RED_ADVANTAGE
            )

            if not is_red:
                continue

            red_pixels += 1
            x_sum += x

    total_pixels = width * height

    red_fraction = (
        red_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    visible = (
        red_fraction
        >= VISIBLE_PIXEL_FRACTION
    )

    if red_pixels <= 0:
        centroid = float(
            "nan"
        )
    else:
        centroid_px = (
            x_sum / red_pixels
        )

        centroid = (
            2.0
            * centroid_px
            / (width - 1)
            - 1.0
        )

    return {
        "visible": visible,
        "red_fraction": red_fraction,
        "centroid_x_norm": centroid,
    }


def load_visual_bridge():
    paths_path = (
        PROJECT_ROOT
        / "connectome"
        / "output"
        / "53_lc10a_aotu019_dna_paths.csv"
    )

    aotu_path = (
        PROJECT_ROOT
        / "connectome"
        / "output"
        / "53_aotu019_inventory.csv"
    )

    paths = pd.read_csv(
        paths_path
    )

    aotu = pd.read_csv(
        aotu_path
    )

    nts = set(
        aotu["predictedNt"]
        .dropna()
        .astype(str)
        .str.lower()
    )

    if "gaba" not in nts:
        raise RuntimeError(
            "AOTU019 GABA annotation missing."
        )

    left_path = paths[
        (paths["lc10a_side"] == "L")
        & (paths["aotu_side"] == "L")
        & (paths["dna_type"] == "DNa02")
        & (paths["dna_side"] == "R")
    ]

    right_path = paths[
        (paths["lc10a_side"] == "R")
        & (paths["aotu_side"] == "R")
        & (paths["dna_type"] == "DNa02")
        & (paths["dna_side"] == "L")
    ]

    if left_path.empty:
        raise RuntimeError(
            "Missing visual left pathway."
        )

    if right_path.empty:
        raise RuntimeError(
            "Missing visual right pathway."
        )

    left_score = float(
        left_path[
            "bottleneck_weight"
        ].sum()
    )

    right_score = float(
        right_path[
            "bottleneck_weight"
        ].sum()
    )

    return {
        "left_score": left_score,
        "right_score": right_score,
        "left_gain":
            1.0 / left_score,
        "right_gain":
            1.0 / right_score,
    }


def visual_neural_steering(
    visual,
    bridge,
):
    if (
        not visual["visible"]
        or math.isnan(
            visual[
                "centroid_x_norm"
            ]
        )
    ):
        return {
            "confidence": 0.0,
            "dna02_L_drive": 1.0,
            "dna02_R_drive": 1.0,
            "steering": 0.0,
        }

    centroid = clamp(
        float(
            visual[
                "centroid_x_norm"
            ]
        ),
        -1.0,
        +1.0,
    )

    confidence = clamp(
        visual[
            "red_fraction"
        ]
        / CONFIDENCE_RED_FRACTION,
        0.0,
        1.0,
    )

    # ENGINEERED camera-field interface.
    visual_L = (
        max(
            0.0,
            -centroid,
        )
        * confidence
    )

    visual_R = (
        max(
            0.0,
            +centroid,
        )
        * confidence
    )

    # Connectome-derived side-specific inhibitory routes,
    # normalized only to remove absolute anatomical scale.
    inhibit_R = clamp(
        visual_L
        * bridge["left_score"]
        * bridge["left_gain"],
        0.0,
        1.0,
    )

    inhibit_L = clamp(
        visual_R
        * bridge["right_score"]
        * bridge["right_gain"],
        0.0,
        1.0,
    )

    dna02_L = (
        1.0 - inhibit_L
    )

    dna02_R = (
        1.0 - inhibit_R
    )

    # Frozen FlyNav sign:
    # + = DNa02_R dominance = right turn
    # - = DNa02_L dominance = left turn
    steering = (
        dna02_R
        - dna02_L
    )

    return {
        "confidence": confidence,
        "dna02_L_drive": dna02_L,
        "dna02_R_drive": dna02_R,
        "steering": steering,
    }


def visual_forward_scale(
    red_fraction,
    centroid,
):
    if (
        red_fraction
        >= STOP_RED_FRACTION
    ):
        size_scale = 0.0

    elif (
        red_fraction
        <= SLOW_RED_FRACTION
    ):
        size_scale = 1.0

    else:
        size_scale = (
            STOP_RED_FRACTION
            - red_fraction
        ) / (
            STOP_RED_FRACTION
            - SLOW_RED_FRACTION
        )

    alignment_scale = clamp(
        1.0
        - 0.75
        * abs(centroid),
        0.25,
        1.0,
    )

    scale = (
        size_scale
        * alignment_scale
    )

    if (
        size_scale > 0.0
        and scale > 0.0
    ):
        scale = max(
            scale,
            VISUAL_MIN_FORWARD_SPEED_FRACTION
            / VISUAL_FORWARD_SPEED_FRACTION,
        )

    return clamp(
        scale,
        0.0,
        1.0,
    )


def wrap_angle_rad(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def camera_centroid_to_bearing_rad(
    centroid_x_norm,
    camera_fov_rad,
):
    """
    ENGINEERED pinhole-camera bearing estimate.

    Webots image convention validated in Step 49:
      camera-left  -> negative centroid
      camera-right -> positive centroid

    FlyNav world-heading convention:
      positive heading error -> LEFT / CCW

    Therefore image sign is inverted when converting to relative
    world bearing.
    """
    centroid = clamp(
        float(centroid_x_norm),
        -1.0,
        +1.0,
    )

    return -math.atan(
        centroid
        * math.tan(
            camera_fov_rad / 2.0
        )
    )


def remembered_target_world_bearing(
    self_node,
    camera,
    centroid_x_norm,
):
    relative_bearing = (
        camera_centroid_to_bearing_rad(
            centroid_x_norm,
            camera.getFov(),
        )
    )

    return wrap_angle_rad(
        get_heading(self_node)
        + relative_bearing
    )


def memory_reacquire_turn(
    self_node,
    remembered_world_bearing,
    last_seen_centroid,
):
    """
    Return (turn_sign, heading_error_rad, aligned).

    Motor convention:
      positive final_turn -> RIGHT
      negative final_turn -> LEFT

    Heading convention:
      positive heading error -> target lies LEFT / CCW

    Hence motor turn sign is the negative of heading-error sign.
    """
    current_heading = (
        get_heading(self_node)
    )

    heading_error = wrap_angle_rad(
        remembered_world_bearing
        - current_heading
    )

    tolerance = math.radians(
        REACQUIRE_HEADING_TOLERANCE_DEG
    )

    if abs(heading_error) > tolerance:
        turn_sign = (
            -1.0
            if heading_error > 0.0
            else +1.0
        )

        return (
            turn_sign,
            heading_error,
            False,
        )

    # At the remembered bearing but still no target:
    # continue a bounded local/full scan rather than stopping.
    if abs(last_seen_centroid) >= REACQUIRE_MIN_ABS_CENTROID:
        scan_sign = (
            +1.0
            if last_seen_centroid > 0.0
            else -1.0
        )
    else:
        scan_sign = +1.0

    return (
        scan_sign,
        heading_error,
        True,
    )


def encode_proximity(value):
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


def obstacle_blend_weight(
    max_raw,
):
    return clamp(
        (
            max_raw
            - OBSTACLE_EXIT_RAW
        )
        / (
            OBSTACLE_FULL_CONTROL_RAW
            - OBSTACLE_EXIT_RAW
        ),
        0.0,
        1.0,
    )


# ============================================================
# Main
# ============================================================

def main():
    robot = Supervisor()

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 57C requires supervisor TRUE."
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
        float(
            left_motor.getMaxVelocity()
        ),
        float(
            right_motor.getMaxVelocity()
        ),
    )

    camera = robot.getDevice(
        "camera"
    )

    camera.enable(
        TIME_STEP
    )

    proximity = []

    for i in range(8):
        sensor = robot.getDevice(
            f"ps{i}"
        )

        sensor.enable(
            TIME_STEP
        )

        proximity.append(
            sensor
        )

    bridge = load_visual_bridge()

    target_node, target_created = (
        get_or_create_target(
            robot
        )
    )

    obstacle_node = (
        find_top_level_node_by_name(
            robot,
            "test obstacle",
        )
    )

    if obstacle_node is None:
        raise RuntimeError(
            "Could not find top-level Solid named "
            "'test obstacle'."
        )

    obstacle_translation = (
        obstacle_node.getField(
            "translation"
        )
    )

    obstacle_original = list(
        obstacle_translation.getSFVec3f()
    )

    output_path = (
        PROJECT_ROOT
        / "connectome"
        / "output"
        / "60_multisensory_neural_navigation_v3.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "case",
        "step",
        "time_s",
        "mode",
        "target_visible",
        "red_fraction",
        "centroid_x_norm",
        "last_seen_centroid",
        "remembered_target_world_bearing_deg",
        "reacquire_heading_error_deg",
        "lost_streak",
        "reacquire_steps",
        "visual_raw",
        "visual_smoothed",
        "front_left_raw",
        "front_right_raw",
        "obstacle_left_input",
        "obstacle_right_input",
        "obstacle_raw",
        "obstacle_smoothed",
        "dna_left_spikes",
        "dna_right_spikes",
        "dna_left_membrane",
        "dna_right_membrane",
        "max_proximity_raw",
        "obstacle_blend",
        "visual_forward",
        "obstacle_forward",
        "final_forward",
        "final_turn",
        "left_wheel",
        "right_wheel",
        "arrival_hold",
    ]

    summaries = []

    print()
    print("=" * 122)
    print(
        "STEP 60 - MULTISENSORY NEURAL NAVIGATION V3 - HEADING-ASSISTED VISUAL MEMORY"
    )
    print("=" * 122)
    print()
    print(
        "Vision: camera -> LC10a -> AOTU019 -| DNa02"
    )
    print(
        "Obstacle: proximity -> FlyNavBrain.step() -> DNa steering"
    )
    print(
        "Fusion: engineered hysteresis / arbitration / blending"
    )
    print(
        "Arithmetic obstacle steering bypass: NONE"
    )
    print(
        "Geometry fix: rover starts at x=-0.25, target is 0.50 m away, "
        "and obstacle cases are laterally offset so vision is available "
        "before proximity activation."
    )
    print(
        "Recovery: camera centroid + rover heading are stored as an "
        "engineered world-bearing memory while the target is visible."
    )
    print(
        "After obstacle clearance the rover turns toward that remembered "
        "bearing; if aligned but still blind, it continues a bounded scan."
    )
    print(
        f"Reacquisition limit: {REACQUIRE_MAX_TIME_S:.1f}s | "
        f"turn fraction={REACQUIRE_TURN_SPEED_FRACTION:.3f} | "
        f"heading tolerance={REACQUIRE_HEADING_TOLERANCE_DEG:.1f} deg"
    )

    print(
        "Close-arrival release: "
        f"red>={STOP_RED_FRACTION:.3f}, "
        f"|centroid|<={CLOSE_ARRIVAL_CENTROID_TOLERANCE:.2f}, "
        f"blend<={CLOSE_ARRIVAL_MAX_BLEND:.2f}"
    )
    print()

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
            )

            writer.writeheader()

            for case in CASES:
                left_motor.setVelocity(
                    0.0
                )
                right_motor.setVelocity(
                    0.0
                )

                reset_rover(
                    self_node
                )

                move_node_xy(
                    obstacle_node,
                    *case[
                        "obstacle_xy"
                    ],
                )

                for _ in range(
                    SETTLE_STEPS
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                tx, ty, tz = (
                    place_target_relative(
                        target_node,
                        self_node,
                        case[
                            "target_angle_deg"
                        ],
                    )
                )

                for _ in range(
                    SETTLE_STEPS
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                initial_visual = (
                    decode_red_target(
                        camera
                    )
                )

                initial_centroid = (
                    initial_visual[
                        "centroid_x_norm"
                    ]
                )

                initial_centroid_text = (
                    "NA"
                    if math.isnan(
                        initial_centroid
                    )
                    else
                    f"{initial_centroid:+.3f}"
                )

                print("-" * 122)
                print(
                    f"CASE {case['name']} | "
                    f"target_angle="
                    f"{case['target_angle_deg']:+.1f} deg | "
                    f"target=({tx:+.3f},{ty:+.3f}) | "
                    f"obstacle=("
                    f"{case['obstacle_xy'][0]:+.3f},"
                    f"{case['obstacle_xy'][1]:+.3f}) | "
                    f"vis={initial_visual['visible']} "
                    f"red={initial_visual['red_fraction']:.3f} "
                    f"cent={initial_centroid_text}"
                )

                # A multisensory fusion case is invalid if the target is
                # already fully occluded before the rover has moved and
                # before proximity is active. Fail the fixture immediately
                # rather than spending the whole timeout in LOST_STOP.
                initial_sensor_values = [
                    float(sensor.getValue())
                    for sensor in proximity
                ]

                initial_front_left = max(
                    initial_sensor_values[6],
                    initial_sensor_values[7],
                )

                initial_front_right = max(
                    initial_sensor_values[0],
                    initial_sensor_values[1],
                )

                initial_max_prox = max(
                    initial_front_left,
                    initial_front_right,
                )

                if not initial_visual["visible"]:
                    print(
                        "  SCENE INVALID: target is fully occluded at "
                        "trial start. This is a geometry failure, not a "
                        "neural-fusion failure."
                    )

                    summaries.append(
                        {
                            "case": case["name"],
                            "passed": False,
                            "target_reached": False,
                            "obstacle_requirement": False,
                            "recovery_requirement": False,
                            "obstacle_activated": False,
                            "elapsed_s": 0.0,
                            "final_red": 0.0,
                            "final_centroid": float("nan"),
                            "lost_frames": 0,
                            "reacquired_count": 0,
                            "max_prox": initial_max_prox,
                            "max_blend": 0.0,
                            "max_neural": 0.0,
                            "dna_left_max": 0,
                            "dna_right_max": 0,
                        }
                    )

                    continue

                # Fresh obstacle neural state per controlled case.
                brain = FlyNavBrain(
                    PROJECT_ROOT
                )

                visual_smoothed = 0.0
                obstacle_smoothed = 0.0
                obstacle_latched = False

                previous_left = 0.0
                previous_right = 0.0

                arrival_hold = 0
                lost_frames = 0
                reacquired_count = 0
                lost_streak = 0
                reacquire_steps = 0

                initial_c = float(
                    initial_visual[
                        "centroid_x_norm"
                    ]
                )

                last_seen_centroid = (
                    initial_c
                    if not math.isnan(initial_c)
                    else 0.0
                )

                remembered_world_bearing = (
                    remembered_target_world_bearing(
                        self_node,
                        camera,
                        initial_c,
                    )
                    if (
                        initial_visual["visible"]
                        and not math.isnan(initial_c)
                    )
                    else None
                )

                reacquire_heading_error = 0.0

                was_visible = bool(
                    initial_visual[
                        "visible"
                    ]
                )

                max_prox_seen = 0.0
                max_blend_seen = 0.0
                max_neural_seen = 0.0
                max_dna_left = 0
                max_dna_right = 0

                obstacle_activated = False
                passed = False

                final_red = 0.0
                final_centroid = float(
                    "nan"
                )

                max_steps = int(
                    TRIAL_TIMEOUT_S
                    * 1000.0
                    / TIME_STEP
                )

                step_count = 0

                while (
                    step_count
                    < max_steps
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                    step_count += 1

                    visual = (
                        decode_red_target(
                            camera
                        )
                    )

                    visible = bool(
                        visual["visible"]
                    )

                    if visible:
                        if not was_visible:
                            reacquired_count += 1

                        lost_streak = 0
                        reacquire_steps = 0

                        final_red = float(
                            visual[
                                "red_fraction"
                            ]
                        )

                        final_centroid = float(
                            visual[
                                "centroid_x_norm"
                            ]
                        )

                        # Keep the latest visible bearing. Near center,
                        # preserve the previous non-trivial side memory
                        # instead of replacing it with pixel noise.
                        if (
                            abs(final_centroid)
                            >= REACQUIRE_MIN_ABS_CENTROID
                        ):
                            last_seen_centroid = (
                                final_centroid
                            )

                        if (
                            visual["red_fraction"]
                            >= REACQUIRE_MEMORY_MIN_RED_FRACTION
                        ):
                            remembered_world_bearing = (
                                remembered_target_world_bearing(
                                    self_node,
                                    camera,
                                    final_centroid,
                                )
                            )

                        reacquire_heading_error = 0.0

                    else:
                        lost_frames += 1
                        lost_streak += 1

                    was_visible = visible

                    visual_output = (
                        visual_neural_steering(
                            visual,
                            bridge,
                        )
                    )

                    visual_raw = float(
                        visual_output[
                            "steering"
                        ]
                    )

                    visual_smoothed = (
                        (
                            1.0
                            - VISUAL_SMOOTH_ALPHA
                        )
                        * visual_smoothed
                        +
                        VISUAL_SMOOTH_ALPHA
                        * visual_raw
                    )

                    sensor_values = [
                        float(
                            sensor.getValue()
                        )
                        for sensor in proximity
                    ]

                    front_left = max(
                        sensor_values[6],
                        sensor_values[7],
                    )

                    front_right = max(
                        sensor_values[0],
                        sensor_values[1],
                    )

                    obstacle_left = (
                        encode_proximity(
                            front_left
                        )
                    )

                    obstacle_right = (
                        encode_proximity(
                            front_right
                        )
                    )

                    neural = brain.step(
                        obstacle_left=
                            obstacle_left,
                        obstacle_right=
                            obstacle_right,
                        duration_ms=TIME_STEP,
                    )

                    obstacle_raw = float(
                        neural["steering"]
                    )

                    obstacle_smoothed = (
                        (
                            1.0
                            - OBSTACLE_SMOOTH_ALPHA
                        )
                        * obstacle_smoothed
                        +
                        OBSTACLE_SMOOTH_ALPHA
                        * obstacle_raw
                    )

                    dna_left = int(
                        neural[
                            "left_spikes"
                        ]
                    )

                    dna_right = int(
                        neural[
                            "right_spikes"
                        ]
                    )

                    dna_left_v = float(
                        neural[
                            "left_membrane"
                        ]
                    )

                    dna_right_v = float(
                        neural[
                            "right_membrane"
                        ]
                    )

                    max_raw = max(
                        front_left,
                        front_right,
                    )

                    max_prox_seen = max(
                        max_prox_seen,
                        max_raw,
                    )

                    max_neural_seen = max(
                        max_neural_seen,
                        abs(
                            obstacle_raw
                        ),
                    )

                    max_dna_left = max(
                        max_dna_left,
                        dna_left,
                    )

                    max_dna_right = max(
                        max_dna_right,
                        dna_right,
                    )

                    if not obstacle_latched:
                        if (
                            max_raw
                            >= OBSTACLE_ENTER_RAW
                        ):
                            obstacle_latched = True
                            obstacle_activated = True
                    else:
                        if (
                            max_raw
                            <= OBSTACLE_EXIT_RAW
                        ):
                            obstacle_latched = False

                    if obstacle_latched:
                        blend = max(
                            0.15,
                            obstacle_blend_weight(
                                max_raw
                            ),
                        )
                    else:
                        blend = 0.0

                    max_blend_seen = max(
                        max_blend_seen,
                        blend,
                    )

                    obstacle_speed_scale = clamp(
                        (
                            1.0
                            - OBSTACLE_SPEED_REDUCTION
                            * (
                                obstacle_left
                                + obstacle_right
                            )
                            / 2.0
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
                        obstacle_smoothed
                        * OBSTACLE_STEERING_GAIN
                        * max_speed
                    )

                    if visible:
                        centroid = float(
                            visual[
                                "centroid_x_norm"
                            ]
                        )

                        visual_scale = (
                            visual_forward_scale(
                                visual[
                                    "red_fraction"
                                ],
                                centroid,
                            )
                        )

                        visual_forward = (
                            VISUAL_FORWARD_SPEED_FRACTION
                            * max_speed
                            * visual_scale
                        )

                        visual_turn = (
                            visual_smoothed
                            * VISUAL_TURN_GAIN
                            * max_speed
                        )

                        strict_arrived = (
                            visual[
                                "red_fraction"
                            ]
                            >= STOP_RED_FRACTION
                            and abs(
                                centroid
                            )
                            <= CENTER_TOLERANCE
                        )

                        close_safe_arrived = (
                            visual[
                                "red_fraction"
                            ]
                            >= STOP_RED_FRACTION
                            and abs(
                                centroid
                            )
                            <= CLOSE_ARRIVAL_CENTROID_TOLERANCE
                            and blend
                            <= CLOSE_ARRIVAL_MAX_BLEND
                        )

                        arrived = (
                            strict_arrived
                            or close_safe_arrived
                        )

                    else:
                        visual_forward = 0.0
                        visual_turn = 0.0
                        arrived = False

                    if arrived:
                        arrival_hold += 1

                        final_forward = 0.0
                        final_turn = 0.0

                        mode = "ARRIVED"

                    else:
                        arrival_hold = 0

                        if visible:
                            final_forward = min(
                                visual_forward,
                                obstacle_forward,
                            )

                            final_turn = (
                                (1.0 - blend)
                                * visual_turn
                                +
                                blend
                                * obstacle_turn
                            )

                            if blend >= 0.75:
                                mode = "AVOID"
                            elif blend > 0.0:
                                mode = "BLEND"
                            else:
                                mode = "VISION"

                        elif obstacle_latched:
                            # Target temporarily occluded:
                            # continue only with the neural
                            # obstacle branch until vision
                            # returns or the obstacle clears.
                            final_forward = min(
                                LOST_TARGET_AVOID_FORWARD_FRACTION
                                * max_speed,
                                obstacle_forward,
                            )

                            final_turn = (
                                obstacle_turn
                            )

                            mode = (
                                "LOST_AVOID"
                            )

                        else:
                            # Obstacle has cleared but vision may still be
                            # outside the camera FOV because avoidance just
                            # rotated the rover away from the target.
                            #
                            # Use a bounded engineering memory of the LAST
                            # SEEN target side to rotate in place until the
                            # camera sees the target again.
                            max_reacquire_steps = int(
                                REACQUIRE_MAX_TIME_S
                                * 1000.0
                                / TIME_STEP
                            )

                            if (
                                remembered_world_bearing
                                is not None
                                and reacquire_steps
                                < max_reacquire_steps
                            ):
                                (
                                    direction,
                                    reacquire_heading_error,
                                    memory_aligned,
                                ) = memory_reacquire_turn(
                                    self_node,
                                    remembered_world_bearing,
                                    last_seen_centroid,
                                )

                                reacquire_steps += 1

                                final_forward = 0.0
                                final_turn = (
                                    direction
                                    * REACQUIRE_TURN_SPEED_FRACTION
                                    * max_speed
                                )

                                if memory_aligned:
                                    mode = "REACQUIRE_SCAN"
                                else:
                                    mode = "REACQUIRE_MEMORY"

                            else:
                                reacquire_heading_error = 0.0
                                final_forward = 0.0
                                final_turn = 0.0

                                mode = "LOST_STOP"

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

                    previous_left = (
                        left_velocity
                    )

                    previous_right = (
                        right_velocity
                    )

                    writer.writerow(
                        {
                            "case":
                                case["name"],
                            "step":
                                step_count,
                            "time_s":
                                f"{step_count * TIME_STEP / 1000.0:.4f}",
                            "mode":
                                mode,
                            "target_visible":
                                int(visible),
                            "red_fraction":
                                f"{visual['red_fraction']:.6f}",
                            "centroid_x_norm":
                                (
                                    ""
                                    if math.isnan(
                                        visual[
                                            "centroid_x_norm"
                                        ]
                                    )
                                    else
                                    f"{visual['centroid_x_norm']:.6f}"
                                ),
                            "last_seen_centroid":
                                f"{last_seen_centroid:.6f}",
                            "remembered_target_world_bearing_deg":
                                (
                                    ""
                                    if remembered_world_bearing is None
                                    else
                                    f"{math.degrees(remembered_world_bearing):.6f}"
                                ),
                            "reacquire_heading_error_deg":
                                f"{math.degrees(reacquire_heading_error):.6f}",
                            "lost_streak":
                                lost_streak,
                            "reacquire_steps":
                                reacquire_steps,
                            "visual_raw":
                                f"{visual_raw:.6f}",
                            "visual_smoothed":
                                f"{visual_smoothed:.6f}",
                            "front_left_raw":
                                f"{front_left:.6f}",
                            "front_right_raw":
                                f"{front_right:.6f}",
                            "obstacle_left_input":
                                f"{obstacle_left:.6f}",
                            "obstacle_right_input":
                                f"{obstacle_right:.6f}",
                            "obstacle_raw":
                                f"{obstacle_raw:.6f}",
                            "obstacle_smoothed":
                                f"{obstacle_smoothed:.6f}",
                            "dna_left_spikes":
                                dna_left,
                            "dna_right_spikes":
                                dna_right,
                            "dna_left_membrane":
                                f"{dna_left_v:.6f}",
                            "dna_right_membrane":
                                f"{dna_right_v:.6f}",
                            "max_proximity_raw":
                                f"{max_raw:.6f}",
                            "obstacle_blend":
                                f"{blend:.6f}",
                            "visual_forward":
                                f"{visual_forward:.6f}",
                            "obstacle_forward":
                                f"{obstacle_forward:.6f}",
                            "final_forward":
                                f"{final_forward:.6f}",
                            "final_turn":
                                f"{final_turn:.6f}",
                            "left_wheel":
                                f"{left_velocity:.6f}",
                            "right_wheel":
                                f"{right_velocity:.6f}",
                            "arrival_hold":
                                arrival_hold,
                        }
                    )

                    if (
                        step_count
                        % PRINT_EVERY_STEPS
                        == 0
                    ):
                        centroid = visual[
                            "centroid_x_norm"
                        ]

                        centroid_text = (
                            "NA"
                            if math.isnan(
                                centroid
                            )
                            else
                            f"{centroid:+.3f}"
                        )

                        print(
                            f"  t="
                            f"{step_count * TIME_STEP / 1000.0:5.2f}s "
                            f"mode={mode:<10} "
                            f"vis={visible} "
                            f"red={visual['red_fraction']:.3f} "
                            f"cent={centroid_text:>7} "
                            f"vRAW={visual_raw:+.3f} "
                            f"FL={front_left:7.1f} "
                            f"FR={front_right:7.1f} "
                            f"nRAW={obstacle_raw:+.3f} "
                            f"DNa=({dna_left},{dna_right}) "
                            f"blend={blend:.2f} "
                            f"last={last_seen_centroid:+.3f} "
                            f"mem="
                            f"{'NA' if remembered_world_bearing is None else f'{math.degrees(remembered_world_bearing):+.1f}'} "
                            f"herr={math.degrees(reacquire_heading_error):+.1f} "
                            f"rq={reacquire_steps:3d} "
                            f"wheels=("
                            f"{left_velocity:+.2f},"
                            f"{right_velocity:+.2f})"
                        )

                    if (
                        arrival_hold
                        >= CENTER_HOLD_FRAMES
                    ):
                        passed = True
                        break

                left_motor.setVelocity(
                    0.0
                )

                right_motor.setVelocity(
                    0.0
                )

                elapsed = (
                    step_count
                    * TIME_STEP
                    / 1000.0
                )

                obstacle_requirement = (
                    obstacle_activated
                    if case[
                        "expect_obstacle"
                    ]
                    else not obstacle_activated
                )

                recovery_requirement = (
                    True
                    if lost_frames == 0
                    else reacquired_count > 0
                )

                overall_pass = (
                    passed
                    and obstacle_requirement
                    and recovery_requirement
                )

                summaries.append(
                    {
                        "case":
                            case["name"],
                        "passed":
                            overall_pass,
                        "target_reached":
                            passed,
                        "obstacle_requirement":
                            obstacle_requirement,
                        "recovery_requirement":
                            recovery_requirement,
                        "obstacle_activated":
                            obstacle_activated,
                        "elapsed_s":
                            elapsed,
                        "final_red":
                            final_red,
                        "final_centroid":
                            final_centroid,
                        "lost_frames":
                            lost_frames,
                        "reacquired_count":
                            reacquired_count,
                        "max_prox":
                            max_prox_seen,
                        "max_blend":
                            max_blend_seen,
                        "max_neural":
                            max_neural_seen,
                        "dna_left_max":
                            max_dna_left,
                        "dna_right_max":
                            max_dna_right,
                    }
                )

                centroid_text = (
                    "NA"
                    if math.isnan(
                        final_centroid
                    )
                    else
                    f"{final_centroid:+.3f}"
                )

                print(
                    f"  RESULT "
                    f"{'PASS' if overall_pass else 'FAIL'} | "
                    f"target={passed} | "
                    f"obs_active={obstacle_activated} | "
                    f"recovery={recovery_requirement} | "
                    f"time={elapsed:.2f}s | "
                    f"red={final_red:.3f} | "
                    f"cent={centroid_text} | "
                    f"prox={max_prox_seen:.1f} | "
                    f"blend={max_blend_seen:.2f} | "
                    f"nmax={max_neural_seen:.3f} | "
                    f"DNaMax=("
                    f"{max_dna_left},"
                    f"{max_dna_right}) | "
                    f"lost={lost_frames} | "
                    f"reacq={reacquired_count}"
                )

                handle.flush()

    finally:
        left_motor.setVelocity(
            0.0
        )

        right_motor.setVelocity(
            0.0
        )

        obstacle_translation.setSFVec3f(
            obstacle_original
        )

        try:
            obstacle_node.resetPhysics()
        except Exception:
            pass

        if target_created:
            try:
                target_node.remove()
            except Exception:
                pass

    passed_count = sum(
        1
        for row in summaries
        if row["passed"]
    )

    print()
    print("=" * 122)
    print(
        "STEP 60 FINAL SUMMARY"
    )
    print("=" * 122)

    for row in summaries:
        centroid = row[
            "final_centroid"
        ]

        centroid_text = (
            "NA"
            if math.isnan(
                centroid
            )
            else
            f"{centroid:+.3f}"
        )

        print(
            f"{row['case']:<40} "
            f"{'PASS' if row['passed'] else 'FAIL':<4} | "
            f"target={str(row['target_reached']):<5} | "
            f"obs={str(row['obstacle_activated']):<5} | "
            f"recover={str(row['recovery_requirement']):<5} | "
            f"time={row['elapsed_s']:.2f}s | "
            f"red={row['final_red']:.3f} | "
            f"cent={centroid_text:>7} | "
            f"prox={row['max_prox']:7.1f} | "
            f"blend={row['max_blend']:.2f} | "
            f"nmax={row['max_neural']:.3f} | "
            f"lost={row['lost_frames']} | "
            f"reacq={row['reacquired_count']}"
        )

    print()
    print(
        f"Passed multisensory cases: "
        f"{passed_count}/{len(summaries)}"
    )
    print(
        f"CSV: {output_path}"
    )

    if passed_count == len(summaries):
        print()
        print(
            "STEP 60 PASS."
        )
        print(
            "Vision pursuit, connectome-derived neural proximity "
            "avoidance, and bounded visual reacquisition are now "
            "operating together under one multisensory controller."
        )
    else:
        print()
        print(
            "STEP 60 REVIEW REQUIRED."
        )
        print(
            "Use the per-case neural, visual, blend, lost-target "
            "and reacquisition evidence before changing gains."
        )


if __name__ == "__main__":
    main()
