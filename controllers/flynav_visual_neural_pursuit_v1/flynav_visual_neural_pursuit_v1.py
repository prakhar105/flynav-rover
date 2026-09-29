import csv
import math
from pathlib import Path

import pandas as pd
from controller import Camera, Supervisor


# ============================================================
# STEP 56C - CLOSED-LOOP VISUAL NEURAL PURSUIT
#            SAFE-START + CLEAR-SCENE FIX
#
# Fix:
#   Step 56 originally placed the target relative to the rover's
#   CURRENT pose. Because Step 55 had left the rover near the
#   arena boundary, a 0.34 m target offset placed the red target
#   outside the arena.
#
#   Step 56C resets the rover to a safe central pose before
#   every trial AND parks the existing 'test obstacle' away
#   from the visual pursuit corridor.
#
# Everything else remains the same visual-neural pursuit test.
# ============================================================


TIME_STEP = 32

SAFE_START_X = 0.0
SAFE_START_Y = 0.0
SAFE_START_HEADING_RAD = 0.0

TARGET_DISTANCE_M = 0.34
TARGET_LATERAL_ANGLE_DEG = 15.0
TARGET_RADIUS_M = 0.030
TARGET_Z_OFFSET_M = 0.040

SETTLE_STEPS = 12
TRIAL_TIMEOUT_S = 18.0

CENTER_TOLERANCE = 0.10
CENTER_HOLD_FRAMES = 8

MIN_RED = 120
MIN_RED_ADVANTAGE = 45
VISIBLE_PIXEL_FRACTION = 0.002

STOP_RED_FRACTION = 0.155
SLOW_RED_FRACTION = 0.090
CONFIDENCE_RED_FRACTION = 0.10

VISUAL_SMOOTH_ALPHA = 0.30
FORWARD_SPEED_FRACTION = 0.12
MIN_FORWARD_SPEED_FRACTION = 0.035
TURN_SPEED_FRACTION = 0.11

PRINT_EVERY_STEPS = 10

TRIALS = [
    ("LEFT", +TARGET_LATERAL_ANGLE_DEG),
    ("CENTER", 0.0),
    ("RIGHT", -TARGET_LATERAL_ANGLE_DEG),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def get_heading(self_node):
    orientation = self_node.getOrientation()
    return math.atan2(
        orientation[3],
        orientation[0],
    )


def reset_rover_to_safe_start(self_node):
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
        # E-puck rotates around world Z in this arena.
        rotation.setSFRotation(
            [
                0.0,
                0.0,
                1.0,
                SAFE_START_HEADING_RAD,
            ]
        )

    self_node.resetPhysics()


def find_top_level_node_by_name(robot, wanted_name):
    """
    Find a top-level world node whose SFString 'name' field
    matches wanted_name. In the current FlyNav world the
    existing white obstacle is a top-level Solid named
    'test obstacle'.
    """
    root = robot.getRoot()
    children = root.getField("children")

    if children is None:
        return None

    for i in range(children.getCount()):
        node = children.getMFNode(i)

        if node is None:
            continue

        name_field = node.getField("name")

        if name_field is None:
            continue

        try:
            node_name = name_field.getSFString()
        except Exception:
            continue

        if node_name == wanted_name:
            return node

    return None


def park_test_obstacle(robot):
    """
    Step 56 is a VISUAL-ONLY pursuit test, so the old obstacle
    must not occlude the red target. Park it at the same
    clear-path location used by the earlier navigation tests.

    Returns:
        (node, original_translation) or (None, None)
    """
    obstacle = find_top_level_node_by_name(
        robot,
        "test obstacle",
    )

    if obstacle is None:
        return None, None

    translation = obstacle.getField(
        "translation"
    )

    if translation is None:
        return obstacle, None

    original = list(
        translation.getSFVec3f()
    )

    z = float(original[2])

    translation.setSFVec3f(
        [0.0, 0.35, z]
    )

    try:
        obstacle.resetPhysics()
    except Exception:
        pass

    return obstacle, original


def restore_obstacle(
    obstacle,
    original_translation,
):
    if (
        obstacle is None
        or original_translation is None
    ):
        return

    field = obstacle.getField(
        "translation"
    )

    if field is not None:
        field.setSFVec3f(
            original_translation
        )

    try:
        obstacle.resetPhysics()
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

    robot_x = float(position[0])
    robot_y = float(position[1])
    robot_z = float(position[2])

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
        robot_x
        + TARGET_DISTANCE_M
        * math.cos(world_angle)
    )

    ty = (
        robot_y
        + TARGET_DISTANCE_M
        * math.sin(world_angle)
    )

    tz = (
        robot_z
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

    return (
        tx,
        ty,
        tz,
        math.degrees(heading),
    )


def decode_red_target(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()

    if image is None:
        return {
            "visible": False,
            "red_fraction": 0.0,
            "centroid_x_norm": float("nan"),
        }

    red_pixels = 0
    x_sum = 0.0

    for y in range(height):
        for x in range(width):
            r = Camera.imageGetRed(
                image, width, x, y
            )
            g = Camera.imageGetGreen(
                image, width, x, y
            )
            b = Camera.imageGetBlue(
                image, width, x, y
            )

            if not (
                r >= MIN_RED
                and (r - g)
                >= MIN_RED_ADVANTAGE
                and (r - b)
                >= MIN_RED_ADVANTAGE
            ):
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
        centroid = float("nan")
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


def load_structural_bridge(
    project_root,
):
    paths_path = (
        project_root
        / "connectome"
        / "output"
        / "53_lc10a_aotu019_dna_paths.csv"
    )

    aotu_path = (
        project_root
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

    dna02_L_drive = (
        1.0 - inhibit_L
    )

    dna02_R_drive = (
        1.0 - inhibit_R
    )

    steering = (
        dna02_R_drive
        - dna02_L_drive
    )

    return {
        "confidence": confidence,
        "dna02_L_drive": dna02_L_drive,
        "dna02_R_drive": dna02_R_drive,
        "steering": steering,
    }


def forward_scale(
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
            MIN_FORWARD_SPEED_FRACTION
            / FORWARD_SPEED_FRACTION,
        )

    return clamp(
        scale,
        0.0,
        1.0,
    )


def main():
    robot = Supervisor()

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 56C requires supervisor TRUE."
        )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    bridge = load_structural_bridge(
        project_root
    )

    output_path = (
        project_root
        / "connectome"
        / "output"
        / "56c_visual_neural_pursuit_clear_scene.csv"
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

    target_node, target_created = (
        get_or_create_target(
            robot
        )
    )

    # --------------------------------------------------------
    # VISUAL-ONLY TEST SCENE:
    # Park the existing white "test obstacle" away from the
    # pursuit corridor so it cannot occlude CENTER/RIGHT
    # target placements. Step 57 will deliberately bring
    # proximity obstacles back.
    # --------------------------------------------------------
    obstacle_node, obstacle_original = (
        park_test_obstacle(
            robot
        )
    )

    for _ in range(
        SETTLE_STEPS
    ):
        if robot.step(
            TIME_STEP
        ) == -1:
            return

    if obstacle_node is None:
        print(
            "Scene note: no top-level node named "
            "'test obstacle' was found."
        )
    else:
        print(
            "Scene note: 'test obstacle' parked at "
            "(0.00,+0.35) for this visual-only test."
        )

    fields = [
        "trial",
        "step",
        "time_s",
        "target_visible",
        "red_fraction",
        "centroid_x_norm",
        "raw_steering",
        "smoothed_steering",
        "forward_speed",
        "turn_component",
        "left_wheel",
        "right_wheel",
        "arrival_hold_frames",
    ]

    summaries = []

    print()
    print("=" * 118)
    print(
        "STEP 56C - VISUAL NEURAL PURSUIT SAFE-START + CLEAR-SCENE FIX"
    )
    print("=" * 118)
    print(
        "Rover is reset to arena center before EVERY trial."
    )
    print(
        "Rover reset removes the out-of-arena bug; the test obstacle is also parked away from the camera corridor."
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

            for (
                trial_name,
                angle_deg,
            ) in TRIALS:
                left_motor.setVelocity(
                    0.0
                )

                right_motor.setVelocity(
                    0.0
                )

                reset_rover_to_safe_start(
                    self_node
                )

                for _ in range(
                    SETTLE_STEPS
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                (
                    tx,
                    ty,
                    tz,
                    start_heading,
                ) = place_target_relative(
                    target_node,
                    self_node,
                    angle_deg,
                )

                for _ in range(
                    SETTLE_STEPS
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                visual0 = decode_red_target(
                    camera
                )

                actual_position = self_node.getPosition()
                actual_heading = math.degrees(
                    get_heading(
                        self_node
                    )
                )

                initial_centroid = visual0[
                    "centroid_x_norm"
                ]

                initial_centroid_text = (
                    "NA"
                    if math.isnan(
                        initial_centroid
                    )
                    else
                    f"{initial_centroid:+.3f}"
                )

                print("-" * 118)
                print(
                    f"TRIAL {trial_name:<6} | "
                    f"rover=({float(actual_position[0]):+.3f},"
                    f"{float(actual_position[1]):+.3f}) | "
                    f"heading={actual_heading:+.1f} deg | "
                    f"target angle={angle_deg:+.1f} deg | "
                    f"target=({tx:+.3f},{ty:+.3f},{tz:+.3f}) | "
                    f"initial visible={visual0['visible']} "
                    f"red={visual0['red_fraction']:.3f} "
                    f"cent={initial_centroid_text}"
                )

                if not visual0[
                    "visible"
                ]:
                    summaries.append(
                        {
                            "trial": trial_name,
                            "passed": False,
                            "elapsed_s": 0.0,
                            "red": visual0[
                                "red_fraction"
                            ],
                            "centroid": visual0[
                                "centroid_x_norm"
                            ],
                            "lost": 1,
                        }
                    )

                    print(
                        "  FAIL EARLY: target is not visible "
                        "from the safe start."
                    )

                    continue

                smoothed = 0.0
                hold = 0
                lost = 0
                passed = False

                max_steps = int(
                    TRIAL_TIMEOUT_S
                    * 1000.0
                    / TIME_STEP
                )

                final_red = 0.0
                final_centroid = float(
                    "nan"
                )

                step_count = 0

                while (
                    step_count < max_steps
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                    step_count += 1

                    visual = decode_red_target(
                        camera
                    )

                    if not visual[
                        "visible"
                    ]:
                        lost += 1

                        left_speed = 0.0
                        right_speed = 0.0
                        raw = 0.0
                        smoothed = 0.0
                        forward_speed = 0.0
                        turn = 0.0
                        hold = 0

                    else:
                        centroid = float(
                            visual[
                                "centroid_x_norm"
                            ]
                        )

                        final_centroid = (
                            centroid
                        )

                        final_red = float(
                            visual[
                                "red_fraction"
                            ]
                        )

                        neural = (
                            visual_neural_steering(
                                visual,
                                bridge,
                            )
                        )

                        raw = float(
                            neural[
                                "steering"
                            ]
                        )

                        smoothed = (
                            (
                                1.0
                                - VISUAL_SMOOTH_ALPHA
                            )
                            * smoothed
                            +
                            VISUAL_SMOOTH_ALPHA
                            * raw
                        )

                        arrived_size = (
                            visual[
                                "red_fraction"
                            ]
                            >= STOP_RED_FRACTION
                        )

                        centered = (
                            abs(
                                centroid
                            )
                            <= CENTER_TOLERANCE
                        )

                        if (
                            arrived_size
                            and centered
                        ):
                            hold += 1

                            forward_speed = 0.0
                            turn = 0.0

                            left_speed = 0.0
                            right_speed = 0.0

                        else:
                            hold = 0

                            scale = (
                                forward_scale(
                                    visual[
                                        "red_fraction"
                                    ],
                                    centroid,
                                )
                            )

                            forward_speed = (
                                FORWARD_SPEED_FRACTION
                                * max_speed
                                * scale
                            )

                            turn = (
                                TURN_SPEED_FRACTION
                                * max_speed
                                * smoothed
                            )

                            left_speed = (
                                forward_speed
                                + turn
                            )

                            right_speed = (
                                forward_speed
                                - turn
                            )

                    left_speed = clamp(
                        left_speed,
                        -max_speed,
                        +max_speed,
                    )

                    right_speed = clamp(
                        right_speed,
                        -max_speed,
                        +max_speed,
                    )

                    left_motor.setVelocity(
                        left_speed
                    )

                    right_motor.setVelocity(
                        right_speed
                    )

                    writer.writerow(
                        {
                            "trial":
                                trial_name,
                            "step":
                                step_count,
                            "time_s":
                                f"{step_count * TIME_STEP / 1000.0:.4f}",
                            "target_visible":
                                int(
                                    visual[
                                        "visible"
                                    ]
                                ),
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
                            "raw_steering":
                                f"{raw:.6f}",
                            "smoothed_steering":
                                f"{smoothed:.6f}",
                            "forward_speed":
                                f"{forward_speed:.6f}",
                            "turn_component":
                                f"{turn:.6f}",
                            "left_wheel":
                                f"{left_speed:.6f}",
                            "right_wheel":
                                f"{right_speed:.6f}",
                            "arrival_hold_frames":
                                hold,
                        }
                    )

                    if (
                        step_count
                        % PRINT_EVERY_STEPS
                        == 0
                    ):
                        c = visual[
                            "centroid_x_norm"
                        ]

                        c_text = (
                            "NA"
                            if math.isnan(c)
                            else
                            f"{c:+.3f}"
                        )

                        print(
                            f"  t={step_count * TIME_STEP / 1000.0:5.2f}s "
                            f"vis={visual['visible']} "
                            f"red={visual['red_fraction']:.3f} "
                            f"cent={c_text:>7} "
                            f"raw={raw:+.3f} "
                            f"smooth={smoothed:+.3f} "
                            f"fwd={forward_speed:+.2f} "
                            f"turn={turn:+.2f} "
                            f"hold={hold}"
                        )

                    if (
                        hold
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

                summaries.append(
                    {
                        "trial": trial_name,
                        "passed": passed,
                        "elapsed_s": elapsed,
                        "red": final_red,
                        "centroid": final_centroid,
                        "lost": lost,
                    }
                )

                c_text = (
                    "NA"
                    if math.isnan(
                        final_centroid
                    )
                    else
                    f"{final_centroid:+.3f}"
                )

                print(
                    f"  RESULT "
                    f"{'PASS' if passed else 'FAIL'} | "
                    f"time={elapsed:.2f}s | "
                    f"red={final_red:.3f} | "
                    f"centroid={c_text} | "
                    f"lost={lost}"
                )

                handle.flush()

    finally:
        left_motor.setVelocity(
            0.0
        )

        right_motor.setVelocity(
            0.0
        )

        restore_obstacle(
            obstacle_node,
            obstacle_original,
        )

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
    print("=" * 118)
    print(
        "STEP 56C FINAL SUMMARY"
    )
    print("=" * 118)

    for row in summaries:
        c = row["centroid"]

        c_text = (
            "NA"
            if math.isnan(c)
            else
            f"{c:+.3f}"
        )

        print(
            f"{row['trial']:<8} "
            f"{'PASS' if row['passed'] else 'FAIL':<4} | "
            f"time={row['elapsed_s']:.2f}s | "
            f"red={row['red']:.3f} | "
            f"centroid={c_text:>7} | "
            f"lost={row['lost']}"
        )

    print()
    print(
        f"Passed visual-pursuit trials: "
        f"{passed_count}/{len(summaries)}"
    )
    print(
        f"CSV: {output_path}"
    )


if __name__ == "__main__":
    main()
