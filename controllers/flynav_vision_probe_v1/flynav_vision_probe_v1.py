import csv
import math
import statistics
from pathlib import Path

from controller import Camera, Supervisor


# ============================================================
# STEP 49 - WEBOTS CAMERA + VISUAL FIELD ENCODER PROBE
#
# Purpose:
#   Bring up the E-puck camera and validate an ENGINEERED
#   visual-field encoder before connecting camera output to
#   the MaleCNS LC10a -> TuBu -> ER -> EPG pathway.
#
# This controller:
#   1. Keeps the rover stationary.
#   2. Creates/reuses a red visual target.
#   3. Places the target LEFT, CENTER, and RIGHT of the rover.
#   4. Detects red pixels in the camera image.
#   5. Measures target centroid and left/center/right evidence.
#   6. Writes a CSV and prints a 3-position validation summary.
#
# Scientific boundary:
#   Camera pixels -> red-target detection is ENGINEERING.
#   We do NOT yet claim pixel sectors correspond directly to
#   LC10a_L / LC10a_R neurons.
#
# Success target:
#   LEFT   visible with negative image centroid
#   CENTER visible near image center
#   RIGHT  visible with positive image centroid
# ============================================================


TIME_STEP = 32

TARGET_DISTANCE_M = 0.18
TARGET_LATERAL_ANGLE_DEG = 18.0
TARGET_RADIUS_M = 0.030
TARGET_Z_OFFSET_M = 0.040

SETTLE_STEPS = 12
SAMPLE_STEPS = 100

MIN_RED = 120
MIN_RED_ADVANTAGE = 45
VISIBLE_PIXEL_FRACTION = 0.002

PRINT_EVERY = 20


PHASES = [
    {
        "name": "LEFT",
        "relative_angle_deg": +TARGET_LATERAL_ANGLE_DEG,
        "expected": "negative",
    },
    {
        "name": "CENTER",
        "relative_angle_deg": 0.0,
        "expected": "center",
    },
    {
        "name": "RIGHT",
        "relative_angle_deg": -TARGET_LATERAL_ANGLE_DEG,
        "expected": "positive",
    },
]


def get_heading(self_node):
    orientation = self_node.getOrientation()

    # Same world-heading convention behaviorally validated
    # by the existing FlyNav goal-navigation controllers.
    return math.atan2(
        orientation[3],
        orientation[0],
    )


def get_or_create_target(robot):
    existing = robot.getFromDef("VISUAL_TARGET")

    if existing is not None:
        return existing, False

    root = robot.getRoot()
    children = root.getField("children")

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

    target = robot.getFromDef("VISUAL_TARGET")

    if target is None:
        raise RuntimeError(
            "Could not create DEF VISUAL_TARGET."
        )

    return target, True


def place_target(
    target_node,
    robot_x,
    robot_y,
    robot_z,
    heading_rad,
    relative_angle_deg,
):
    relative_rad = math.radians(
        relative_angle_deg
    )

    world_angle = (
        heading_rad
        + relative_rad
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

    translation = target_node.getField(
        "translation"
    )

    if translation is None:
        raise RuntimeError(
            "VISUAL_TARGET has no translation field."
        )

    translation.setSFVec3f(
        [tx, ty, tz]
    )

    return tx, ty, tz


def decode_red_target(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()

    if image is None:
        return {
            "width": width,
            "height": height,
            "visible": False,
            "red_pixels": 0,
            "red_fraction": 0.0,
            "centroid_x_px": float("nan"),
            "centroid_x_norm": float("nan"),
            "left_fraction": 0.0,
            "center_fraction": 0.0,
            "right_fraction": 0.0,
        }

    red_pixels = 0
    x_sum = 0.0

    left_red = 0
    center_red = 0
    right_red = 0

    one_third = width / 3.0
    two_thirds = 2.0 * width / 3.0

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
                and (r - g) >= MIN_RED_ADVANTAGE
                and (r - b) >= MIN_RED_ADVANTAGE
            )

            if not is_red:
                continue

            red_pixels += 1
            x_sum += x

            if x < one_third:
                left_red += 1
            elif x < two_thirds:
                center_red += 1
            else:
                right_red += 1

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

    if red_pixels > 0:
        centroid_x_px = (
            x_sum / red_pixels
        )

        if width > 1:
            centroid_x_norm = (
                2.0
                * centroid_x_px
                / (width - 1)
                - 1.0
            )
        else:
            centroid_x_norm = 0.0

        left_fraction = (
            left_red / red_pixels
        )
        center_fraction = (
            center_red / red_pixels
        )
        right_fraction = (
            right_red / red_pixels
        )

    else:
        centroid_x_px = float("nan")
        centroid_x_norm = float("nan")
        left_fraction = 0.0
        center_fraction = 0.0
        right_fraction = 0.0

    return {
        "width": width,
        "height": height,
        "visible": visible,
        "red_pixels": red_pixels,
        "red_fraction": red_fraction,
        "centroid_x_px": centroid_x_px,
        "centroid_x_norm": centroid_x_norm,
        "left_fraction": left_fraction,
        "center_fraction": center_fraction,
        "right_fraction": right_fraction,
    }


def phase_pass(
    phase_name,
    visible_frames,
    median_centroid,
):
    if visible_frames <= 0:
        return False

    if phase_name == "LEFT":
        return median_centroid <= -0.15

    if phase_name == "CENTER":
        return abs(median_centroid) <= 0.25

    if phase_name == "RIGHT":
        return median_centroid >= +0.15

    return False


def main():
    robot = Supervisor()

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 49 requires supervisor TRUE."
        )

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    output_path = (
        project_root
        / "connectome"
        / "output"
        / "49_vision_probe.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Motors OFF: this is a pure visual sensing test.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Camera bring-up.
    # --------------------------------------------------------

    try:
        camera = robot.getDevice("camera")
    except Exception as exc:
        camera = None
        camera_error = exc
    else:
        camera_error = None

    if camera is None:
        print()
        print(
            "ERROR: device 'camera' was not found."
        )

        print(
            "Available devices:"
        )

        try:
            for i in range(
                robot.getNumberOfDevices()
            ):
                device = (
                    robot.getDeviceByIndex(i)
                )

                print(
                    f"  {i:02d}: "
                    f"{device.getName()}"
                )
        except Exception:
            pass

        if camera_error is not None:
            print(
                f"Original camera error: "
                f"{camera_error}"
            )

        raise RuntimeError(
            "E-puck camera not available. "
            "Check the robot PROTO/device configuration."
        )

    camera.enable(TIME_STEP)

    # Give Webots a few steps to populate camera frames.
    for _ in range(SETTLE_STEPS):
        if robot.step(TIME_STEP) == -1:
            return

    print()
    print("=" * 112)
    print(
        "STEP 49 - WEBOTS CAMERA + "
        "VISUAL FIELD ENCODER PROBE"
    )
    print("=" * 112)
    print()
    print(
        f"Camera resolution: "
        f"{camera.getWidth()} x "
        f"{camera.getHeight()}"
    )
    print(
        f"Horizontal FOV: "
        f"{math.degrees(camera.getFov()):.1f} deg"
    )
    print(
        "Rover remains stationary."
    )
    print(
        "Target color detection = engineered red-pixel encoder."
    )
    print(
        "No LC10a/EPG neural mapping is applied yet."
    )
    print()

    target_node, target_created = (
        get_or_create_target(robot)
    )

    position = self_node.getPosition()

    robot_x = float(position[0])
    robot_y = float(position[1])
    robot_z = float(position[2])

    heading = get_heading(self_node)

    print(
        f"Robot pose: "
        f"x={robot_x:+.3f}, "
        f"y={robot_y:+.3f}, "
        f"heading="
        f"{math.degrees(heading):+.1f} deg"
    )

    fields = [
        "phase",
        "relative_angle_deg",
        "step",
        "time_s",
        "target_x",
        "target_y",
        "target_z",
        "image_width",
        "image_height",
        "target_visible",
        "red_pixels",
        "red_fraction",
        "centroid_x_px",
        "centroid_x_norm",
        "left_fraction",
        "center_fraction",
        "right_fraction",
    ]

    summary = []

    global_step = 0

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

            for phase in PHASES:
                tx, ty, tz = place_target(
                    target_node,
                    robot_x,
                    robot_y,
                    robot_z,
                    heading,
                    phase["relative_angle_deg"],
                )

                print()
                print("-" * 112)
                print(
                    f"PHASE {phase['name']} | "
                    f"relative angle="
                    f"{phase['relative_angle_deg']:+.1f} deg | "
                    f"target=({tx:+.3f},"
                    f"{ty:+.3f},{tz:+.3f})"
                )

                # Let camera image update after target moves.
                for _ in range(SETTLE_STEPS):
                    if robot.step(TIME_STEP) == -1:
                        return

                centroids = []
                visible_frames = 0
                max_red_fraction = 0.0

                for phase_step in range(
                    1,
                    SAMPLE_STEPS + 1,
                ):
                    if robot.step(TIME_STEP) == -1:
                        return

                    global_step += 1

                    visual = decode_red_target(
                        camera
                    )

                    if visual["visible"]:
                        visible_frames += 1

                        centroids.append(
                            visual[
                                "centroid_x_norm"
                            ]
                        )

                    max_red_fraction = max(
                        max_red_fraction,
                        visual[
                            "red_fraction"
                        ],
                    )

                    writer.writerow(
                        {
                            "phase":
                                phase["name"],
                            "relative_angle_deg":
                                phase[
                                    "relative_angle_deg"
                                ],
                            "step":
                                global_step,
                            "time_s":
                                f"{global_step * TIME_STEP / 1000.0:.4f}",
                            "target_x":
                                f"{tx:.6f}",
                            "target_y":
                                f"{ty:.6f}",
                            "target_z":
                                f"{tz:.6f}",
                            "image_width":
                                visual["width"],
                            "image_height":
                                visual["height"],
                            "target_visible":
                                int(
                                    visual[
                                        "visible"
                                    ]
                                ),
                            "red_pixels":
                                visual[
                                    "red_pixels"
                                ],
                            "red_fraction":
                                f"{visual['red_fraction']:.6f}",
                            "centroid_x_px":
                                (
                                    ""
                                    if math.isnan(
                                        visual[
                                            "centroid_x_px"
                                        ]
                                    )
                                    else
                                    f"{visual['centroid_x_px']:.4f}"
                                ),
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
                            "left_fraction":
                                f"{visual['left_fraction']:.6f}",
                            "center_fraction":
                                f"{visual['center_fraction']:.6f}",
                            "right_fraction":
                                f"{visual['right_fraction']:.6f}",
                        }
                    )

                    if (
                        phase_step % PRINT_EVERY
                        == 0
                    ):
                        centroid_text = (
                            "NA"
                            if math.isnan(
                                visual[
                                    "centroid_x_norm"
                                ]
                            )
                            else
                            f"{visual['centroid_x_norm']:+.3f}"
                        )

                        print(
                            f"  frame={phase_step:3d} "
                            f"visible="
                            f"{visual['visible']} "
                            f"red="
                            f"{visual['red_fraction']:.3f} "
                            f"centroid="
                            f"{centroid_text} "
                            f"L/C/R="
                            f"{visual['left_fraction']:.2f}/"
                            f"{visual['center_fraction']:.2f}/"
                            f"{visual['right_fraction']:.2f}"
                        )

                if centroids:
                    median_centroid = (
                        statistics.median(
                            centroids
                        )
                    )
                else:
                    median_centroid = float(
                        "nan"
                    )

                passed = (
                    False
                    if math.isnan(
                        median_centroid
                    )
                    else
                    phase_pass(
                        phase["name"],
                        visible_frames,
                        median_centroid,
                    )
                )

                summary.append(
                    {
                        "phase":
                            phase["name"],
                        "visible_frames":
                            visible_frames,
                        "median_centroid":
                            median_centroid,
                        "max_red_fraction":
                            max_red_fraction,
                        "pass":
                            passed,
                    }
                )

                print(
                    f"  RESULT "
                    f"{'PASS' if passed else 'FAIL'} | "
                    f"visible="
                    f"{visible_frames}/"
                    f"{SAMPLE_STEPS} | "
                    f"median centroid="
                    f"{median_centroid:+.3f} | "
                    f"max red="
                    f"{max_red_fraction:.3f}"
                )

                handle.flush()

    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        if target_created:
            try:
                target_node.remove()
            except Exception:
                pass

    passed_count = sum(
        1
        for row in summary
        if row["pass"]
    )

    print()
    print("=" * 112)
    print("STEP 49 FINAL SUMMARY")
    print("=" * 112)

    for row in summary:
        centroid = row[
            "median_centroid"
        ]

        centroid_text = (
            "NA"
            if math.isnan(centroid)
            else f"{centroid:+.3f}"
        )

        print(
            f"{row['phase']:<8} "
            f"{'PASS' if row['pass'] else 'FAIL':<4} | "
            f"visible="
            f"{row['visible_frames']:>3}/"
            f"{SAMPLE_STEPS} | "
            f"centroid="
            f"{centroid_text:>7} | "
            f"max_red="
            f"{row['max_red_fraction']:.3f}"
        )

    print()
    print(
        f"Passed camera positions: "
        f"{passed_count}/{len(summary)}"
    )
    print(
        f"CSV: {output_path}"
    )

    if passed_count == len(summary):
        print()
        print(
            "PASS: camera and visual-field encoder "
            "are behaviorally aligned."
        )
        print(
            "NEXT: Step 50 will convert this visual "
            "bearing into a connectome-derived LC10a -> "
            "TuBu -> ER -> EPG navigation signal."
        )
    else:
        print()
        print(
            "REVIEW: do not connect vision to the "
            "navigation network yet."
        )
        print(
            "First inspect camera orientation, target "
            "visibility, or left/right image convention."
        )


if __name__ == "__main__":
    main()
