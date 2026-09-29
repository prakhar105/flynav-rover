import csv
import math
from pathlib import Path

import pandas as pd
from controller import Camera, Supervisor


# ============================================================
# STEP 55 - CLOSED-LOOP VISUAL NEURAL CENTERING
#
# Goal:
#   Let the E-puck ROTATE under the connectome-derived visual
#   action bridge validated in Steps 53-54.
#
# Camera bearing
#   -> engineered visual-field encoding
#   -> LC10a_L / LC10a_R
#   -> AOTU019_L / AOTU019_R (GABA)
#   -> contralateral DNa02
#   -> frozen FlyNav steering convention
#   -> differential-drive rotation
#
# IMPORTANT:
#   - Camera bearing -> LC10a side is an ENGINEERED interface.
#   - LC10a/AOTU019/DNa02 identities, AOTU019 GABA sign,
#     and anatomical weights are MaleCNS-derived.
#   - This test is ROTATION ONLY. No forward pursuit yet.
#   - If the target disappears, the rover STOPS. Search /
#     reacquisition is a later behavior.
#
# Automated trials:
#   LEFT   target at +18 deg
#   RIGHT  target at -18 deg
#   CENTER target at  0 deg
#
# PASS:
#   abs(camera centroid) <= CENTER_TOLERANCE
#   for CENTER_HOLD_FRAMES consecutive frames.
# ============================================================


TIME_STEP = 32

TARGET_DISTANCE_M = 0.18
TARGET_LATERAL_ANGLE_DEG = 18.0
TARGET_RADIUS_M = 0.030
TARGET_Z_OFFSET_M = 0.040

SETTLE_STEPS = 8
TRIAL_TIMEOUT_S = 10.0

CENTER_TOLERANCE = 0.08
CENTER_HOLD_FRAMES = 10

MIN_RED = 120
MIN_RED_ADVANTAGE = 45
VISIBLE_PIXEL_FRACTION = 0.002

# Continuous camera-bearing encoder.
# Confidence reaches 1 around the red-area levels seen in Step 49.
CONFIDENCE_RED_FRACTION = 0.10

# Robotics-layer smoothing and motor calibration.
VISUAL_SMOOTH_ALPHA = 0.30
TURN_SPEED_FRACTION = 0.08

PRINT_EVERY_STEPS = 8

TRIALS = [
    ("LEFT", +TARGET_LATERAL_ANGLE_DEG),
    ("RIGHT", -TARGET_LATERAL_ANGLE_DEG),
    ("CENTER", 0.0),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def get_heading(self_node):
    orientation = self_node.getOrientation()

    # Same engineering world-heading convention used by
    # the existing FlyNav navigation controllers.
    return math.atan2(
        orientation[3],
        orientation[0],
    )


def get_or_create_target(robot):
    target = robot.getFromDef("VISUAL_TARGET")

    if target is not None:
        return target, False

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


def place_target_relative(
    target_node,
    self_node,
    relative_angle_deg,
):
    position = self_node.getPosition()

    robot_x = float(position[0])
    robot_y = float(position[1])
    robot_z = float(position[2])

    heading = get_heading(self_node)

    world_angle = (
        heading
        + math.radians(relative_angle_deg)
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


def load_structural_bridge(project_root):
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

    if not paths_path.exists():
        raise FileNotFoundError(
            f"Missing Step 53 file: {paths_path}"
        )

    if not aotu_path.exists():
        raise FileNotFoundError(
            f"Missing Step 53 file: {aotu_path}"
        )

    paths = pd.read_csv(paths_path)
    aotu = pd.read_csv(aotu_path)

    predicted_nt = set(
        aotu["predictedNt"]
        .dropna()
        .astype(str)
        .str.lower()
    )

    if "gaba" not in predicted_nt:
        raise RuntimeError(
            "AOTU019 GABA annotation not present. "
            "Stop before visual motor control."
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
            "Missing LC10a_L -> AOTU019_L -> DNa02_R."
        )

    if right_path.empty:
        raise RuntimeError(
            "Missing LC10a_R -> AOTU019_R -> DNa02_L."
        )

    left_score = float(
        left_path["bottleneck_weight"].sum()
    )

    right_score = float(
        right_path["bottleneck_weight"].sum()
    )

    return {
        "left_score": left_score,
        "right_score": right_score,
        "left_gain": 1.0 / left_score,
        "right_gain": 1.0 / right_score,
    }


def visual_neural_steering(
    visual,
    bridge,
):
    if (
        not visual["visible"]
        or math.isnan(
            visual["centroid_x_norm"]
        )
    ):
        return {
            "confidence": 0.0,
            "visual_L": 0.0,
            "visual_R": 0.0,
            "inhibit_DNa02_R": 0.0,
            "inhibit_DNa02_L": 0.0,
            "dna02_L_drive": 1.0,
            "dna02_R_drive": 1.0,
            "steering": 0.0,
        }

    centroid = clamp(
        float(
            visual["centroid_x_norm"]
        ),
        -1.0,
        +1.0,
    )

    confidence = clamp(
        visual["red_fraction"]
        / CONFIDENCE_RED_FRACTION,
        0.0,
        1.0,
    )

    # ENGINEERED sensor interface:
    # camera-left evidence -> LC10a_L channel
    # camera-right evidence -> LC10a_R channel
    visual_L = (
        max(0.0, -centroid)
        * confidence
    )

    visual_R = (
        max(0.0, +centroid)
        * confidence
    )

    raw_inhibit_R = (
        visual_L
        * bridge["left_score"]
    )

    raw_inhibit_L = (
        visual_R
        * bridge["right_score"]
    )

    # Per-side normalization from Step 54.
    inhibit_R = clamp(
        raw_inhibit_R
        * bridge["left_gain"],
        0.0,
        1.0,
    )

    inhibit_L = clamp(
        raw_inhibit_L
        * bridge["right_gain"],
        0.0,
        1.0,
    )

    # Equal tonic baseline is a deterministic surrogate
    # for signed DNa02 motor-bias validation.
    dna02_L_drive = (
        1.0 - inhibit_L
    )

    dna02_R_drive = (
        1.0 - inhibit_R
    )

    # Frozen FlyNav sign convention:
    # + = DNa02_R dominance = RIGHT
    # - = DNa02_L dominance = LEFT
    steering = (
        dna02_R_drive
        - dna02_L_drive
    )

    return {
        "confidence": confidence,
        "visual_L": visual_L,
        "visual_R": visual_R,
        "inhibit_DNa02_R": inhibit_R,
        "inhibit_DNa02_L": inhibit_L,
        "dna02_L_drive": dna02_L_drive,
        "dna02_R_drive": dna02_R_drive,
        "steering": steering,
    }


def main():
    robot = Supervisor()
    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 55 requires supervisor TRUE."
        )

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    bridge = load_structural_bridge(
        project_root
    )

    output_path = (
        project_root
        / "connectome"
        / "output"
        / "55_visual_neural_centering.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)

    for _ in range(SETTLE_STEPS):
        if robot.step(TIME_STEP) == -1:
            return

    target_node, target_created = (
        get_or_create_target(robot)
    )

    print()
    print("=" * 118)
    print(
        "STEP 55 - CLOSED-LOOP VISUAL NEURAL CENTERING"
    )
    print("=" * 118)
    print()
    print(
        f"Camera: {camera.getWidth()} x "
        f"{camera.getHeight()} | "
        f"FOV={math.degrees(camera.getFov()):.1f} deg"
    )
    print(
        "Motor mode: rotation only; no forward pursuit."
    )
    print(
        "Target lost behavior: STOP."
    )
    print()
    print(
        "MaleCNS bridge loaded:"
    )
    print(
        "  LC10a_L -> AOTU019_L -| DNa02_R "
        f"score={bridge['left_score']:.1f}"
    )
    print(
        "  LC10a_R -> AOTU019_R -| DNa02_L "
        f"score={bridge['right_score']:.1f}"
    )
    print()

    fields = [
        "trial",
        "step",
        "time_s",
        "target_relative_angle_deg",
        "target_visible",
        "red_fraction",
        "centroid_x_norm",
        "confidence",
        "visual_L",
        "visual_R",
        "inhibit_DNa02_R",
        "inhibit_DNa02_L",
        "dna02_L_drive",
        "dna02_R_drive",
        "raw_steering",
        "smoothed_steering",
        "left_wheel",
        "right_wheel",
        "center_hold_frames",
    ]

    summaries = []

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

            for trial_name, angle_deg in TRIALS:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                (
                    tx,
                    ty,
                    tz,
                    start_heading_deg,
                ) = place_target_relative(
                    target_node,
                    self_node,
                    angle_deg,
                )

                for _ in range(SETTLE_STEPS):
                    if robot.step(TIME_STEP) == -1:
                        return

                smoothed_steering = 0.0
                center_hold = 0
                trial_step = 0
                passed = False
                last_centroid = float("nan")

                max_steps = int(
                    TRIAL_TIMEOUT_S
                    * 1000.0
                    / TIME_STEP
                )

                print("-" * 118)
                print(
                    f"TRIAL {trial_name:<6} | "
                    f"target angle={angle_deg:+.1f} deg | "
                    f"start heading={start_heading_deg:+.1f} deg | "
                    f"target=({tx:+.3f},{ty:+.3f},{tz:+.3f})"
                )

                while trial_step < max_steps:
                    if robot.step(TIME_STEP) == -1:
                        return

                    trial_step += 1

                    visual = decode_red_target(
                        camera
                    )

                    neural = visual_neural_steering(
                        visual,
                        bridge,
                    )

                    raw_steering = float(
                        neural["steering"]
                    )

                    smoothed_steering = (
                        (1.0 - VISUAL_SMOOTH_ALPHA)
                        * smoothed_steering
                        +
                        VISUAL_SMOOTH_ALPHA
                        * raw_steering
                    )

                    centroid = visual[
                        "centroid_x_norm"
                    ]

                    last_centroid = centroid

                    if not visual["visible"]:
                        # No search behavior yet.
                        left_speed = 0.0
                        right_speed = 0.0
                        center_hold = 0

                    elif (
                        not math.isnan(centroid)
                        and abs(centroid)
                        <= CENTER_TOLERANCE
                    ):
                        left_speed = 0.0
                        right_speed = 0.0

                        center_hold += 1

                    else:
                        center_hold = 0

                        turn_component = (
                            TURN_SPEED_FRACTION
                            * max_speed
                            * smoothed_steering
                        )

                        # Frozen FlyNav differential-drive
                        # steering convention:
                        # + -> RIGHT
                        # - -> LEFT
                        left_speed = turn_component
                        right_speed = -turn_component

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
                            "trial": trial_name,
                            "step": trial_step,
                            "time_s":
                                f"{trial_step * TIME_STEP / 1000.0:.4f}",
                            "target_relative_angle_deg":
                                angle_deg,
                            "target_visible":
                                int(visual["visible"]),
                            "red_fraction":
                                f"{visual['red_fraction']:.6f}",
                            "centroid_x_norm":
                                (
                                    ""
                                    if math.isnan(centroid)
                                    else f"{centroid:.6f}"
                                ),
                            "confidence":
                                f"{neural['confidence']:.6f}",
                            "visual_L":
                                f"{neural['visual_L']:.6f}",
                            "visual_R":
                                f"{neural['visual_R']:.6f}",
                            "inhibit_DNa02_R":
                                f"{neural['inhibit_DNa02_R']:.6f}",
                            "inhibit_DNa02_L":
                                f"{neural['inhibit_DNa02_L']:.6f}",
                            "dna02_L_drive":
                                f"{neural['dna02_L_drive']:.6f}",
                            "dna02_R_drive":
                                f"{neural['dna02_R_drive']:.6f}",
                            "raw_steering":
                                f"{raw_steering:.6f}",
                            "smoothed_steering":
                                f"{smoothed_steering:.6f}",
                            "left_wheel":
                                f"{left_speed:.6f}",
                            "right_wheel":
                                f"{right_speed:.6f}",
                            "center_hold_frames":
                                center_hold,
                        }
                    )

                    if (
                        trial_step % PRINT_EVERY_STEPS
                        == 0
                    ):
                        centroid_text = (
                            "NA"
                            if math.isnan(centroid)
                            else f"{centroid:+.3f}"
                        )

                        print(
                            f"  t={trial_step * TIME_STEP / 1000.0:5.2f}s "
                            f"visible={visual['visible']} "
                            f"centroid={centroid_text:>7} "
                            f"DNaProxy=("
                            f"{neural['dna02_L_drive']:.2f},"
                            f"{neural['dna02_R_drive']:.2f}) "
                            f"raw={raw_steering:+.3f} "
                            f"smooth={smoothed_steering:+.3f} "
                            f"wheels=("
                            f"{left_speed:+.2f},"
                            f"{right_speed:+.2f}) "
                            f"hold={center_hold}"
                        )

                    if (
                        center_hold
                        >= CENTER_HOLD_FRAMES
                    ):
                        passed = True
                        break

                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

                final_heading = math.degrees(
                    get_heading(self_node)
                )

                elapsed_s = (
                    trial_step
                    * TIME_STEP
                    / 1000.0
                )

                summaries.append(
                    {
                        "trial": trial_name,
                        "passed": passed,
                        "elapsed_s": elapsed_s,
                        "final_centroid":
                            last_centroid,
                        "start_heading_deg":
                            start_heading_deg,
                        "final_heading_deg":
                            final_heading,
                    }
                )

                centroid_text = (
                    "NA"
                    if math.isnan(
                        last_centroid
                    )
                    else f"{last_centroid:+.3f}"
                )

                print(
                    f"  RESULT "
                    f"{'PASS' if passed else 'FAIL'} | "
                    f"time={elapsed_s:.2f}s | "
                    f"final centroid={centroid_text} | "
                    f"heading "
                    f"{start_heading_deg:+.1f}"
                    f" -> "
                    f"{final_heading:+.1f} deg"
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
        for row in summaries
        if row["passed"]
    )

    print()
    print("=" * 118)
    print("STEP 55 FINAL SUMMARY")
    print("=" * 118)

    for row in summaries:
        centroid_text = (
            "NA"
            if math.isnan(
                row["final_centroid"]
            )
            else
            f"{row['final_centroid']:+.3f}"
        )

        print(
            f"{row['trial']:<8} "
            f"{'PASS' if row['passed'] else 'FAIL':<4} | "
            f"time={row['elapsed_s']:.2f}s | "
            f"centroid={centroid_text:>7} | "
            f"heading="
            f"{row['start_heading_deg']:+.1f}"
            f" -> "
            f"{row['final_heading_deg']:+.1f}"
        )

    print()
    print(
        f"Passed visual-centering trials: "
        f"{passed_count}/{len(summaries)}"
    )
    print(
        f"CSV: {output_path}"
    )

    if passed_count == len(summaries):
        print()
        print(
            "STEP 55 PASS."
        )
        print(
            "The rover can now close the loop from camera "
            "bearing through the MaleCNS-derived "
            "LC10a -> AOTU019 -> DNa02 bridge and rotate "
            "to center the target."
        )
        print(
            "NEXT: add controlled forward visual pursuit, "
            "then combine it with neural proximity avoidance."
        )
    else:
        print()
        print(
            "STEP 55 REVIEW REQUIRED."
        )
        print(
            "Do not add forward pursuit yet."
        )


if __name__ == "__main__":
    main()
