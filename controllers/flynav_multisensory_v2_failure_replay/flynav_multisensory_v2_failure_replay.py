import csv
import importlib.util
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from controller import Supervisor


# ============================================================
# STEP 59A - V2 FAILED-GEOMETRY REPLAY
#
# Frozen controlled baseline under test:
#   Step 57C = 4/4 controlled PASS
#
# This script DOES NOT tune the neural pathways.
# It stress-tests the same architecture across 30 deterministic
# randomized scenarios.
#
# Sensory branches:
#
#   CAMERA
#     -> engineered image/visual-field interface
#     -> LC10a -> AOTU019 (GABA) -> DNa02 visual steering
#
#   PROXIMITY
#     -> engineered sensor normalization
#     -> FlyNavBrain.step()
#     -> connectome-derived neural obstacle steering
#
# Fusion:
#   engineered hysteresis / blending / bounded reacquisition
#
# Randomized scenario seed:
#   20260928
#
# Scenario families:
#   - clear
#   - left_conflict
#   - right_conflict
#   - near_center
#
# Outputs:
#   59a_v2_failure_replay_scenarios.csv
#   59a_v2_failure_replay_detail.csv
#   59a_v2_failure_replay_results.csv
#
# Scientific boundary:
#   Random geometry, arbitration, visual reacquisition, and
#   camera/proximity encoders are engineering.
#   Connectome-derived visual/obstacle pathways remain unchanged.
# ============================================================


SEED = 20260928
TRIAL_COUNT = 30
TIME_STEP = 32
TRIAL_TIMEOUT_S = 45.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Dynamically load Step 59B Multisensory Neural Navigation V2.
# This replay uses the SAME V2 sensory encoders, connectome
# bridge, obstacle-brain interface, deadlock release, and
# two-stage reacquisition policy. No retuning happens here.
# ------------------------------------------------------------

BASE_CONTROLLER_PATH = (
    PROJECT_ROOT
    / "controllers"
    / "flynav_multisensory_neural_v2"
    / "flynav_multisensory_neural_v2.py"
)

if not BASE_CONTROLLER_PATH.exists():
    raise FileNotFoundError(
        "Step 59B V2 controller was not found:\n"
        f"{BASE_CONTROLLER_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "flynav_step59b_base",
    BASE_CONTROLLER_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not import Step 59B V2 controller."
    )

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# ============================================================
# Randomized validation envelope
# ============================================================

START_X = -0.25
START_Y_MIN = -0.06
START_Y_MAX = +0.06
START_HEADING_MIN_DEG = -8.0
START_HEADING_MAX_DEG = +8.0

TARGET_DISTANCE_MIN = 0.42
TARGET_DISTANCE_MAX = 0.48

CLEAR_TARGET_ANGLE_MIN = -12.0
CLEAR_TARGET_ANGLE_MAX = +12.0

CONFLICT_TARGET_MIN_ABS = 4.0
CONFLICT_TARGET_MAX_ABS = 12.0

CENTER_TARGET_ANGLE_MIN = -5.0
CENTER_TARGET_ANGLE_MAX = +5.0

OBSTACLE_PATH_FRACTION_MIN = 0.50
OBSTACLE_PATH_FRACTION_MAX = 0.58

CONFLICT_LATERAL_MIN = 0.070
CONFLICT_LATERAL_MAX = 0.100

CENTER_LATERAL_MIN = 0.040
CENTER_LATERAL_MAX = 0.065

CLEAR_OBSTACLE_X = 0.00
CLEAR_OBSTACLE_Y = 0.35

SETTLE_STEPS = 12
PRINT_EVERY_STEPS = 25

CATEGORY_COUNTS = {
    "clear": 6,
    "left_conflict": 8,
    "right_conflict": 8,
    "near_center": 8,
}

if sum(CATEGORY_COUNTS.values()) != TRIAL_COUNT:
    raise RuntimeError(
        "CATEGORY_COUNTS must sum to TRIAL_COUNT."
    )


# ============================================================
# Helpers
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def reset_rover_pose(
    self_node,
    x,
    y,
    heading_deg,
):
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
            float(x),
            float(y),
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
                math.radians(
                    heading_deg
                ),
            ]
        )

    self_node.resetPhysics()


def euclidean_xy(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def place_target(
    target_node,
    start_x,
    start_y,
    start_z,
    heading_deg,
    relative_angle_deg,
    distance_m,
):
    world_angle = math.radians(
        heading_deg
        + relative_angle_deg
    )

    tx = (
        start_x
        + distance_m
        * math.cos(world_angle)
    )

    ty = (
        start_y
        + distance_m
        * math.sin(world_angle)
    )

    tz = (
        start_z
        + base.TARGET_Z_OFFSET_M
    )

    field = target_node.getField(
        "translation"
    )

    if field is None:
        raise RuntimeError(
            "VISUAL_TARGET has no translation field."
        )

    field.setSFVec3f(
        [
            tx,
            ty,
            tz,
        ]
    )

    return tx, ty, tz


def obstacle_from_path(
    start_x,
    start_y,
    heading_deg,
    target_angle_deg,
    target_distance,
    path_fraction,
    lateral_offset,
):
    world_angle = math.radians(
        heading_deg
        + target_angle_deg
    )

    ux = math.cos(
        world_angle
    )
    uy = math.sin(
        world_angle
    )

    # Left normal to the target ray.
    px = -uy
    py = +ux

    along = (
        target_distance
        * path_fraction
    )

    ox = (
        start_x
        + along * ux
        + lateral_offset * px
    )

    oy = (
        start_y
        + along * uy
        + lateral_offset * py
    )

    return ox, oy


def generate_scenarios():
    rng = random.Random(
        SEED
    )

    categories = []

    for category, count in (
        CATEGORY_COUNTS.items()
    ):
        categories.extend(
            [category] * count
        )

    rng.shuffle(
        categories
    )

    scenarios = []

    for index, category in enumerate(
        categories,
        start=1,
    ):
        start_y = rng.uniform(
            START_Y_MIN,
            START_Y_MAX,
        )

        heading_deg = rng.uniform(
            START_HEADING_MIN_DEG,
            START_HEADING_MAX_DEG,
        )

        target_distance = rng.uniform(
            TARGET_DISTANCE_MIN,
            TARGET_DISTANCE_MAX,
        )

        if category == "clear":
            target_angle = rng.uniform(
                CLEAR_TARGET_ANGLE_MIN,
                CLEAR_TARGET_ANGLE_MAX,
            )

            obstacle_x = (
                CLEAR_OBSTACLE_X
            )

            obstacle_y = (
                CLEAR_OBSTACLE_Y
            )

            lateral_offset = float(
                "nan"
            )

            path_fraction = float(
                "nan"
            )

            expect_obstacle = False

        elif (
            category
            == "left_conflict"
        ):
            target_angle = rng.uniform(
                CONFLICT_TARGET_MIN_ABS,
                CONFLICT_TARGET_MAX_ABS,
            )

            lateral_offset = rng.uniform(
                CONFLICT_LATERAL_MIN,
                CONFLICT_LATERAL_MAX,
            )

            path_fraction = rng.uniform(
                OBSTACLE_PATH_FRACTION_MIN,
                OBSTACLE_PATH_FRACTION_MAX,
            )

            obstacle_x, obstacle_y = (
                obstacle_from_path(
                    START_X,
                    start_y,
                    heading_deg,
                    target_angle,
                    target_distance,
                    path_fraction,
                    lateral_offset,
                )
            )

            expect_obstacle = True

        elif (
            category
            == "right_conflict"
        ):
            target_angle = -rng.uniform(
                CONFLICT_TARGET_MIN_ABS,
                CONFLICT_TARGET_MAX_ABS,
            )

            lateral_offset = -rng.uniform(
                CONFLICT_LATERAL_MIN,
                CONFLICT_LATERAL_MAX,
            )

            path_fraction = rng.uniform(
                OBSTACLE_PATH_FRACTION_MIN,
                OBSTACLE_PATH_FRACTION_MAX,
            )

            obstacle_x, obstacle_y = (
                obstacle_from_path(
                    START_X,
                    start_y,
                    heading_deg,
                    target_angle,
                    target_distance,
                    path_fraction,
                    lateral_offset,
                )
            )

            expect_obstacle = True

        elif (
            category
            == "near_center"
        ):
            target_angle = rng.uniform(
                CENTER_TARGET_ANGLE_MIN,
                CENTER_TARGET_ANGLE_MAX,
            )

            sign = rng.choice(
                [-1.0, +1.0]
            )

            lateral_offset = (
                sign
                * rng.uniform(
                    CENTER_LATERAL_MIN,
                    CENTER_LATERAL_MAX,
                )
            )

            path_fraction = rng.uniform(
                OBSTACLE_PATH_FRACTION_MIN,
                OBSTACLE_PATH_FRACTION_MAX,
            )

            obstacle_x, obstacle_y = (
                obstacle_from_path(
                    START_X,
                    start_y,
                    heading_deg,
                    target_angle,
                    target_distance,
                    path_fraction,
                    lateral_offset,
                )
            )

            expect_obstacle = True

        else:
            raise RuntimeError(
                f"Unknown category: {category}"
            )

        scenarios.append(
            {
                "trial":
                    index,
                "trial_id":
                    f"trial{index:02d}",
                "category":
                    category,
                "start_x":
                    START_X,
                "start_y":
                    start_y,
                "start_heading_deg":
                    heading_deg,
                "target_distance_m":
                    target_distance,
                "target_angle_deg":
                    target_angle,
                "obstacle_x":
                    obstacle_x,
                "obstacle_y":
                    obstacle_y,
                "obstacle_path_fraction":
                    path_fraction,
                "obstacle_lateral_offset_m":
                    lateral_offset,
                "expect_obstacle":
                    expect_obstacle,
            }
        )

    return scenarios


def mode_counter_text(
    counter,
):
    if not counter:
        return ""

    return ";".join(
        f"{key}:{counter[key]}"
        for key in sorted(
            counter
        )
    )


# ============================================================
# Main randomized validator
# ============================================================

def main():
    robot = Supervisor()

    self_node = robot.getSelf()

    if self_node is None:
        raise RuntimeError(
            "Step 58 requires supervisor TRUE."
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

    left_motor.setVelocity(
        0.0
    )

    right_motor.setVelocity(
        0.0
    )

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

    bridge = (
        base.load_visual_bridge()
    )

    target_node, target_created = (
        base.get_or_create_target(
            robot
        )
    )

    obstacle_node = (
        base.find_top_level_node_by_name(
            robot,
            "test obstacle",
        )
    )

    if obstacle_node is None:
        raise RuntimeError(
            "Could not find top-level "
            "'test obstacle'."
        )

    obstacle_translation = (
        obstacle_node.getField(
            "translation"
        )
    )

    if obstacle_translation is None:
        raise RuntimeError(
            "test obstacle has no translation field."
        )

    obstacle_original = list(
        obstacle_translation.getSFVec3f()
    )

    output_dir = (
        PROJECT_ROOT
        / "connectome"
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios_path = (
        output_dir
        / "59a_v2_failure_replay_scenarios.csv"
    )

    detail_path = (
        output_dir
        / "59a_v2_failure_replay_detail.csv"
    )

    results_path = (
        output_dir
        / "59a_v2_failure_replay_results.csv"
    )

    scenarios = (
        generate_scenarios()
    )

    # --------------------------------------------------------
    # Replay ONLY the 8 genuine navigation failures from
    # Step 58. We intentionally exclude the six
    # obstacle_requirement_failed cases because those trials
    # reached the target and the obstacle simply never entered
    # the configured proximity activation envelope.
    #
    # Each geometry is repeated twice with a fresh FlyNavBrain
    # instance so we can distinguish deterministic geometry/
    # arbitration problems from neural stochastic variability.
    # --------------------------------------------------------
    source_failure_trials = [
        3, 6, 9, 12, 16, 19, 25, 26
    ]

    source_map = {
        int(s["trial"]): s
        for s in scenarios
    }

    replay_scenarios = []
    replay_index = 1

    for source_trial in source_failure_trials:
        source = source_map[
            source_trial
        ]

        for repeat in (1, 2):
            item = dict(
                source
            )

            item[
                "source_trial"
            ] = source_trial

            item[
                "repeat"
            ] = repeat

            item[
                "trial"
            ] = replay_index

            item[
                "trial_id"
            ] = (
                f"trial{source_trial:02d}"
                f"_r{repeat}"
            )

            replay_scenarios.append(
                item
            )

            replay_index += 1

    scenarios = replay_scenarios

    scenario_fields = list(
        scenarios[0].keys()
    )

    with scenarios_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=scenario_fields,
        )

        writer.writeheader()
        writer.writerows(
            scenarios
        )

    detail_fields = [
        "trial",
        "trial_id",
        "category",
        "step",
        "time_s",
        "mode",
        "x",
        "y",
        "target_visible",
        "red_fraction",
        "centroid_x_norm",
        "last_seen_centroid",
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

    results = []

    print()
    print("=" * 126)
    print(
        "STEP 59A - V2 FAILED-GEOMETRY REPLAY"
    )
    print("=" * 126)
    print()
    print(
        f"Scenario seed: {SEED}"
    )
    print(
        f"Replay runs: {len(scenarios)} (8 failed geometries x 2 repeats)"
    )
    print(
        "Controlled architecture under test: Step 59B V2"
    )
    print(
        "No neural gains, thresholds, connectome pathways, arrival parameters, or recovery parameters are retuned here."
    )
    print(
        "Scenario randomization only; neural dynamics may still "
        "retain their normal stochastic variability."
    )
    print()

    try:
        with detail_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as detail_handle:
            detail_writer = (
                csv.DictWriter(
                    detail_handle,
                    fieldnames=
                        detail_fields,
                )
            )

            detail_writer.writeheader()

            for scenario in scenarios:
                trial = int(
                    scenario["trial"]
                )

                trial_id = scenario[
                    "trial_id"
                ]

                category = scenario[
                    "category"
                ]

                expect_obstacle = bool(
                    scenario[
                        "expect_obstacle"
                    ]
                )

                left_motor.setVelocity(
                    0.0
                )

                right_motor.setVelocity(
                    0.0
                )

                reset_rover_pose(
                    self_node,
                    scenario[
                        "start_x"
                    ],
                    scenario[
                        "start_y"
                    ],
                    scenario[
                        "start_heading_deg"
                    ],
                )

                base.move_node_xy(
                    obstacle_node,
                    scenario[
                        "obstacle_x"
                    ],
                    scenario[
                        "obstacle_y"
                    ],
                )

                for _ in range(
                    SETTLE_STEPS
                ):
                    if robot.step(
                        TIME_STEP
                    ) == -1:
                        return

                start_position = list(
                    self_node.getPosition()
                )

                tx, ty, tz = (
                    place_target(
                        target_node,
                        float(
                            start_position[0]
                        ),
                        float(
                            start_position[1]
                        ),
                        float(
                            start_position[2]
                        ),
                        scenario[
                            "start_heading_deg"
                        ],
                        scenario[
                            "target_angle_deg"
                        ],
                        scenario[
                            "target_distance_m"
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
                    base.decode_red_target(
                        camera
                    )
                )

                initial_sensor_values = [
                    float(
                        sensor.getValue()
                    )
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

                print("-" * 126)
                print(
                    f"{trial_id} "
                    f"{category:<14} | "
                    f"start=("
                    f"{scenario['start_x']:+.3f},"
                    f"{scenario['start_y']:+.3f}) "
                    f"h={scenario['start_heading_deg']:+.1f}° | "
                    f"target_d="
                    f"{scenario['target_distance_m']:.3f} "
                    f"a={scenario['target_angle_deg']:+.1f}° | "
                    f"obs=("
                    f"{scenario['obstacle_x']:+.3f},"
                    f"{scenario['obstacle_y']:+.3f}) | "
                    f"vis={initial_visual['visible']} "
                    f"red={initial_visual['red_fraction']:.3f} "
                    f"cent={initial_centroid_text} "
                    f"prox0={initial_max_prox:.1f}"
                )

                # ------------------------------------------------
                # Invalid fixture check.
                # ------------------------------------------------

                if not initial_visual[
                    "visible"
                ]:
                    results.append(
                        {
                            **scenario,
                            "success":
                                False,
                            "failure_reason":
                                "scene_invalid_target_not_visible",
                            "target_reached":
                                False,
                            "obstacle_activated":
                                False,
                            "obstacle_requirement":
                                not expect_obstacle,
                            "recovery_requirement":
                                False,
                            "elapsed_s":
                                0.0,
                            "path_length_m":
                                0.0,
                            "net_displacement_m":
                                0.0,
                            "target_progress_m":
                                0.0,
                            "path_efficiency":
                                0.0,
                            "final_target_distance_m":
                                scenario[
                                    "target_distance_m"
                                ],
                            "final_red_fraction":
                                0.0,
                            "final_centroid":
                                float(
                                    "nan"
                                ),
                            "lost_frames":
                                0,
                            "reacquired_count":
                                0,
                            "max_proximity_raw":
                                initial_max_prox,
                            "max_blend":
                                0.0,
                            "max_abs_neural_steering":
                                0.0,
                            "dna_left_max":
                                0,
                            "dna_right_max":
                                0,
                            "mode_counts":
                                "",
                        }
                    )

                    print(
                        "  RESULT FAIL | "
                        "scene invalid: target not visible initially"
                    )

                    continue

                # Fresh neural obstacle network for every trial.
                brain = base.FlyNavBrain(
                    PROJECT_ROOT
                )

                visual_smoothed = 0.0
                obstacle_smoothed = 0.0
                obstacle_latched = False

                previous_left = 0.0
                previous_right = 0.0

                arrival_hold = 0
                lost_frames = 0
                lost_streak = 0
                reacquired_count = 0
                reacquire_steps = 0

                initial_c = float(
                    initial_visual[
                        "centroid_x_norm"
                    ]
                )

                last_seen_centroid = (
                    initial_c
                    if not math.isnan(
                        initial_c
                    )
                    else 0.0
                )

                was_visible = bool(
                    initial_visual[
                        "visible"
                    ]
                )

                obstacle_activated = (
                    False
                )

                max_prox_seen = (
                    initial_max_prox
                )

                max_blend_seen = 0.0
                max_neural_seen = 0.0

                max_dna_left = 0
                max_dna_right = 0

                mode_counts = Counter()

                passed_target = False
                failure_reason = (
                    "timeout"
                )

                final_red = 0.0
                final_centroid = float(
                    "nan"
                )

                previous_position = list(
                    self_node.getPosition()
                )

                path_length = 0.0

                initial_target_distance = (
                    euclidean_xy(
                        previous_position,
                        [tx, ty],
                    )
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

                    current_position = list(
                        self_node.getPosition()
                    )

                    path_length += (
                        euclidean_xy(
                            previous_position,
                            current_position,
                        )
                    )

                    previous_position = (
                        current_position
                    )

                    visual = (
                        base.decode_red_target(
                            camera
                        )
                    )

                    visible = bool(
                        visual[
                            "visible"
                        ]
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

                        if (
                            abs(
                                final_centroid
                            )
                            >= base.REACQUIRE_MIN_ABS_CENTROID
                        ):
                            last_seen_centroid = (
                                final_centroid
                            )

                    else:
                        lost_frames += 1
                        lost_streak += 1

                    was_visible = visible

                    visual_output = (
                        base.visual_neural_steering(
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
                            - base.VISUAL_SMOOTH_ALPHA
                        )
                        * visual_smoothed
                        +
                        base.VISUAL_SMOOTH_ALPHA
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
                        base.encode_proximity(
                            front_left
                        )
                    )

                    obstacle_right = (
                        base.encode_proximity(
                            front_right
                        )
                    )

                    neural = brain.step(
                        obstacle_left=
                            obstacle_left,
                        obstacle_right=
                            obstacle_right,
                        duration_ms=
                            TIME_STEP,
                    )

                    obstacle_raw = float(
                        neural[
                            "steering"
                        ]
                    )

                    obstacle_smoothed = (
                        (
                            1.0
                            - base.OBSTACLE_SMOOTH_ALPHA
                        )
                        * obstacle_smoothed
                        +
                        base.OBSTACLE_SMOOTH_ALPHA
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
                            >= base.OBSTACLE_ENTER_RAW
                        ):
                            obstacle_latched = True
                            obstacle_activated = True
                    else:
                        if (
                            max_raw
                            <= base.OBSTACLE_EXIT_RAW
                        ):
                            obstacle_latched = False

                    if obstacle_latched:
                        blend = max(
                            0.15,
                            base.obstacle_blend_weight(
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
                            - base.OBSTACLE_SPEED_REDUCTION
                            * (
                                obstacle_left
                                + obstacle_right
                            )
                            / 2.0
                        ),
                        base.OBSTACLE_MIN_SPEED_SCALE,
                        1.0,
                    )

                    obstacle_forward = (
                        base.OBSTACLE_BASE_SPEED_FRACTION
                        * max_speed
                        * obstacle_speed_scale
                    )

                    obstacle_turn = (
                        obstacle_smoothed
                        * base.OBSTACLE_STEERING_GAIN
                        * max_speed
                    )

                    if visible:
                        centroid = float(
                            visual[
                                "centroid_x_norm"
                            ]
                        )

                        visual_scale = (
                            base.visual_forward_scale(
                                visual[
                                    "red_fraction"
                                ],
                                centroid,
                            )
                        )

                        visual_forward = (
                            base.VISUAL_FORWARD_SPEED_FRACTION
                            * max_speed
                            * visual_scale
                        )

                        visual_turn = (
                            visual_smoothed
                            * base.VISUAL_TURN_GAIN
                            * max_speed
                        )

                        strict_arrived = (
                            visual[
                                "red_fraction"
                            ]
                            >= base.STOP_RED_FRACTION
                            and abs(
                                centroid
                            )
                            <= base.CENTER_TOLERANCE
                        )

                        close_safe_arrived = (
                            visual[
                                "red_fraction"
                            ]
                            >= base.STOP_RED_FRACTION
                            and abs(
                                centroid
                            )
                            <= base.CLOSE_ARRIVAL_CENTROID_TOLERANCE
                            and blend
                            <= base.CLOSE_ARRIVAL_MAX_BLEND
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
                            final_forward = min(
                                base.LOST_TARGET_AVOID_FORWARD_FRACTION
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
                            max_reacquire_steps = int(
                                base.REACQUIRE_MAX_TIME_S
                                * 1000.0
                                / TIME_STEP
                            )

                            if (
                                abs(
                                    last_seen_centroid
                                )
                                >= base.REACQUIRE_MIN_ABS_CENTROID
                                and reacquire_steps
                                < max_reacquire_steps
                            ):
                                direction = (
                                    base.reacquire_turn_direction(
                                        last_seen_centroid,
                                        reacquire_steps,
                                    )
                                )

                                reacquire_steps += 1

                                final_forward = 0.0

                                final_turn = (
                                    direction
                                    * base.REACQUIRE_TURN_SPEED_FRACTION
                                    * max_speed
                                )

                                primary_steps = int(
                                    base.REACQUIRE_PRIMARY_TIME_S
                                    * 1000.0
                                    / TIME_STEP
                                )

                                if (
                                    reacquire_steps
                                    <= primary_steps
                                ):
                                    mode = (
                                        "REACQUIRE_PRIMARY"
                                    )
                                else:
                                    mode = (
                                        "REACQUIRE_REVERSE"
                                    )

                            else:
                                final_forward = 0.0
                                final_turn = 0.0

                                mode = (
                                    "LOST_STOP"
                                )

                    mode_counts[
                        mode
                    ] += 1

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
                            - base.MOTOR_SMOOTH_ALPHA
                        )
                        * previous_left
                        +
                        base.MOTOR_SMOOTH_ALPHA
                        * target_left
                    )

                    right_velocity = (
                        (
                            1.0
                            - base.MOTOR_SMOOTH_ALPHA
                        )
                        * previous_right
                        +
                        base.MOTOR_SMOOTH_ALPHA
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

                    detail_writer.writerow(
                        {
                            "trial":
                                trial,
                            "trial_id":
                                trial_id,
                            "category":
                                category,
                            "step":
                                step_count,
                            "time_s":
                                f"{step_count * TIME_STEP / 1000.0:.4f}",
                            "mode":
                                mode,
                            "x":
                                f"{float(current_position[0]):.6f}",
                            "y":
                                f"{float(current_position[1]):.6f}",
                            "target_visible":
                                int(
                                    visible
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
                            "last_seen_centroid":
                                f"{last_seen_centroid:.6f}",
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
                            f"{mode:<10} "
                            f"vis={visible} "
                            f"red={visual['red_fraction']:.3f} "
                            f"cent={centroid_text:>7} "
                            f"prox={max_raw:7.1f} "
                            f"nRAW={obstacle_raw:+.3f} "
                            f"DNa=({dna_left},{dna_right}) "
                            f"blend={blend:.2f} "
                            f"rq={reacquire_steps:3d}"
                        )

                    if (
                        arrival_hold
                        >= base.CENTER_HOLD_FRAMES
                    ):
                        passed_target = True
                        failure_reason = ""
                        break

                left_motor.setVelocity(
                    0.0
                )

                right_motor.setVelocity(
                    0.0
                )

                elapsed_s = (
                    step_count
                    * TIME_STEP
                    / 1000.0
                )

                final_position = list(
                    self_node.getPosition()
                )

                final_target_distance = (
                    euclidean_xy(
                        final_position,
                        [tx, ty],
                    )
                )

                net_displacement = (
                    euclidean_xy(
                        start_position,
                        final_position,
                    )
                )

                target_progress = max(
                    0.0,
                    initial_target_distance
                    - final_target_distance,
                )

                path_efficiency = (
                    target_progress
                    / path_length
                    if path_length > 1e-9
                    else 0.0
                )

                path_efficiency = clamp(
                    path_efficiency,
                    0.0,
                    1.0,
                )

                obstacle_requirement = (
                    obstacle_activated
                    if expect_obstacle
                    else not obstacle_activated
                )

                recovery_requirement = (
                    True
                    if lost_frames == 0
                    else reacquired_count > 0
                )

                success = (
                    passed_target
                    and obstacle_requirement
                    and recovery_requirement
                )

                if not success:
                    if (
                        passed_target
                        and not obstacle_requirement
                    ):
                        failure_reason = (
                            "obstacle_requirement_failed"
                        )

                    elif (
                        passed_target
                        and not recovery_requirement
                    ):
                        failure_reason = (
                            "recovery_requirement_failed"
                        )

                    elif not passed_target:
                        if (
                            mode_counts[
                                "LOST_STOP"
                            ]
                            > 0
                        ):
                            failure_reason = (
                                "timeout_after_lost_stop"
                            )
                        elif (
                            lost_frames > 0
                        ):
                            failure_reason = (
                                "timeout_after_visual_loss"
                            )
                        else:
                            failure_reason = (
                                "timeout_target_not_reached"
                            )

                result = {
                    **scenario,
                    "success":
                        success,
                    "failure_reason":
                        failure_reason,
                    "target_reached":
                        passed_target,
                    "obstacle_activated":
                        obstacle_activated,
                    "obstacle_requirement":
                        obstacle_requirement,
                    "recovery_requirement":
                        recovery_requirement,
                    "elapsed_s":
                        elapsed_s,
                    "path_length_m":
                        path_length,
                    "net_displacement_m":
                        net_displacement,
                    "target_progress_m":
                        target_progress,
                    "path_efficiency":
                        path_efficiency,
                    "final_target_distance_m":
                        final_target_distance,
                    "final_red_fraction":
                        final_red,
                    "final_centroid":
                        final_centroid,
                    "lost_frames":
                        lost_frames,
                    "reacquired_count":
                        reacquired_count,
                    "max_proximity_raw":
                        max_prox_seen,
                    "max_blend":
                        max_blend_seen,
                    "max_abs_neural_steering":
                        max_neural_seen,
                    "dna_left_max":
                        max_dna_left,
                    "dna_right_max":
                        max_dna_right,
                    "mode_counts":
                        mode_counter_text(
                            mode_counts
                        ),
                }

                results.append(
                    result
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
                    f"{'PASS' if success else 'FAIL'} | "
                    f"target={passed_target} "
                    f"obs={obstacle_activated} "
                    f"recover={recovery_requirement} | "
                    f"time={elapsed_s:.2f}s | "
                    f"eff={path_efficiency:.3f} | "
                    f"red={final_red:.3f} "
                    f"cent={centroid_text} | "
                    f"prox={max_prox_seen:.1f} "
                    f"blend={max_blend_seen:.2f} "
                    f"nmax={max_neural_seen:.3f} | "
                    f"lost={lost_frames} "
                    f"reacq={reacquired_count}"
                )

                detail_handle.flush()

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

    # --------------------------------------------------------
    # Write per-trial results.
    # --------------------------------------------------------

    if not results:
        raise RuntimeError(
            "No Step 58 results were produced."
        )

    result_fields = list(
        results[0].keys()
    )

    with results_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=result_fields,
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------

    success_rows = [
        row
        for row in results
        if row["success"]
    ]

    success_count = len(
        success_rows
    )

    success_rate = (
        success_count
        / len(results)
        * 100.0
    )

    success_times = [
        float(
            row["elapsed_s"]
        )
        for row in success_rows
    ]

    success_efficiencies = [
        float(
            row[
                "path_efficiency"
            ]
        )
        for row in success_rows
    ]

    print()
    print("=" * 126)
    print(
        "STEP 59A FINAL SUMMARY"
    )
    print("=" * 126)

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Overall: {success_count}/{len(results)} "
        f"({success_rate:.1f}%)"
    )

    if success_times:
        print(
            f"Successful time mean="
            f"{statistics.mean(success_times):.2f}s | "
            f"median="
            f"{statistics.median(success_times):.2f}s | "
            f"max="
            f"{max(success_times):.2f}s"
        )

    if success_efficiencies:
        print(
            f"Path efficiency mean="
            f"{statistics.mean(success_efficiencies):.3f} | "
            f"median="
            f"{statistics.median(success_efficiencies):.3f}"
        )

    print()

    by_category = defaultdict(
        list
    )

    for row in results:
        by_category[
            row["category"]
        ].append(
            row
        )

    for category in [
        "clear",
        "left_conflict",
        "right_conflict",
        "near_center",
    ]:
        rows = by_category[
            category
        ]

        passed = sum(
            1
            for row in rows
            if row["success"]
        )

        expected_obs_rows = [
            row
            for row in rows
            if row[
                "expect_obstacle"
            ]
        ]

        obs_activated = sum(
            1
            for row in expected_obs_rows
            if row[
                "obstacle_activated"
            ]
        )

        lost_rows = [
            row
            for row in rows
            if row[
                "lost_frames"
            ]
            > 0
        ]

        recovered = sum(
            1
            for row in lost_rows
            if row[
                "reacquired_count"
            ]
            > 0
        )

        print(
            f"{category:<15} "
            f"{passed}/{len(rows)} PASS | "
            f"obs="
            f"{obs_activated}/"
            f"{len(expected_obs_rows)} | "
            f"lost="
            f"{len(lost_rows)} | "
            f"recovered="
            f"{recovered}/"
            f"{len(lost_rows)}"
        )

    expected_obstacle_rows = [
        row
        for row in results
        if row[
            "expect_obstacle"
        ]
    ]

    expected_obstacle_active = sum(
        1
        for row in expected_obstacle_rows
        if row[
            "obstacle_activated"
        ]
    )

    visual_loss_rows = [
        row
        for row in results
        if row[
            "lost_frames"
        ]
        > 0
    ]

    visual_loss_recovered = sum(
        1
        for row in visual_loss_rows
        if row[
            "reacquired_count"
        ]
        > 0
    )

    print()
    print(
        "Expected-obstacle activation: "
        f"{expected_obstacle_active}/"
        f"{len(expected_obstacle_rows)}"
    )

    print(
        "Visual-loss recovery: "
        f"{visual_loss_recovered}/"
        f"{len(visual_loss_rows)}"
    )

    failures = [
        row
        for row in results
        if not row[
            "success"
        ]
    ]

    if failures:
        print()
        print("FAILED TRIALS:")

        for row in failures:
            print(
                f"  {row['trial_id']} "
                f"{row['category']} | "
                f"reason="
                f"{row['failure_reason']} | "
                f"target="
                f"{row['target_reached']} | "
                f"obs="
                f"{row['obstacle_activated']} | "
                f"lost="
                f"{row['lost_frames']} | "
                f"reacq="
                f"{row['reacquired_count']} | "
                f"prox="
                f"{float(row['max_proximity_raw']):.1f} | "
                f"nmax="
                f"{float(row['max_abs_neural_steering']):.3f}"
            )

    print()
    print("FILES:")
    print(
        scenarios_path
    )
    print(
        detail_path
    )
    print(
        results_path
    )

    print()

    print()
    print(
        "STEP 59A REPLAY COMPLETE."
    )
    print(
        "This replay directly tests whether Step 59B V2 fixes "
        "the 8 deterministic Step 58/58A failure geometries."
    )
    print(
        "Primary success target: substantially improve on the "
        "Step 58A baseline of 0/16 without regressing neural "
        "obstacle activation."
    )


if __name__ == "__main__":
    main()
