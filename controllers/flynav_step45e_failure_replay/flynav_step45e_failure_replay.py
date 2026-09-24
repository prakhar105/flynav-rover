import csv
import importlib.util
import math
from collections import deque
from pathlib import Path

from controller import Supervisor

TIME_STEP = 32
TIMEOUT_S = 45.0
SETTLE_STEPS = 8
REPEATS = 2
GOAL_TOLERANCE_M = 0.035
OBSTACLE_NAME = "test obstacle"

CASES = [
    {
        "name": "trial03_direct",
        "source_trial": 3,
        "start": (0.31, 0.34),
        "goal": (0.36, -0.33),
        "obstacle": (0.32, -0.05),
        "heading_deg": -38.7,
    },
    {
        "name": "trial14_near_wall",
        "source_trial": 14,
        "start": (-0.31, 0.32),
        "goal": (-0.31, -0.32),
        "obstacle": (-0.31, 0.02),
        "heading_deg": -40.2,
    },
]


def load_neural(project_root):
    path = (
        project_root
        / "controllers"
        / "flynav_combined_neural_v2"
        / "flynav_combined_neural_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "flynav_combined_neural_v2_impl", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_obstacle(robot):
    node = robot.getFromDef("TEST_OBSTACLE")
    if node is not None:
        return node

    children = robot.getRoot().getField("children")
    for i in range(children.getCount()):
        candidate = children.getMFNode(i)
        name_field = candidate.getField("name")
        if name_field is None:
            continue
        try:
            if name_field.getSFString() == OBSTACLE_NAME:
                return candidate
        except Exception:
            pass
    return None


def sign(v, eps=0.04):
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def collapse(seq):
    out = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return "->".join(out)


def main():
    robot = Supervisor()
    root = Path(__file__).resolve().parents[2]
    neural = load_neural(root)
    base = neural.base

    self_node = robot.getSelf()
    tfield = self_node.getField("translation")
    rfield = self_node.getField("rotation")

    original_t = list(tfield.getSFVec3f())
    original_r = list(rfield.getSFRotation())
    robot_z = float(original_t[2])

    obstacle_node = find_obstacle(robot)
    if obstacle_node is None:
        raise RuntimeError('Could not find Solid named "test obstacle".')

    ofield = obstacle_node.getField("translation")
    original_o = list(ofield.getSFVec3f())
    obstacle_z = float(original_o[2])

    lm = robot.getDevice("left wheel motor")
    rm = robot.getDevice("right wheel motor")
    lm.setPosition(float("inf"))
    rm.setPosition(float("inf"))
    lm.setVelocity(0.0)
    rm.setVelocity(0.0)
    max_speed = min(lm.getMaxVelocity(), rm.getMaxVelocity())

    sensors = {}
    for i in range(8):
        name = f"ps{i}"
        s = robot.getDevice(name)
        s.enable(TIME_STEP)
        sensors[name] = s

    matrix = base.load_frozen_matrix()

    out_dir = root / "connectome" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "step45e_failure_replay_detail.csv"
    summary_csv = out_dir / "step45e_failure_replay_summary.csv"

    detail_fields = [
        "case","repeat","step","time_s","x","y","goal_distance_m",
        "heading_deg","bearing_error_deg",
        "neural_fl_raw","neural_fr_raw",
        "dna_left_spikes","dna_right_spikes",
        "raw_neural","smooth_neural","neural_blend","mode",
        "shadow_raw","shadow_smooth","shadow_blend",
        "neural_shadow_sign_disagree",
        "goal_steering","final_turn","left_wheel","right_wheel"
    ]

    summaries = []

    print("\n" + "=" * 110)
    print("STEP 45E - FAILURE REPLAY / NEURAL VS ARITHMETIC SHADOW")
    print("=" * 110)
    print("Rover control = Combined Neural v2")
    print("Arithmetic Combined v1 = shadow diagnostic only\n")

    try:
        with detail_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=detail_fields)
            writer.writeheader()

            for case in CASES:
                for repeat in range(1, REPEATS + 1):
                    sx, sy = case["start"]
                    gx, gy = case["goal"]
                    ox, oy = case["obstacle"]

                    lm.setVelocity(0.0)
                    rm.setVelocity(0.0)

                    tfield.setSFVec3f([sx, sy, robot_z])
                    rfield.setSFRotation(
                        [0.0, 0.0, 1.0, math.radians(case["heading_deg"])]
                    )
                    self_node.resetPhysics()

                    ofield.setSFVec3f([ox, oy, obstacle_z])
                    obstacle_node.resetPhysics()

                    for _ in range(SETTLE_STEPS):
                        if robot.step(TIME_STEP) == -1:
                            return

                    brain = neural.FlyNavBrain(root)

                    prev_l = prev_r = 0.0
                    smooth_n = 0.0
                    smooth_shadow = 0.0
                    neural_latched = False
                    shadow_latched = False

                    last_n_sign = 0
                    last_s_sign = 0
                    n_flips = 0
                    s_flips = 0
                    sign_disagree = 0
                    bilateral = 0
                    equal_bilateral = 0

                    blend_steps = 0
                    avoid_steps = 0
                    modes = []
                    recent = deque()

                    elapsed = 0.0
                    step = 0
                    success = False
                    max_prox = 0.0
                    max_n = 0.0
                    max_shadow = 0.0

                    print(f"{case['name']} repeat {repeat}/{REPEATS}")

                    while elapsed < TIMEOUT_S:
                        if robot.step(TIME_STEP) == -1:
                            return

                        elapsed += TIME_STEP / 1000.0
                        step += 1

                        pos = self_node.getPosition()
                        x, y = float(pos[0]), float(pos[1])
                        dx, dy = gx - x, gy - y
                        dist = math.hypot(dx, dy)

                        recent.append((elapsed, x, y, dist))
                        cutoff = elapsed - 5.0
                        while recent and recent[0][0] < cutoff:
                            recent.popleft()

                        if dist <= GOAL_TOLERANCE_M:
                            success = True
                            lm.setVelocity(0.0)
                            rm.setVelocity(0.0)
                            break

                        heading = base.get_robot_heading(self_node)
                        goal_bearing = math.atan2(dy, dx)
                        berr = base.wrap_pi(goal_bearing - heading)

                        if berr > neural.NEUTRAL_DEADBAND_RAD:
                            desired = -neural.ACTION_STEERING
                        elif berr < -neural.NEUTRAL_DEADBAND_RAD:
                            desired = neural.ACTION_STEERING
                        else:
                            desired = 0.0

                        hidx, _ = base.quantize_heading(heading)
                        _, _, goal_steer, _ = base.choose_connectome_goal_action(
                            matrix, hidx, desired
                        )

                        ae = abs(berr)
                        if ae > math.radians(90):
                            goal_forward = neural.GOAL_HARD_TURN_SPEED_FRACTION * max_speed
                        elif ae > math.radians(35):
                            goal_forward = neural.GOAL_TURN_SPEED_FRACTION * max_speed
                        else:
                            goal_forward = neural.GOAL_BASE_SPEED_FRACTION * max_speed

                        goal_turn = goal_steer * neural.GOAL_STEERING_GAIN * max_speed

                        obs = neural.read_neural_obstacle_state(sensors, brain)
                        raw_values = obs["raw_values"]
                        nmaxraw = obs["max_raw"]
                        max_prox = max(max_prox, nmaxraw)

                        raw_n = obs["raw_neural_steering"]
                        max_n = max(max_n, abs(raw_n))

                        if obs["left_spikes"] > 0 and obs["right_spikes"] > 0:
                            bilateral += 1
                            if obs["left_spikes"] == obs["right_spikes"]:
                                equal_bilateral += 1

                        ns = sign(raw_n)
                        if ns and last_n_sign and ns != last_n_sign:
                            n_flips += 1
                        if ns:
                            last_n_sign = ns

                        if not neural_latched and nmaxraw >= neural.OBSTACLE_ENTER_RAW:
                            neural_latched = True
                        elif neural_latched and nmaxraw <= neural.OBSTACLE_EXIT_RAW:
                            neural_latched = False

                        smooth_n = (
                            (1.0 - neural.OBSTACLE_SMOOTH_ALPHA) * smooth_n
                            + neural.OBSTACLE_SMOOTH_ALPHA * raw_n
                        )

                        obs_scale = neural.clamp(
                            1.0 - neural.OBSTACLE_SPEED_REDUCTION * obs["average_obstacle"],
                            neural.OBSTACLE_MIN_SPEED_SCALE,
                            1.0,
                        )
                        obs_forward = (
                            neural.OBSTACLE_BASE_SPEED_FRACTION * max_speed * obs_scale
                        )
                        obs_turn = smooth_n * neural.OBSTACLE_STEERING_GAIN * max_speed

                        if neural_latched:
                            nblend = max(
                                base.obstacle_blend_weight(nmaxraw), 0.15
                            )
                        else:
                            nblend = 0.0

                        if nblend >= 0.75:
                            mode = "AVOID"
                            avoid_steps += 1
                        elif nblend > 0:
                            mode = "BLEND"
                            blend_steps += 1
                        else:
                            mode = "GOAL"
                        modes.append(mode)

                        # Old arithmetic shadow, computed from same sensor snapshot.
                        shadow_left = max(
                            base.normalize_sensor(raw_values[n])
                            for n in base.LEFT_SENSOR_NAMES
                        )
                        shadow_right = max(
                            base.normalize_sensor(raw_values[n])
                            for n in base.RIGHT_SENSOR_NAMES
                        )
                        shadow_raw = base.clamp(
                            shadow_left - shadow_right, -1.0, 1.0
                        )
                        max_shadow = max(max_shadow, abs(shadow_raw))

                        ss = sign(shadow_raw)
                        if ss and last_s_sign and ss != last_s_sign:
                            s_flips += 1
                        if ss:
                            last_s_sign = ss

                        if ns and ss and ns != ss:
                            sign_disagree += 1

                        smooth_shadow = (
                            (1.0 - base.OBSTACLE_SMOOTH_ALPHA) * smooth_shadow
                            + base.OBSTACLE_SMOOTH_ALPHA * shadow_raw
                        )

                        shadow_maxraw = max(
                            max(raw_values[n] for n in base.LEFT_SENSOR_NAMES),
                            max(raw_values[n] for n in base.RIGHT_SENSOR_NAMES),
                            max(raw_values[n] for n in base.FRONT_SENSOR_NAMES),
                        )

                        if not shadow_latched and shadow_maxraw >= base.OBSTACLE_ENTER_RAW:
                            shadow_latched = True
                        elif shadow_latched and shadow_maxraw <= base.OBSTACLE_EXIT_RAW:
                            shadow_latched = False

                        if shadow_latched:
                            sblend = max(
                                base.obstacle_blend_weight(shadow_maxraw), 0.15
                            )
                        else:
                            sblend = 0.0

                        final_turn = (
                            (1.0 - nblend) * goal_turn
                            + nblend * obs_turn
                        )
                        final_forward = min(goal_forward, obs_forward)

                        target_l = neural.clamp(
                            final_forward + final_turn, -max_speed, max_speed
                        )
                        target_r = neural.clamp(
                            final_forward - final_turn, -max_speed, max_speed
                        )

                        vl = (
                            (1.0 - neural.MOTOR_SMOOTH_ALPHA) * prev_l
                            + neural.MOTOR_SMOOTH_ALPHA * target_l
                        )
                        vr = (
                            (1.0 - neural.MOTOR_SMOOTH_ALPHA) * prev_r
                            + neural.MOTOR_SMOOTH_ALPHA * target_r
                        )

                        lm.setVelocity(vl)
                        rm.setVelocity(vr)
                        prev_l, prev_r = vl, vr

                        writer.writerow({
                            "case": case["name"],
                            "repeat": repeat,
                            "step": step,
                            "time_s": f"{elapsed:.4f}",
                            "x": f"{x:.6f}",
                            "y": f"{y:.6f}",
                            "goal_distance_m": f"{dist:.6f}",
                            "heading_deg": f"{math.degrees(heading):.4f}",
                            "bearing_error_deg": f"{math.degrees(berr):.4f}",
                            "neural_fl_raw": f"{obs['front_left_raw']:.3f}",
                            "neural_fr_raw": f"{obs['front_right_raw']:.3f}",
                            "dna_left_spikes": obs["left_spikes"],
                            "dna_right_spikes": obs["right_spikes"],
                            "raw_neural": f"{raw_n:.6f}",
                            "smooth_neural": f"{smooth_n:.6f}",
                            "neural_blend": f"{nblend:.6f}",
                            "mode": mode,
                            "shadow_raw": f"{shadow_raw:.6f}",
                            "shadow_smooth": f"{smooth_shadow:.6f}",
                            "shadow_blend": f"{sblend:.6f}",
                            "neural_shadow_sign_disagree": int(ns and ss and ns != ss),
                            "goal_steering": f"{goal_steer:.6f}",
                            "final_turn": f"{final_turn:.6f}",
                            "left_wheel": f"{vl:.6f}",
                            "right_wheel": f"{vr:.6f}",
                        })

                    final_pos = self_node.getPosition()
                    fx, fy = float(final_pos[0]), float(final_pos[1])
                    final_dist = math.hypot(gx - fx, gy - fy)

                    if len(recent) >= 2:
                        _, x0, y0, d0 = recent[0]
                        last5_move = math.hypot(fx - x0, fy - y0)
                        last5_progress = d0 - final_dist
                    else:
                        last5_move = 0.0
                        last5_progress = 0.0

                    row = {
                        "case": case["name"],
                        "source_trial": case["source_trial"],
                        "repeat": repeat,
                        "goal_reached": success,
                        "elapsed_s": round(elapsed, 4),
                        "final_distance_m": round(final_dist, 6),
                        "last_5s_displacement_m": round(last5_move, 6),
                        "last_5s_goal_progress_m": round(last5_progress, 6),
                        "max_proximity": round(max_prox, 3),
                        "max_abs_neural_steering": round(max_n, 6),
                        "max_abs_shadow_steering": round(max_shadow, 6),
                        "bilateral_dna_steps": bilateral,
                        "equal_nonzero_dna_steps": equal_bilateral,
                        "neural_sign_flip_count": n_flips,
                        "shadow_sign_flip_count": s_flips,
                        "neural_shadow_sign_disagreement_steps": sign_disagree,
                        "neural_blend_steps": blend_steps,
                        "neural_avoid_steps": avoid_steps,
                        "mode_transitions": collapse(modes),
                    }
                    summaries.append(row)

                    print(
                        f"  {'PASS' if success else 'FAIL'} "
                        f"final={final_dist:.3f}m "
                        f"last5move={last5_move:.3f}m "
                        f"last5progress={last5_progress:+.3f}m "
                        f"nmax={max_n:.3f} shadow={max_shadow:.3f} "
                        f"bilateral={bilateral} equalDNA={equal_bilateral} "
                        f"nFlips={n_flips} shadowFlips={s_flips} "
                        f"disagree={sign_disagree}"
                    )

                    fh.flush()

    finally:
        lm.setVelocity(0.0)
        rm.setVelocity(0.0)

        tfield.setSFVec3f(original_t)
        rfield.setSFRotation(original_r)
        self_node.resetPhysics()

        ofield.setSFVec3f(original_o)
        obstacle_node.resetPhysics()

    summary_fields = list(summaries[0].keys())

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 110)
    print("STEP 45E SUMMARY")
    print("=" * 110)

    for row in summaries:
        print(
            f"{row['case']:<22} repeat={row['repeat']} "
            f"goal={str(row['goal_reached']):<5} "
            f"final={row['final_distance_m']:.3f}m "
            f"last5move={row['last_5s_displacement_m']:.3f}m "
            f"last5progress={row['last_5s_goal_progress_m']:+.3f}m "
            f"bilateral={row['bilateral_dna_steps']} "
            f"equalDNA={row['equal_nonzero_dna_steps']} "
            f"nFlips={row['neural_sign_flip_count']} "
            f"shadowFlips={row['shadow_sign_flip_count']} "
            f"disagree={row['neural_shadow_sign_disagreement_steps']}"
        )

    print(f"\nDetail CSV:  {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(
        "\nNEXT: inspect this evidence before changing any "
        "neural or arbitration parameter."
    )


if __name__ == "__main__":
    main()
