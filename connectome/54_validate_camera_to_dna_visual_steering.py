import math
from pathlib import Path

import pandas as pd


# ============================================================
# STEP 54 - CAMERA -> LC10a -> AOTU019 -> DNa02
#           SIGNED VISUAL-STEERING VALIDATION
#
# Purpose:
#   Fuse the REAL Step 49 camera measurements with the REAL
#   Step 53 MaleCNS visual-action bridge before allowing the
#   rover to move.
#
# Biological pathway:
#
#   visual channel L
#       -> LC10a_L (ACh)
#       -> AOTU019_L (GABA)
#       -| DNa02_R
#
#   visual channel R
#       -> LC10a_R (ACh)
#       -> AOTU019_R (GABA)
#       -| DNa02_L
#
# Frozen FlyNav motor convention:
#   DNa02_R dominance -> positive steering -> RIGHT
#   DNa02_L dominance -> negative steering -> LEFT
#
# Therefore:
#   target LEFT  -> suppress right-turn DNa02_R -> LEFT bias
#   target RIGHT -> suppress left-turn DNa02_L  -> RIGHT bias
#
# IMPORTANT SCIENTIFIC BOUNDARY:
#   Camera image LEFT/RIGHT -> LC10a_L/R is an ENGINEERED
#   sensor-interface calibration. It is not claimed to be a
#   literal one-to-one model of fly retinal coordinates.
#
#   LC10a/AOTU019/DNa02 identities, transmitter sign, and
#   anatomical weights come from MaleCNS.
#
# Inputs:
#   49_vision_probe.csv
#   53_lc10a_aotu019_dna_paths.csv
#   53_aotu019_inventory.csv
#
# Outputs:
#   54_camera_to_dna_frame_validation.csv
#   54_camera_to_dna_phase_summary.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "connectome" / "output"

VISION_PATH = OUTPUT_DIR / "49_vision_probe.csv"
PATHS_PATH = OUTPUT_DIR / "53_lc10a_aotu019_dna_paths.csv"
AOTU_PATH = OUTPUT_DIR / "53_aotu019_inventory.csv"

FRAME_OUT = OUTPUT_DIR / "54_camera_to_dna_frame_validation.csv"
SUMMARY_OUT = OUTPUT_DIR / "54_camera_to_dna_phase_summary.csv"


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input is missing:\n{path}"
        )


for path in [
    VISION_PATH,
    PATHS_PATH,
    AOTU_PATH,
]:
    require(path)


vision = pd.read_csv(VISION_PATH)
paths = pd.read_csv(PATHS_PATH)
aotu = pd.read_csv(AOTU_PATH)


print()
print("=" * 118)
print("STEP 54 - CAMERA -> LC10a -> AOTU019 -> DNa02 SIGNED VISUAL-STEERING VALIDATION")
print("=" * 118)
print()


# ------------------------------------------------------------
# 1. Verify AOTU019 inhibitory annotation from Step 53.
# ------------------------------------------------------------

if aotu.empty:
    raise RuntimeError(
        "AOTU019 inventory is empty."
    )

aotu_nt = set(
    aotu["predictedNt"]
    .dropna()
    .astype(str)
    .str.lower()
)

if "gaba" not in aotu_nt:
    raise RuntimeError(
        "AOTU019 is not annotated as GABA in the supplied "
        "MaleCNS Step 53 evidence. Stop before motor mapping."
    )

print(
    "AOTU019 transmitter check: "
    f"PASS ({', '.join(sorted(aotu_nt))})"
)


# ------------------------------------------------------------
# 2. Extract the two contralateral DNa02 structural channels.
#
# LEFT camera channel:
#   LC10a_L -> AOTU019_L -> DNa02_R
#
# RIGHT camera channel:
#   LC10a_R -> AOTU019_R -> DNa02_L
#
# We use summed path bottleneck weights only as a structural
# channel-strength proxy.
# ------------------------------------------------------------

left_to_right_dna = paths[
    (paths["lc10a_side"] == "L")
    & (paths["aotu_side"] == "L")
    & (paths["dna_type"] == "DNa02")
    & (paths["dna_side"] == "R")
].copy()

right_to_left_dna = paths[
    (paths["lc10a_side"] == "R")
    & (paths["aotu_side"] == "R")
    & (paths["dna_type"] == "DNa02")
    & (paths["dna_side"] == "L")
].copy()


if left_to_right_dna.empty:
    raise RuntimeError(
        "Missing LC10a_L -> AOTU019_L -> DNa02_R path."
    )

if right_to_left_dna.empty:
    raise RuntimeError(
        "Missing LC10a_R -> AOTU019_R -> DNa02_L path."
    )


left_channel_score = float(
    left_to_right_dna[
        "bottleneck_weight"
    ].sum()
)

right_channel_score = float(
    right_to_left_dna[
        "bottleneck_weight"
    ].sum()
)

print()
print("MaleCNS structural channel scores:")
print(
    f"  LC10a_L -> AOTU019_L -| DNa02_R: "
    f"{left_channel_score:.1f}"
)
print(
    f"  LC10a_R -> AOTU019_R -| DNa02_L: "
    f"{right_channel_score:.1f}"
)


# ------------------------------------------------------------
# 3. Engineering calibration.
#
# The anatomical left/right totals differ slightly.
# We normalise each sensory channel to its own measured maximum
# so equal visual evidence produces a neutral robot command.
#
# This is robotics/sensory calibration, not a biological gain.
# ------------------------------------------------------------

left_gain = (
    1.0 / left_channel_score
)

right_gain = (
    1.0 / right_channel_score
)

print()
print("Engineered per-channel normalisation:")
print(
    f"  left visual gain  = {left_gain:.8f}"
)
print(
    f"  right visual gain = {right_gain:.8f}"
)
print(
    "  This compensates structural scale only; "
    "it is not a biological synaptic gain."
)


# ------------------------------------------------------------
# 4. Fuse Step 49 image sectors into left/right visual drive.
#
# CENTER evidence is split equally between the two channels.
# This preserves a neutral centered target:
#
#   visual_L = left_fraction  + 0.5 * center_fraction
#   visual_R = right_fraction + 0.5 * center_fraction
#
# The mapping of robot-camera image side to LC10a side is an
# explicitly engineered sensor interface.
# ------------------------------------------------------------

required_cols = [
    "phase",
    "target_visible",
    "left_fraction",
    "center_fraction",
    "right_fraction",
]

missing = [
    c
    for c in required_cols
    if c not in vision.columns
]

if missing:
    raise RuntimeError(
        "Step 49 CSV is missing columns: "
        + ", ".join(missing)
    )


rows = []

for row in vision.itertuples(index=False):
    visible = bool(
        int(row.target_visible)
    )

    left_fraction = float(
        row.left_fraction
    )

    center_fraction = float(
        row.center_fraction
    )

    right_fraction = float(
        row.right_fraction
    )

    visual_L = (
        left_fraction
        + 0.5 * center_fraction
    )

    visual_R = (
        right_fraction
        + 0.5 * center_fraction
    )

    # Structural inhibitory influence before calibration.
    raw_inhibit_R = (
        visual_L
        * left_channel_score
    )

    raw_inhibit_L = (
        visual_R
        * right_channel_score
    )

    # Calibrated 0..1 inhibitory channel activation.
    inhibit_R = min(
        1.0,
        max(
            0.0,
            raw_inhibit_R
            * left_gain,
        ),
    )

    inhibit_L = min(
        1.0,
        max(
            0.0,
            raw_inhibit_L
            * right_gain,
        ),
    )

    # Equal tonic reference drive is an analysis device only.
    # We are validating DNa02 steering SIGN, not claiming this
    # baseline is the fly's biological tonic drive.
    dna02_L_drive = (
        1.0 - inhibit_L
    )

    dna02_R_drive = (
        1.0 - inhibit_R
    )

    # Frozen FlyNav convention:
    #   + = DNa02_R dominance = RIGHT
    #   - = DNa02_L dominance = LEFT
    steering = (
        dna02_R_drive
        - dna02_L_drive
    )

    rows.append(
        {
            "phase": row.phase,
            "target_visible": int(visible),
            "left_fraction": left_fraction,
            "center_fraction": center_fraction,
            "right_fraction": right_fraction,
            "visual_L": visual_L,
            "visual_R": visual_R,
            "raw_inhibit_DNa02_R": raw_inhibit_R,
            "raw_inhibit_DNa02_L": raw_inhibit_L,
            "inhibit_DNa02_R": inhibit_R,
            "inhibit_DNa02_L": inhibit_L,
            "dna02_L_drive_proxy": dna02_L_drive,
            "dna02_R_drive_proxy": dna02_R_drive,
            "steering_proxy": steering,
        }
    )


frame_df = pd.DataFrame(
    rows
)

frame_df.to_csv(
    FRAME_OUT,
    index=False,
)


# ------------------------------------------------------------
# 5. Phase validation.
# ------------------------------------------------------------

phase_rows = []

for phase, group in frame_df.groupby(
    "phase",
    sort=False,
):
    visible_group = group[
        group["target_visible"] == 1
    ]

    if visible_group.empty:
        median_steering = float("nan")
        mean_steering = float("nan")
        visible_frames = 0
    else:
        median_steering = float(
            visible_group[
                "steering_proxy"
            ].median()
        )

        mean_steering = float(
            visible_group[
                "steering_proxy"
            ].mean()
        )

        visible_frames = int(
            len(visible_group)
        )

    phase_upper = str(
        phase
    ).upper()

    if phase_upper == "LEFT":
        passed = (
            visible_frames > 0
            and median_steering < -0.15
        )
        expected = "negative / LEFT"

    elif phase_upper == "CENTER":
        passed = (
            visible_frames > 0
            and abs(
                median_steering
            ) <= 0.10
        )
        expected = "near zero"

    elif phase_upper == "RIGHT":
        passed = (
            visible_frames > 0
            and median_steering > +0.15
        )
        expected = "positive / RIGHT"

    else:
        passed = False
        expected = "unknown"

    phase_rows.append(
        {
            "phase": phase,
            "visible_frames": visible_frames,
            "mean_steering_proxy": mean_steering,
            "median_steering_proxy": median_steering,
            "expected": expected,
            "passed": passed,
        }
    )


summary_df = pd.DataFrame(
    phase_rows
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ------------------------------------------------------------
# Console report.
# ------------------------------------------------------------

print()
print("=" * 118)
print("CAMERA + CONNECTOME SIGN TEST")
print("=" * 118)

for row in summary_df.itertuples(index=False):
    median = (
        float(row.median_steering_proxy)
    )

    print(
        f"{str(row.phase):<8} "
        f"{'PASS' if row.passed else 'FAIL':<4} | "
        f"visible={int(row.visible_frames):>3} | "
        f"median steering={median:+.4f} | "
        f"expected={row.expected}"
    )


passed_count = int(
    summary_df["passed"].sum()
)

print()
print(
    f"Passed phases: "
    f"{passed_count}/{len(summary_df)}"
)

print()
print("=" * 118)
print("SCIENTIFIC INTERPRETATION")
print("=" * 118)

print(
    "Biological evidence used:"
)
print(
    "  LC10a -> AOTU019 anatomical edges"
)
print(
    "  AOTU019 = GABA"
)
print(
    "  AOTU019_L -> DNa02_R and "
    "AOTU019_R -> DNa02_L"
)
print()
print(
    "Engineering interfaces used:"
)
print(
    "  robot camera LEFT/RIGHT -> "
    "LC10a_L/R visual channels"
)
print(
    "  center-sector evidence split equally"
)
print(
    "  per-side structural normalisation"
)
print(
    "  equal tonic drive proxy for offline sign validation"
)

print()
print("=" * 118)
print("FILES CREATED")
print("=" * 118)
print(FRAME_OUT)
print(SUMMARY_OUT)

if passed_count == len(summary_df):
    print()
    print(
        "STEP 54 PASS."
    )
    print(
        "The measured camera bearing and MaleCNS "
        "LC10a -> AOTU019 -> DNa02 bridge produce "
        "the correct frozen FlyNav steering signs."
    )
    print(
        "NEXT: Step 55 can safely let the rover rotate/move "
        "under this visual neural steering signal."
    )
else:
    print()
    print(
        "STEP 54 REVIEW REQUIRED."
    )
    print(
        "Do not control the rover with this mapping yet."
    )


if __name__ == "__main__":
    pass
