from pathlib import Path

from src.flynav_brain import FlyNavBrain

ROOT = Path(__file__).resolve().parents[1]

brain = FlyNavBrain(ROOT)


tests = [
    ("CLEAR", 0.0, 0.0),
    ("OBSTACLE LEFT", 1.0, 0.0),
    ("OBSTACLE RIGHT", 0.0, 1.0),
    ("BOTH", 1.0, 1.0),
]
 

for name, left, right in tests:

    print()
    print("=" * 60)
    print(name)

    # Give each condition several control windows
    for _ in range(5):

        result = brain.step(
            obstacle_left=left,
            obstacle_right=right,
            duration_ms=20,
        )

        print(result)