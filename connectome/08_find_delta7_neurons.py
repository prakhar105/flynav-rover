import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "delta7_candidates.csv"
)


# =========================================================
# neuPrint
# =========================================================

load_dotenv(ENV_FILE)

token = os.getenv(
    "NEUPRINT_TOKEN"
)

if not token:
    raise RuntimeError(
        "NEUPRINT_TOKEN not found"
    )


client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)


print(
    "Connected to:",
    client.dataset
)


# =========================================================
# Search candidate Delta7 naming
#
# Different connectome releases may use slightly
# different typography / aliases.
# =========================================================

query = """
MATCH (n:Neuron)

WHERE n.type IS NOT NULL
AND (
       toLower(n.type) CONTAINS 'delta7'
    OR n.type CONTAINS 'Δ7'
    OR n.type CONTAINS '∆7'
    OR toLower(n.type) CONTAINS 'delta 7'
)

RETURN
    n.bodyId AS bodyId,
    n.type AS type,
    n.instance AS instance,
    n.somaSide AS somaSide,
    n.predictedNt AS predictedNt,
    n.consensusNt AS consensusNt

ORDER BY
    n.type,
    n.instance
"""


df = client.fetch_custom(
    query
)


print()
print("=" * 90)
print("DELTA7 CANDIDATES")
print("=" * 90)
print()


if df.empty:

    print(
        "No obvious Delta7-labelled neurons found."
    )

else:

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("COUNTS BY TYPE")
    print("=" * 90)
    print()

    counts = (
        df.groupby("type")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        counts.to_string()
    )

    print()
    print("=" * 90)
    print("NEUROTRANSMITTERS")
    print("=" * 90)
    print()

    print(
        df[
            [
                "type",
                "predictedNt",
                "consensusNt",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )