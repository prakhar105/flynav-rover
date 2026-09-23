import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Credentials
# ---------------------------------------------------------

load_dotenv(ENV_FILE)

token = os.getenv("NEUPRINT_TOKEN")

if not token:
    raise RuntimeError(
        f"NEUPRINT_TOKEN not found in {ENV_FILE}"
    )

# ---------------------------------------------------------
# Connect to MaleCNS
# ---------------------------------------------------------

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

print("Connected to:", client.dataset)

# ---------------------------------------------------------
# Candidate navigation / steering neuron families
# ---------------------------------------------------------

candidate_terms = [
    "EPG",
    "FC2",
    "PFL3",
    "PFL2",
    "DNa02",
    "DNa03",
]

terms = ", ".join(
    f"'{term}'"
    for term in candidate_terms
)

# ---------------------------------------------------------
# Search MaleCNS
# ---------------------------------------------------------

query = f"""
MATCH (n:Neuron)
WHERE n.type IS NOT NULL
AND any(
    term IN [{terms}]
    WHERE toLower(n.type) CONTAINS toLower(term)
)
RETURN
    n.bodyId AS bodyId,
    n.type AS type,
    n.instance AS instance
ORDER BY
    n.type,
    n.instance
"""

df = client.fetch_custom(query)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 80)
print("NAVIGATION NEURONS FOUND")
print("=" * 80)

if df.empty:
    print("No matching neurons found.")
else:
    print(df.to_string(index=False))

    print()
    print("=" * 80)
    print("COUNTS BY TYPE")
    print("=" * 80)

    counts = (
        df.groupby("type")
        .size()
        .sort_values(ascending=False)
    )

    print(counts.to_string())

    output_file = (
        OUTPUT_DIR
        / "navigation_neurons.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Saved {len(df)} neurons to:"
    )
    print(output_file)