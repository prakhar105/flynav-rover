import os
from pathlib import Path

import networkx as nx
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

NEURON_OUTPUT = OUTPUT_DIR / "steering_core_neurons.csv"
EDGE_OUTPUT = OUTPUT_DIR / "steering_core_edges.csv"
PATH_OUTPUT = OUTPUT_DIR / "steering_paths.csv"


# =========================================================
# Connection
# =========================================================

load_dotenv(ENV_FILE)

token = os.getenv("NEUPRINT_TOKEN")

if not token:
    raise RuntimeError(
        f"NEUPRINT_TOKEN not found in {ENV_FILE}"
    )

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

print("Connected to:", client.dataset)


# =========================================================
# Exact neuron families
#
# Intentionally exclude:
#   GFC2
#
# Keep EPGt separate for now.
# =========================================================

STEERING_TYPES = [
    "EPG",
    "FC2A",
    "FC2B",
    "FC2C",
    "PFL2",
    "PFL3",
    "DNa02",
    "DNa03",
]

type_string = ", ".join(
    f"'{t}'"
    for t in STEERING_TYPES
)


# =========================================================
# Fetch exact neurons + useful annotations
# =========================================================

query_neurons = f"""
MATCH (n:Neuron)

WHERE n.type IN [{type_string}]

RETURN
    n.bodyId AS bodyId,
    n.type AS type,
    n.instance AS instance,
    n.somaSide AS somaSide,
    n.rootSide AS rootSide,
    n.predictedNt AS predictedNt,
    n.consensusNt AS consensusNt,
    n.transmission AS transmission

ORDER BY
    n.type,
    n.instance
"""

neurons = client.fetch_custom(query_neurons)

print()
print("=" * 100)
print("EXACT STEERING CORE")
print("=" * 100)

print(f"Neurons: {len(neurons)}")

print()
print(
    neurons[
        [
            "bodyId",
            "type",
            "instance",
            "somaSide",
            "rootSide",
            "predictedNt",
            "consensusNt",
            "transmission",
        ]
    ].to_string(index=False)
)

neurons.to_csv(
    NEURON_OUTPUT,
    index=False,
)


# =========================================================
# Fetch direct connections
# =========================================================

body_ids = [
    int(x)
    for x in neurons["bodyId"]
]

body_id_string = ", ".join(
    str(x)
    for x in body_ids
)

query_edges = f"""
MATCH
    (source:Neuron)-[c:ConnectsTo]->(target:Neuron)

WHERE source.bodyId IN [{body_id_string}]
AND target.bodyId IN [{body_id_string}]

RETURN
    source.bodyId AS source_bodyId,
    source.type AS source_type,
    source.instance AS source_instance,

    target.bodyId AS target_bodyId,
    target.type AS target_type,
    target.instance AS target_instance,

    c.weight AS weight

ORDER BY
    c.weight DESC
"""

edges = client.fetch_custom(query_edges)

edges["weight"] = pd.to_numeric(
    edges["weight"],
    errors="coerce",
).fillna(0.0)

edges.to_csv(
    EDGE_OUTPUT,
    index=False,
)

print()
print(f"Connections: {len(edges)}")


# =========================================================
# Build directed graph
# =========================================================

G = nx.DiGraph()

for _, row in neurons.iterrows():

    body_id = int(row["bodyId"])

    G.add_node(
        body_id,
        neuron_type=row["type"],
        instance=row["instance"],
    )


for _, row in edges.iterrows():

    source = int(row["source_bodyId"])
    target = int(row["target_bodyId"])
    weight = float(row["weight"])

    G.add_edge(
        source,
        target,
        weight=weight,
    )


# =========================================================
# Find output neurons
# =========================================================

outputs = neurons[
    neurons["type"].isin(
        ["DNa02", "DNa03"]
    )
].copy()

upstream = neurons[
    neurons["type"].isin(
        [
            "EPG",
            "FC2A",
            "FC2B",
            "FC2C",
        ]
    )
].copy()


print()
print("=" * 100)
print("OUTPUT / DESCENDING NEURONS")
print("=" * 100)

print(
    outputs[
        [
            "bodyId",
            "type",
            "instance",
            "somaSide",
            "rootSide",
        ]
    ].to_string(index=False)
)


# =========================================================
# Trace short paths
#
# We only care about short functional routes:
#
# EPG / FC2
#      ↓
# PFL2 / PFL3
#      ↓
# DNa02 / DNa03
#
# max path length = 3 edges
# =========================================================

path_rows = []

for _, source_row in upstream.iterrows():

    source_id = int(
        source_row["bodyId"]
    )

    for _, target_row in outputs.iterrows():

        target_id = int(
            target_row["bodyId"]
        )

        try:
            paths = nx.all_simple_paths(
                G,
                source=source_id,
                target=target_id,
                cutoff=3,
            )

            for path in paths:

                edge_weights = []

                for i in range(
                    len(path) - 1
                ):
                    edge_weights.append(
                        float(
                            G[
                                path[i]
                            ][
                                path[i + 1]
                            ]["weight"]
                        )
                    )

                # Conservative path strength:
                # weakest biological connection
                # in the path.
                bottleneck = min(
                    edge_weights
                )

                total_weight = sum(
                    edge_weights
                )

                node_types = [
                    G.nodes[node][
                        "neuron_type"
                    ]
                    for node in path
                ]

                instances = [
                    G.nodes[node][
                        "instance"
                    ]
                    for node in path
                ]

                path_rows.append(
                    {
                        "source_bodyId":
                            source_id,

                        "source_type":
                            source_row["type"],

                        "source_instance":
                            source_row[
                                "instance"
                            ],

                        "target_bodyId":
                            target_id,

                        "target_type":
                            target_row["type"],

                        "target_instance":
                            target_row[
                                "instance"
                            ],

                        "path_length":
                            len(path) - 1,

                        "path_bodyIds":
                            " -> ".join(
                                str(x)
                                for x in path
                            ),

                        "path_types":
                            " -> ".join(
                                node_types
                            ),

                        "path_instances":
                            " -> ".join(
                                str(x)
                                for x
                                in instances
                            ),

                        "bottleneck_weight":
                            bottleneck,

                        "total_weight":
                            total_weight,
                    }
                )

        except nx.NetworkXNoPath:
            pass


paths_df = pd.DataFrame(
    path_rows
)

if paths_df.empty:

    print()
    print(
        "No short steering paths found."
    )

else:

    paths_df = (
        paths_df
        .sort_values(
            [
                "bottleneck_weight",
                "total_weight",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    paths_df.to_csv(
        PATH_OUTPUT,
        index=False,
    )

    print()
    print("=" * 100)
    print("STRONGEST BIOLOGICAL STEERING PATHS")
    print("=" * 100)
    print()

    display_columns = [
        "source_instance",
        "path_types",
        "target_instance",
        "bottleneck_weight",
        "total_weight",
    ]

    print(
        paths_df[
            display_columns
        ]
        .head(40)
        .to_string(
            index=False
        )
    )


# =========================================================
# Explicit motor-side connectivity
# =========================================================

print()
print("=" * 100)
print("DIRECT PFL -> DESCENDING MOTOR CONNECTIONS")
print("=" * 100)
print()

motor_edges = edges[
    edges["source_type"].isin(
        ["PFL2", "PFL3"]
    )
    &
    edges["target_type"].isin(
        ["DNa02", "DNa03"]
    )
].copy()

motor_edges = motor_edges.sort_values(
    "weight",
    ascending=False,
)

print(
    motor_edges[
        [
            "source_type",
            "source_instance",
            "target_type",
            "target_instance",
            "weight",
        ]
    ]
    .head(60)
    .to_string(index=False)
)


print()
print("=" * 100)
print("FILES CREATED")
print("=" * 100)

print(NEURON_OUTPUT)
print(EDGE_OUTPUT)
print(PATH_OUTPUT)