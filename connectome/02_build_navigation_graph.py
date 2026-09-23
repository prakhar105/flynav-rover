import os
from pathlib import Path

import matplotlib.pyplot as plt
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

NEURON_FILE = OUTPUT_DIR / "navigation_neurons.csv"
EDGE_FILE = OUTPUT_DIR / "navigation_edges.csv"
TYPE_EDGE_FILE = OUTPUT_DIR / "navigation_type_connections.csv"
GRAPH_FILE = OUTPUT_DIR / "navigation_graph.graphml"
PLOT_FILE = OUTPUT_DIR / "navigation_graph.png"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# Credentials
# =========================================================

load_dotenv(ENV_FILE)

token = os.getenv("NEUPRINT_TOKEN")

if not token:
    raise RuntimeError(
        f"NEUPRINT_TOKEN not found in {ENV_FILE}"
    )


# =========================================================
# Connect to neuPrint
# =========================================================

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

print("Connected to:", client.dataset)


# =========================================================
# Load neurons discovered in Step 1
# =========================================================

if not NEURON_FILE.exists():
    raise FileNotFoundError(
        f"{NEURON_FILE} does not exist.\n"
        "Run 01_find_navigation_neurons.py first."
    )

neurons = pd.read_csv(NEURON_FILE)

if neurons.empty:
    raise RuntimeError(
        "navigation_neurons.csv contains no neurons."
    )

neurons["bodyId"] = neurons["bodyId"].astype("int64")

body_ids = neurons["bodyId"].tolist()

print()
print(f"Loaded {len(body_ids)} candidate navigation neurons.")


# =========================================================
# Query actual synaptic connectivity
# =========================================================

body_id_string = ", ".join(
    str(int(body_id))
    for body_id in body_ids
)

query = f"""
MATCH (source:Neuron)-[connection:ConnectsTo]->(target:Neuron)

WHERE source.bodyId IN [{body_id_string}]
AND target.bodyId IN [{body_id_string}]

RETURN
    source.bodyId AS source_bodyId,
    source.type AS source_type,
    source.instance AS source_instance,

    target.bodyId AS target_bodyId,
    target.type AS target_type,
    target.instance AS target_instance,

    connection.weight AS weight

ORDER BY connection.weight DESC
"""

edges = client.fetch_custom(query)


# =========================================================
# Basic checks
# =========================================================

print()
print("=" * 90)
print("ACTUAL CONNECTOME EDGES")
print("=" * 90)

if edges.empty:
    print(
        "No direct connections were found between "
        "the selected neuron families."
    )
    raise SystemExit(0)

edges["weight"] = pd.to_numeric(
    edges["weight"],
    errors="coerce"
).fillna(0)

print()
print(f"Direct neuron-to-neuron edges: {len(edges)}")
print()

print(
    edges.head(30).to_string(index=False)
)


# =========================================================
# Save raw neuron-to-neuron connections
# =========================================================

edges.to_csv(
    EDGE_FILE,
    index=False
)

print()
print("Saved:")
print(EDGE_FILE)


# =========================================================
# Summarise TYPE -> TYPE connectivity
# =========================================================

type_connections = (
    edges
    .groupby(
        ["source_type", "target_type"],
        dropna=False
    )
    .agg(
        total_synaptic_weight=("weight", "sum"),
        connection_count=("weight", "size"),
        max_connection_weight=("weight", "max"),
    )
    .reset_index()
    .sort_values(
        "total_synaptic_weight",
        ascending=False
    )
)

type_connections.to_csv(
    TYPE_EDGE_FILE,
    index=False
)

print()
print("=" * 90)
print("STRONGEST TYPE -> TYPE CONNECTIONS")
print("=" * 90)
print()

print(
    type_connections.head(30).to_string(index=False)
)


# =========================================================
# Build NetworkX directed graph
# =========================================================

G = nx.DiGraph()

# Add neurons
for _, row in neurons.iterrows():

    body_id = int(row["bodyId"])

    neuron_type = (
        str(row["type"])
        if pd.notna(row["type"])
        else "Unknown"
    )

    instance = (
        str(row["instance"])
        if pd.notna(row["instance"])
        else ""
    )

    G.add_node(
        body_id,
        neuron_type=neuron_type,
        instance=instance,
    )


# Add biological connections
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
# Graph statistics
# =========================================================

print()
print("=" * 90)
print("GRAPH STATISTICS")
print("=" * 90)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

isolated = list(nx.isolates(G))

print(f"Isolated neurons: {len(isolated)}")


if G.number_of_edges() > 0:

    degrees = sorted(
        G.degree,
        key=lambda item: item[1],
        reverse=True
    )

    print()
    print("Highest-degree neurons:")

    for body_id, degree in degrees[:15]:

        data = G.nodes[body_id]

        print(
            f"{body_id} | "
            f"{data['neuron_type']:<20} | "
            f"degree={degree}"
        )


# =========================================================
# Save GraphML
# =========================================================

nx.write_graphml(
    G,
    GRAPH_FILE
)

print()
print("Graph saved:")
print(GRAPH_FILE)


# =========================================================
# Visualise strongest connections only
#
# Plotting every weak edge makes the figure unreadable.
# We keep the strongest 100 biological connections.
# =========================================================

TOP_EDGES = 100

strong_edges = (
    edges
    .sort_values(
        "weight",
        ascending=False
    )
    .head(TOP_EDGES)
)

H = nx.DiGraph()

for _, row in strong_edges.iterrows():

    source = int(row["source_bodyId"])
    target = int(row["target_bodyId"])

    source_type = (
        str(row["source_type"])
        if pd.notna(row["source_type"])
        else "Unknown"
    )

    target_type = (
        str(row["target_type"])
        if pd.notna(row["target_type"])
        else "Unknown"
    )

    weight = float(row["weight"])

    H.add_node(
        source,
        neuron_type=source_type
    )

    H.add_node(
        target,
        neuron_type=target_type
    )

    H.add_edge(
        source,
        target,
        weight=weight
    )


# =========================================================
# Plot
# =========================================================

plt.figure(figsize=(18, 14))

position = nx.spring_layout(
    H,
    seed=42,
    k=1.2,
    iterations=100
)

weights = [
    H[u][v]["weight"]
    for u, v in H.edges()
]

max_weight = max(weights) if weights else 1

edge_widths = [
    0.5 + (weight / max_weight) * 4
    for weight in weights
]

labels = {
    node: H.nodes[node]["neuron_type"]
    for node in H.nodes()
}

nx.draw_networkx_nodes(
    H,
    position,
    node_size=450,
)

nx.draw_networkx_edges(
    H,
    position,
    width=edge_widths,
    arrows=True,
    arrowsize=12,
    alpha=0.5,
)

nx.draw_networkx_labels(
    H,
    position,
    labels=labels,
    font_size=7,
)

plt.title(
    "MaleCNS Navigation / Steering Circuit\n"
    "Top 100 strongest direct connectome connections"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("Graph image saved:")
print(PLOT_FILE)

print()
print("=" * 90)
print("DONE")
print("=" * 90)