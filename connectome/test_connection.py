import os

from dotenv import load_dotenv
from neuprint import Client

load_dotenv("connectome/.env")

token = os.getenv("NEUPRINT_TOKEN")

if not token:
    raise RuntimeError(
        "NEUPRINT_TOKEN was not found in connectome/.env"
    )

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=token,
)

print("Connected to neuPrint")
print("Server version:", client.fetch_version())
print("Dataset:", client.dataset)