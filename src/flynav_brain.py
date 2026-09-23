from pathlib import Path

import numpy as np
import pandas as pd

from brian2 import (
    Network,
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    prefs,
    ms,
)

# Use NumPy backend.
# Avoids requiring Microsoft C++ Build Tools.
prefs.codegen.target = "numpy"


class FlyNavBrain:
    """
    Connectome-derived steering controller.

    Biological topology:
        PFL3 -> DNa02
        PFL3 -> DNa03
        DNa03 -> DNa02

    Connection strengths come from MaleCNS.

    Important:
    The neuron dynamics and synaptic scaling are engineering
    approximations, not exact Drosophila electrophysiology.
    """

    def __init__(self, project_root):

        self.project_root = Path(project_root)

        output_dir = (
            self.project_root
            / "connectome"
            / "output"
        )

        neuron_file = (
            output_dir
            / "steering_core_neurons.csv"
        )

        edge_file = (
            output_dir
            / "steering_core_edges.csv"
        )

        neurons = pd.read_csv(neuron_file)
        edges = pd.read_csv(edge_file)

        neurons["bodyId"] = (
            neurons["bodyId"]
            .astype("int64")
        )

        edges["source_bodyId"] = (
            edges["source_bodyId"]
            .astype("int64")
        )

        edges["target_bodyId"] = (
            edges["target_bodyId"]
            .astype("int64")
        )

        edges["weight"] = pd.to_numeric(
            edges["weight"],
            errors="coerce",
        ).fillna(0.0)

        # ---------------------------------------------
        # Steering core only
        # ---------------------------------------------

        keep_types = [
            "PFL3",
            "DNa03",
            "DNa02",
        ]

        self.neurons = neurons[
            neurons["type"].isin(
                keep_types
            )
        ].copy()

        core_ids = set(
            self.neurons["bodyId"]
        )

        core_edges = edges[
            edges["source_bodyId"]
            .isin(core_ids)
            &
            edges["target_bodyId"]
            .isin(core_ids)
        ].copy()

        valid_edges = (
            (
                (
                    core_edges["source_type"]
                    == "PFL3"
                )
                &
                (
                    core_edges["target_type"]
                    .isin(
                        [
                            "DNa02",
                            "DNa03",
                        ]
                    )
                )
            )
            |
            (
                (
                    core_edges["source_type"]
                    == "DNa03"
                )
                &
                (
                    core_edges["target_type"]
                    == "DNa02"
                )
            )
        )

        self.edges = core_edges[
            valid_edges
        ].copy()

        self.neurons = (
            self.neurons
            .reset_index(drop=True)
        )

        self.body_to_index = {
            int(body_id): idx
            for idx, body_id
            in enumerate(
                self.neurons["bodyId"]
            )
        }

        self.side_map = dict(
            zip(
                neurons["bodyId"],
                neurons["somaSide"],
            )
        )

        self._classify_pfl3_channels()

        self._build_network()

    # =====================================================
    # Determine LEFT / RIGHT turn PFL3 populations
    # =====================================================

    def _classify_pfl3_channels(self):

        pfl3_edges = self.edges[
            self.edges["source_type"]
            == "PFL3"
        ].copy()

        pfl3_edges["target_side"] = (
            pfl3_edges[
                "target_bodyId"
            ].map(
                self.side_map
            )
        )

        strength = (
            pfl3_edges[
                pfl3_edges[
                    "target_side"
                ].isin(["L", "R"])
            ]
            .groupby(
                [
                    "source_bodyId",
                    "target_side",
                ]
            )["weight"]
            .sum()
            .unstack(
                fill_value=0
            )
        )

        if "L" not in strength.columns:
            strength["L"] = 0.0

        if "R" not in strength.columns:
            strength["R"] = 0.0

        self.left_turn_ids = []
        self.right_turn_ids = []

        for body_id, row in strength.iterrows():

            left_target = float(
                row["L"]
            )

            right_target = float(
                row["R"]
            )

            if left_target > right_target:
                self.left_turn_ids.append(
                    int(body_id)
                )

            elif right_target > left_target:
                self.right_turn_ids.append(
                    int(body_id)
                )

    # =====================================================
    # Build Brian2 network
    # =====================================================

    def _build_network(self):

        N = len(self.neurons)

        equations = """
        dv/dt = (-v + drive) / (20*ms) : 1
        drive : 1
        """

        self.G = NeuronGroup(
            N,
            equations,
            threshold="v > 1.0",
            reset="v = 0.0",
            refractory=3 * ms,
            method="euler",
        )

        self.G.v = 0.0
        self.G.drive = 0.05

        self.S = Synapses(
            self.G,
            self.G,
            model="w : 1",
            on_pre="v_post += w",
        )

        source_indices = []
        target_indices = []
        raw_weights = []

        for _, row in self.edges.iterrows():

            source = int(
                row["source_bodyId"]
            )

            target = int(
                row["target_bodyId"]
            )

            if (
                source
                not in self.body_to_index
                or
                target
                not in self.body_to_index
            ):
                continue

            source_indices.append(
                self.body_to_index[
                    source
                ]
            )

            target_indices.append(
                self.body_to_index[
                    target
                ]
            )

            raw_weights.append(
                float(
                    row["weight"]
                )
            )

        self.S.connect(
            i=source_indices,
            j=target_indices,
        )

        raw_weights = np.asarray(
            raw_weights,
            dtype=float,
        )

        max_raw = max(
            raw_weights.max(),
            1.0,
        )

        scaled = (
            0.05
            +
            0.45
            * np.sqrt(
                raw_weights
                / max_raw
            )
        )

        self.S.w = scaled

        self.monitor = SpikeMonitor(
            self.G
        )

        self.network = Network(
            self.G,
            self.S,
            self.monitor,
        )

        # ---------------------------------------------
        # Cache indices
        # ---------------------------------------------

        self.left_pfl3_indices = [
            self.body_to_index[x]
            for x in self.left_turn_ids
        ]

        self.right_pfl3_indices = [
            self.body_to_index[x]
            for x in self.right_turn_ids
        ]

        self.dna02_left_indices = []
        self.dna02_right_indices = []

        for idx, row in self.neurons.iterrows():

            if row["type"] != "DNa02":
                continue

            if row["somaSide"] == "L":
                self.dna02_left_indices.append(
                    idx
                )

            elif row["somaSide"] == "R":
                self.dna02_right_indices.append(
                    idx
                )

    # =====================================================
    # One control step
    # =====================================================

    def step(
        self,
        obstacle_left,
        obstacle_right,
        duration_ms=20,
    ):

        # Baseline spontaneous drive
        self.G.drive = 0.05

        # -------------------------------------------------
        # Cross coupling:
        #
        # obstacle LEFT
        #       ->
        # RIGHT turn channel
        #
        # obstacle RIGHT
        #       ->
        # LEFT turn channel
        # -------------------------------------------------

        obstacle_left = float(
            np.clip(
                obstacle_left,
                0.0,
                1.0,
            )
        )

        obstacle_right = float(
            np.clip(
                obstacle_right,
                0.0,
                1.0,
            )
        )

# -------------------------------------------------
# Sensory drive
#
# The gain is an engineering scaling parameter.
# Biological topology / relative synaptic strengths
# still come from MaleCNS.
# -------------------------------------------------

        SENSORY_GAIN = 3.2

        self.G.drive[
            self.right_pfl3_indices
        ] = (
            0.05
            + SENSORY_GAIN
            * obstacle_left
        )

        self.G.drive[
            self.left_pfl3_indices
        ] = (
            0.05
            + SENSORY_GAIN
            * obstacle_right
        )

        # Spike counts before this window
        before = np.asarray(
            self.monitor.count
        ).copy()

        self.network.run(
            duration_ms * ms
        )

        after = np.asarray(
            self.monitor.count
        )

        delta = after - before

        left_spikes = int(
            delta[
                self.dna02_left_indices
            ].sum()
        )

        right_spikes = int(
            delta[
                self.dna02_right_indices
            ].sum()
        )

        # -------------------------------------------------
        # Spike-based steering
        # -------------------------------------------------

        steering_score = (
            right_spikes
            - left_spikes
        )

        total_output = (
            right_spikes
            + left_spikes
        )


        # -------------------------------------------------
        # Also inspect sub-threshold DNa02 activity.
        #
        # This preserves useful information when neither
        # side produced an extra whole spike during the
        # current short control window.
        # -------------------------------------------------

        membrane_values = np.asarray(
            self.G.v
        )

        left_v = float(
            membrane_values[
                self.dna02_left_indices
            ].mean()
        )

        right_v = float(
            membrane_values[
                self.dna02_right_indices
            ].mean()
        )

        membrane_difference = (
            right_v
            - left_v
        )


        # -------------------------------------------------
        # Combine spikes + membrane state
        # -------------------------------------------------

        if total_output > 0:

            spike_steering = (
                steering_score
                / total_output
            )

            normalized_steering = (
                0.80 * spike_steering
                +
                0.20 * membrane_difference
            )

        else:

            normalized_steering = (
                membrane_difference
            )


        normalized_steering = float(
            np.clip(
                normalized_steering,
                -1.0,
                1.0,
            )
        )


        return {
            "left_spikes":
                left_spikes,

            "right_spikes":
                right_spikes,

            "left_membrane":
                left_v,

            "right_membrane":
                right_v,

            "steering":
                normalized_steering,
        }