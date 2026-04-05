"""
SGP Parcellation Module
=======================
Maps TRIBE v2 fsaverage5 cortical output (~20,484 vertices)
to the 9 SGP nodes using the Schaefer-200 atlas ROI assignments.

Node definitions are grounded in the dual-stream language model
and Human Connectome Project tractography literature.

Nodes:
    G1  Broca         - Phonological production, syntactic processing
    G2  Wernicke      - Auditory comprehension, lexical-semantic decoding
    G3  TPJ           - Stream convergence, sensorimotor interface
    G4  PFC           - Executive control, coherence/veto
    G5  DMN           - Generativity, self-referential processing
    G6  Limbic        - Emotional weighting, memory consolidation
    G7  Sensory       - Primary perceptual input encoding
    G8  ATL           - Cross-modal semantic integration hub
    G9  Premotor      - Action planning, motor speech preparation

White Matter Tracts (edge weights derived from co-activation):
    AF    Arcuate Fasciculus       G2 <-> G1
    SLF   Superior Long. Fasc.     G3 <-> G4
    IFOF  Inf. Fronto-Occip. Fasc. G8 <-> G4
    ILF   Inf. Long. Fasc.         G7 <-> G2
    UF    Uncinate Fasciculus      G8 <-> G6
    CG    Cingulum                 G6 <-> G4 <-> G5
    CC    Corpus Callosum          bilateral integration
    MdLF  Mid. Long. Fasc.         G2 <-> G7
"""

import numpy as np
from typing import Dict, Tuple
import os
import urllib.request

# ─── Schaefer-200 ROI to SGP Node Mapping ────────────────────────────────────
# The Schaefer-200 atlas divides the cortex into 200 parcels (100/hemisphere).
# Each parcel belongs to one of 7 Yeo networks. We map these to our 9 SGP nodes.
# 
# Yeo network assignments:
#   Vis  = Visual
#   SomMot = Somatomotor  
#   DorsAttn = Dorsal Attention
#   SalVentAttn = Salience/Ventral Attention
#   Limbic = Limbic
#   Cont = Control (Frontoparietal)
#   Default = Default Mode
#
# Additional anatomical assignments for language regions not captured by Yeo:
#   IFG pars opercularis/triangularis -> G1 Broca
#   posterior STG/STS -> G2 Wernicke
#   TPJ (supramarginal + angular gyrus) -> G3
#   anterior temporal lobe -> G8 ATL

# Schaefer-200 parcel indices (1-indexed, as in atlas) mapped to SGP nodes.
# Left hemisphere: parcels 1-100, Right hemisphere: 101-200
# These assignments are based on Yeo network membership + anatomical location.

SGP_NODE_DEFINITIONS = {
    "G1_broca": {
        "description": "Broca's Area - phonological production, syntactic processing",
        "stream": "dorsal",
        "hemisphere": "left_dominant",
        # IFG pars opercularis (BA44) and triangularis (BA45)
        # Schaefer-200 Cont network, inferior frontal
        "yeo_networks": ["Cont"],
        "anatomical_keywords": ["IFG", "Oper", "Tri", "Broca"],
        "mni_center": [-51, 12, 18],  # approximate MNI center
    },
    "G2_wernicke": {
        "description": "Wernicke's Area - auditory comprehension, lexical-semantic",
        "stream": "ventral",
        "hemisphere": "left_dominant",
        # Posterior superior temporal gyrus (BA22), planum temporale
        "yeo_networks": ["Default", "SalVentAttn"],
        "anatomical_keywords": ["STG", "STS", "Temp", "Wernicke"],
        "mni_center": [-54, -36, 12],
    },
    "G3_tpj": {
        "description": "Temporoparietal Junction - stream convergence, sensorimotor interface",
        "stream": "convergence",
        "hemisphere": "bilateral",
        # Supramarginal gyrus + angular gyrus
        "yeo_networks": ["DorsAttn", "SalVentAttn"],
        "anatomical_keywords": ["TPJ", "SupraMarginal", "Angular", "ParieTempOcc"],
        "mni_center": [-54, -42, 24],
    },
    "G4_pfc": {
        "description": "Prefrontal Cortex - executive control, coherence/veto",
        "stream": "dorsal",
        "hemisphere": "bilateral",
        # DLPFC, anterior cingulate, orbitofrontal
        "yeo_networks": ["Cont", "SalVentAttn"],
        "anatomical_keywords": ["PFC", "Frontal", "ACC", "OFC", "DLPFC"],
        "mni_center": [-36, 48, 18],
    },
    "G5_dmn": {
        "description": "Default Mode Network - generativity, self-referential processing",
        "stream": "generative",
        "hemisphere": "bilateral",
        # Medial PFC, posterior cingulate, angular gyrus
        "yeo_networks": ["Default"],
        "anatomical_keywords": ["PCC", "mPFC", "Precuneus", "Default"],
        "mni_center": [0, -52, 26],
    },
    "G6_limbic": {
        "description": "Limbic System - emotional weighting, memory consolidation",
        "stream": "modulatory",
        "hemisphere": "bilateral",
        # Amygdala, hippocampus, insula, parahippocampal
        "yeo_networks": ["Limbic", "SalVentAttn"],
        "anatomical_keywords": ["Amyg", "Hipp", "Insula", "ParaHipp", "Limbic"],
        "mni_center": [-24, -18, -18],
    },
    "G7_sensory": {
        "description": "Primary Sensory Cortices - perceptual input encoding",
        "stream": "input",
        "hemisphere": "bilateral",
        # V1/V2 (occipital), A1 (Heschl's gyrus), S1 (postcentral gyrus)
        "yeo_networks": ["Vis", "SomMot"],
        "anatomical_keywords": ["Calc", "Cuneus", "Lingual", "Occip", "Heschl", "PostCentral"],
        "mni_center": [0, -84, 6],
    },
    "G8_atl": {
        "description": "Anterior Temporal Lobe - cross-modal semantic integration",
        "stream": "ventral",
        "hemisphere": "bilateral",
        # Temporal pole, anterior MTG, anterior ITG
        "yeo_networks": ["Default", "Limbic"],
        "anatomical_keywords": ["TempPole", "ATL", "FrontTemp"],
        "mni_center": [-42, 6, -30],
    },
    "G9_premotor": {
        "description": "Premotor/SMA - action planning, motor speech preparation",
        "stream": "dorsal",
        "hemisphere": "left_dominant",
        # Premotor cortex, SMA, precentral gyrus inferior
        "yeo_networks": ["SomMot", "Cont"],
        "anatomical_keywords": ["Precentral", "Premotor", "SMA", "Motor"],
        "mni_center": [-42, 0, 48],
    },
}

# White matter tract definitions — edges in the Resonance Graph
SGP_TRACT_DEFINITIONS = {
    "AF": {
        "name": "Arcuate Fasciculus",
        "connects": ("G2_wernicke", "G1_broca"),
        "function": "Phonological loop, direct sound-to-production",
        "stream": "dorsal",
    },
    "SLF": {
        "name": "Superior Longitudinal Fasciculus",
        "connects": ("G3_tpj", "G4_pfc"),
        "function": "Speech planning, working memory relay",
        "stream": "dorsal",
    },
    "IFOF": {
        "name": "Inferior Fronto-Occipital Fasciculus",
        "connects": ("G8_atl", "G4_pfc"),
        "function": "Semantic/visual meaning stream (ventral)",
        "stream": "ventral",
    },
    "ILF": {
        "name": "Inferior Longitudinal Fasciculus",
        "connects": ("G7_sensory", "G2_wernicke"),
        "function": "Visual word form, perceptual-to-language",
        "stream": "ventral",
    },
    "UF": {
        "name": "Uncinate Fasciculus",
        "connects": ("G8_atl", "G6_limbic"),
        "function": "Emotional-semantic integration",
        "stream": "modulatory",
    },
    "CG_exec": {
        "name": "Cingulum (Executive)",
        "connects": ("G6_limbic", "G4_pfc"),
        "function": "Memory-executive balance",
        "stream": "modulatory",
    },
    "CG_dmn": {
        "name": "Cingulum (DMN)",
        "connects": ("G4_pfc", "G5_dmn"),
        "function": "Ego-creativity balance, DMN regulation",
        "stream": "generative",
    },
    "CC": {
        "name": "Corpus Callosum",
        "connects": ("G3_tpj", "G3_tpj"),  # bilateral — L<->R TPJ as proxy
        "function": "Interhemispheric coherence",
        "stream": "integration",
    },
    "MdLF": {
        "name": "Middle Longitudinal Fasciculus",
        "connects": ("G2_wernicke", "G7_sensory"),
        "function": "Auditory processing continuity",
        "stream": "ventral",
    },
}


class SGPParcellator:
    """
    Maps TRIBE v2 fsaverage5 vertex activations to SGP node scores.
    
    Uses a vertex-to-parcel lookup table derived from the Schaefer-200 atlas
    projected onto the fsaverage5 surface mesh.
    """

    # fsaverage5 has 20,484 vertices (10,242 per hemisphere)
    N_VERTICES = 20484
    N_VERTICES_PER_HEMI = 10242

    # Schaefer-200 atlas file URLs (fsaverage5 surface labels)
    ATLAS_URLS = {
        "lh": "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_200Parcels_7Networks_order.annot",
        "rh": "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_200Parcels_7Networks_order.annot",
    }

    def __init__(self, cache_dir: str = "/tmp/sgp_atlas"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._vertex_to_node = None  # lazy load

    def _build_vertex_to_node_map(self) -> np.ndarray:
        """
        Build a (20484,) array mapping each vertex index to a SGP node name.
        Uses keyword matching against Schaefer-200 parcel labels.
        Falls back to Yeo network assignment if no keyword match.
        """
        try:
            import nibabel as nib
            vertex_map = np.full(self.N_VERTICES, "G5_dmn", dtype=object)  # DMN as default

            for hemi_idx, hemi in enumerate(["lh", "rh"]):
                annot_path = os.path.join(self.cache_dir, f"{hemi}.schaefer200.annot")

                if not os.path.exists(annot_path):
                    print(f"[SGP] Downloading Schaefer-200 {hemi} atlas...", flush=True)
                    try:
                        urllib.request.urlretrieve(self.ATLAS_URLS[hemi], annot_path)
                    except Exception as e:
                        print(f"[SGP] Atlas download failed: {e}. Using fallback.", flush=True)
                        return self._fallback_vertex_map()

                labels, ctab, names = nib.freesurfer.read_annot(annot_path)
                # labels: (10242,) int array, each value is parcel index
                # names: list of parcel name bytes

                vertex_offset = hemi_idx * self.N_VERTICES_PER_HEMI

                for v_local, parcel_idx in enumerate(labels):
                    v_global = v_local + vertex_offset
                    if parcel_idx <= 0 or parcel_idx >= len(names):
                        vertex_map[v_global] = "G7_sensory"  # medial wall -> sensory fallback
                        continue

                    parcel_name = names[parcel_idx].decode("utf-8", errors="ignore")
                    vertex_map[v_global] = self._parcel_name_to_node(parcel_name, hemi)

            print(f"[SGP] Parcellation map built. Node counts:", flush=True)
            for node in SGP_NODE_DEFINITIONS:
                count = np.sum(vertex_map == node)
                print(f"  {node}: {count} vertices", flush=True)

            return vertex_map

        except ImportError:
            print("[SGP] nibabel not available. Using fallback parcellation.", flush=True)
            return self._fallback_vertex_map()

    def _parcel_name_to_node(self, parcel_name: str, hemi: str) -> str:
        """
        Map a Schaefer-200 parcel name to an SGP node.
        Priority: anatomical keyword match > Yeo network match > default (DMN).
        """
        name_upper = parcel_name.upper()

        # Check anatomical keywords first (highest specificity)
        for node_id, node_def in SGP_NODE_DEFINITIONS.items():
            for kw in node_def["anatomical_keywords"]:
                if kw.upper() in name_upper:
                    return node_id

        # Fall back to Yeo network
        yeo_to_node = {
            "VIS": "G7_sensory",
            "SOMMOT": "G9_premotor",
            "DORSATTN": "G3_tpj",
            "SALVENTATT": "G4_pfc",
            "LIMBIC": "G6_limbic",
            "CONT": "G4_pfc",
            "DEFAULT": "G5_dmn",
        }
        for yeo_key, node_id in yeo_to_node.items():
            if yeo_key in name_upper:
                return node_id

        return "G5_dmn"  # final fallback

    def _fallback_vertex_map(self) -> np.ndarray:
        """
        Anatomically-motivated fallback when atlas files unavailable.
        Divides fsaverage5 vertices by approximate cortical location.
        This is a rough approximation — atlas-based mapping is strongly preferred.
        """
        print("[SGP] Using fallback vertex map (approximate)", flush=True)
        vertex_map = np.empty(self.N_VERTICES, dtype=object)

        # Left hemisphere (vertices 0-10241): language dominant
        lh_assignments = [
            (0, 600, "G7_sensory"),       # occipital/visual
            (600, 1200, "G2_wernicke"),   # posterior temporal
            (1200, 1800, "G3_tpj"),       # temporoparietal
            (1800, 2400, "G6_limbic"),    # medial temporal/limbic
            (2400, 3000, "G8_atl"),       # anterior temporal
            (3000, 3600, "G1_broca"),     # inferior frontal
            (3600, 4200, "G4_pfc"),       # prefrontal
            (4200, 4800, "G9_premotor"),  # premotor
            (4800, 5400, "G5_dmn"),       # medial/posterior (DMN)
            (5400, 10242, "G5_dmn"),      # remaining -> DMN
        ]
        for start, end, node in lh_assignments:
            vertex_map[start:end] = node

        # Right hemisphere (vertices 10242-20483): less language dominant
        rh_offset = self.N_VERTICES_PER_HEMI
        rh_assignments = [
            (0, 600, "G7_sensory"),
            (600, 1200, "G7_sensory"),    # bilateral visual
            (1200, 1800, "G3_tpj"),       # bilateral TPJ
            (1800, 2400, "G6_limbic"),
            (2400, 3000, "G8_atl"),
            (3000, 3600, "G4_pfc"),       # bilateral PFC
            (3600, 4200, "G4_pfc"),
            (4200, 4800, "G9_premotor"),
            (4800, 10242, "G5_dmn"),
        ]
        for start, end, node in rh_assignments:
            vertex_map[rh_offset + start:rh_offset + end] = node

        return vertex_map

    def get_vertex_map(self) -> np.ndarray:
        """Lazy-load the vertex-to-node map."""
        if self._vertex_to_node is None:
            self._vertex_to_node = self._build_vertex_to_node_map()
        return self._vertex_to_node

    def parcellate(self, pred_array: np.ndarray) -> Dict:
        """
        Convert TRIBE v2 prediction array to SGP node activation scores.

        Args:
            pred_array: (n_timesteps, n_vertices) float array from TRIBE v2

        Returns:
            dict with keys:
                sgp_nodes: {node_id: float} activation score 0-1 per node
                streams: {stream_name: float} mean activation per stream
                edge_weights: {tract_id: float} co-activation strength per tract
                dominant_hemisphere: "left" | "right" | "bilateral"
                raw_stats: additional diagnostic info
        """
        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(1, -1)

        n_timesteps, n_vertices = pred_array.shape
        vertex_map = self.get_vertex_map()

        # Compute mean absolute activation per node
        node_activations = {}
        node_vertex_counts = {}

        for node_id in SGP_NODE_DEFINITIONS:
            mask = vertex_map[:n_vertices] == node_id
            if mask.sum() == 0:
                node_activations[node_id] = 0.0
                node_vertex_counts[node_id] = 0
                continue
            node_act = float(np.abs(pred_array[:, mask]).mean())
            node_activations[node_id] = node_act
            node_vertex_counts[node_id] = int(mask.sum())

        # Normalize to 0-1 range across nodes
        max_act = max(node_activations.values()) if node_activations else 1.0
        if max_act == 0:
            max_act = 1.0

        sgp_nodes = {
            node_id: round(float(val / max_act), 4)
            for node_id, val in node_activations.items()
        }

        # Stream aggregations
        stream_map = {
            "dorsal": ["G1_broca", "G3_tpj", "G4_pfc", "G9_premotor"],
            "ventral": ["G2_wernicke", "G7_sensory", "G8_atl"],
            "generative": ["G5_dmn"],
            "modulatory": ["G6_limbic"],
            "convergence": ["G3_tpj"],
        }
        streams = {
            stream: round(float(np.mean([sgp_nodes[n] for n in nodes])), 4)
            for stream, nodes in stream_map.items()
        }

        # Edge weights: co-activation strength between connected nodes
        edge_weights = {}
        for tract_id, tract_def in SGP_TRACT_DEFINITIONS.items():
            n1, n2 = tract_def["connects"]
            if n1 == n2:
                # Bilateral tract (CC) — use hemispheric correlation
                mid = n_vertices // 2
                lh_act = float(np.abs(pred_array[:, :mid]).mean())
                rh_act = float(np.abs(pred_array[:, mid:]).mean())
                co_act = 1.0 - abs(lh_act - rh_act) / max(lh_act + rh_act, 1e-8)
            else:
                a1 = node_activations.get(n1, 0.0)
                a2 = node_activations.get(n2, 0.0)
                # Geometric mean as co-activation measure
                co_act = float(np.sqrt(a1 * a2)) / max_act
            edge_weights[tract_id] = round(co_act, 4)

        # Hemispheric dominance
        mid = min(n_vertices // 2, self.N_VERTICES_PER_HEMI)
        lh_mean = float(np.abs(pred_array[:, :mid]).mean())
        rh_mean = float(np.abs(pred_array[:, mid:n_vertices]).mean())
        dom_ratio = (lh_mean - rh_mean) / max(lh_mean + rh_mean, 1e-8)
        if dom_ratio > 0.05:
            dominant_hemisphere = "left"
        elif dom_ratio < -0.05:
            dominant_hemisphere = "right"
        else:
            dominant_hemisphere = "bilateral"

        return {
            "sgp_nodes": sgp_nodes,
            "streams": streams,
            "edge_weights": edge_weights,
            "dominant_hemisphere": dominant_hemisphere,
            "raw_stats": {
                "n_timesteps": n_timesteps,
                "n_vertices": n_vertices,
                "overall_mean_activation": round(float(np.abs(pred_array).mean()), 6),
                "node_vertex_counts": node_vertex_counts,
                "lh_mean": round(lh_mean, 6),
                "rh_mean": round(rh_mean, 6),
            }
        }


# Module-level singleton
_parcellator = None

def get_parcellator(cache_dir: str = "/tmp/sgp_atlas") -> SGPParcellator:
    global _parcellator
    if _parcellator is None:
        _parcellator = SGPParcellator(cache_dir=cache_dir)
    return _parcellator
