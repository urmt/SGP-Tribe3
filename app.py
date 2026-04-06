"""
SGP-Tribe3 — Main API Application
===================================
Multimodal brain encoding API with SGP 9-node parcellation.

Endpoints:
    GET  /           - Service info
    GET  /health     - Model load status
    POST /warmup     - Trigger model loading
    POST /predict    - Run inference on uploaded video file
    GET  /nodes      - SGP node definitions
    GET  /tracts     - White matter tract definitions
"""

import os
import uuid
import math
import json
import warnings
import threading
import traceback
import tempfile
import numpy as np

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from sgp_parcellation import get_parcellator, SGP_NODE_DEFINITIONS, SGP_TRACT_DEFINITIONS

app = Flask(__name__)

# ─── Global model state ───────────────────────────────────────────────────────
_model = None
_model_loaded = False
_model_loading = False
_model_error = None
_model_lock = threading.Lock()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
CKPT = os.environ.get("TRIBE_CKPT", "facebook/tribev2")
MAX_VIDEO_DURATION = int(os.environ.get("MAX_VIDEO_DURATION", "120"))  # seconds
CACHE_DIR = os.environ.get("SGP_CACHE_DIR", "/tmp/sgp_atlas")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", "/tmp/hf_hub_cache")
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["WHISPER_CACHE_DIR"] = os.environ.get("WHISPER_CACHE_DIR", "/tmp/whisper_cache")
os.makedirs(HF_HUB_CACHE, exist_ok=True)

# ─── Result storage (in-memory for now; extend to file/DB for persistence) ───
_stimulus_results = {}


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_model():
    global _model, _model_loaded, _model_loading, _model_error

    with _model_lock:
        if _model_loaded or _model_loading:
            return
        _model_loading = True
        _model_error = None

    try:
        print("[SGP-Tribe3] Starting model load...", flush=True)

        if HF_TOKEN:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            os.environ["HF_TOKEN"] = HF_TOKEN
            os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"
            os.makedirs("/tmp/hf_hub_cache", exist_ok=True)
            try:
                from huggingface_hub import login
                login(token=HF_TOKEN, add_to_git_credential=False)
                print(f"[SGP-Tribe3] HF login OK", flush=True)
            except Exception as e:
                print(f"[SGP-Tribe3] HF login warning: {e}", flush=True)

            # Pre-download whisperx model to avoid rate limits in uvx subprocess
            try:
                from huggingface_hub import snapshot_download
                print("[SGP-Tribe3] Pre-downloading faster-whisper-large-v3...", flush=True)
                snapshot_download(
                    repo_id="Systran/faster-whisper-large-v3",
                    cache_dir="/tmp/hf_hub_cache",
                    token=HF_TOKEN
                )
                print("[SGP-Tribe3] Whisper model cached successfully", flush=True)
            except Exception as e:
                print(f"[SGP-Tribe3] Whisper model download warning: {e}", flush=True)
        else:
            print("[SGP-Tribe3] WARNING: No HF_TOKEN set — LLaMA encoder may fail", flush=True)

        import torch
        print(f"[SGP-Tribe3] PyTorch {torch.__version__}", flush=True)

        # CPU patch for HuggingFace text extractor (same as janrudolph's approach)
        try:
            from neuralset.extractors.text import HuggingFaceText
            orig_load = HuggingFaceText._load_model

            def cpu_patched_load(self):
                object.__setattr__(self, 'device', 'cpu')
                return orig_load(self)

            HuggingFaceText._load_model = cpu_patched_load
            print("[SGP-Tribe3] CPU patch applied", flush=True)
        except Exception as e:
            print(f"[SGP-Tribe3] CPU patch warning: {e}", flush=True)

        from tribev2 import TribeModel
        print("[SGP-Tribe3] Loading TribeModel...", flush=True)
        model = TribeModel.from_pretrained(CKPT, device='cpu')
        print("[SGP-Tribe3] TribeModel loaded!", flush=True)

        # Pre-warm the parcellator (downloads Schaefer atlas if needed)
        print("[SGP-Tribe3] Initializing SGP parcellator...", flush=True)
        parcellator = get_parcellator(CACHE_DIR)
        _ = parcellator.get_vertex_map()  # trigger atlas download
        print("[SGP-Tribe3] Parcellator ready!", flush=True)

        with _model_lock:
            _model = model
            _model_loaded = True
            _model_loading = False
        print("[SGP-Tribe3] READY", flush=True)

    except Exception as e:
        err = traceback.format_exc()
        print(f"[SGP-Tribe3] LOAD ERROR:\n{err}", flush=True)
        with _model_lock:
            _model_loading = False
            _model_error = str(e)


# ─── Video processing ─────────────────────────────────────────────────────────

def _preprocess_video(video_path: str, max_duration: int = MAX_VIDEO_DURATION) -> str:
    """
    Trim video to max_duration and normalize to TRIBE v2 expected format.
    Returns path to processed video file.
    """
    import subprocess

    output_path = video_path.replace(".mp4", "_processed.mp4")

    # Trim to max duration, normalize audio, ensure video stream exists
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", str(max_duration),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-vf", "scale=320:240",  # resize for efficiency on CPU
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg preprocessing failed: {result.stderr}")

    return output_path


def _run_inference(video_path: str) -> dict:
    """
    Run TRIBE v2 inference on a video file and return SGP parcellation.
    """
    import pandas as pd

    # Get video duration and events dataframe from TribeModel
    processed_path = _preprocess_video(video_path)

    try:
        events_df = _model.get_events_dataframe(video_path=processed_path)
        print(f"[SGP-Tribe3] Events extracted: {len(events_df)} rows", flush=True)

        preds, segments = _model.predict(events=events_df, verbose=False)

        # Convert to numpy
        if hasattr(preds, "numpy"):
            pred_array = preds.numpy()
        else:
            pred_array = np.array(preds)

        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(1, -1)

        print(f"[SGP-Tribe3] Prediction shape: {pred_array.shape}", flush=True)

        # Apply SGP parcellation
        parcellator = get_parcellator(CACHE_DIR)
        result = parcellator.parcellate(pred_array)

        # Add activation timeline (mean activation per timestep)
        result["activation_timeline"] = [
            round(float(np.abs(pred_array[t]).mean()), 4)
            for t in range(pred_array.shape[0])
        ]

        return result

    finally:
        # Clean up processed file
        if os.path.exists(processed_path) and processed_path != video_path:
            os.remove(processed_path)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "SGP-Tribe3",
        "version": "1.0.0",
        "description": "Sentient Generative Principal — Brain Encoding Calibration System",
        "status": "ok",
        "endpoints": {
            "GET /health": "Model load status",
            "POST /warmup": "Trigger model loading",
            "POST /predict": "Run multimodal inference on video file",
            "GET /nodes": "SGP node definitions",
            "GET /tracts": "White matter tract definitions",
            "GET /results": "All stored stimulus results",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if _model_loaded else ("loading" if _model_loading else "offline"),
        "model_loaded": _model_loaded,
        "model_loading": _model_loading,
        "error": _model_error,
        "n_stored_results": len(_stimulus_results),
    })


@app.route("/warmup", methods=["POST"])
def warmup():
    if not _model_loaded and not _model_loading:
        threading.Thread(target=_load_model, daemon=True).start()
    return jsonify({
        "status": "warming_up",
        "model_loaded": _model_loaded,
        "model_loading": _model_loading,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not _model_loaded:
        return jsonify({
            "error": "Model not loaded. POST to /warmup first.",
            "model_loading": _model_loading,
            "load_error": _model_error,
        }), 503

    # Accept video file upload
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Use multipart/form-data with 'video' field."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    stimulus_id = request.form.get("stimulus_id", str(uuid.uuid4()))
    stimulus_label = request.form.get("label", "unlabeled")
    target_node = request.form.get("target_node", "unknown")

    # Save uploaded file to temp
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        print(f"[SGP-Tribe3] Running inference: stimulus_id={stimulus_id}, label={stimulus_label}", flush=True)
        result = _run_inference(tmp_path)

        # Attach metadata
        result["stimulus_id"] = stimulus_id
        result["label"] = stimulus_label
        result["target_node"] = target_node

        # Store result
        _stimulus_results[stimulus_id] = result

        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        err = traceback.format_exc()
        print(f"[SGP-Tribe3] Inference error:\n{err}", flush=True)
        return jsonify({"error": str(e), "trace": err}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/nodes", methods=["GET"])
def nodes():
    return jsonify({
        "sgp_nodes": SGP_NODE_DEFINITIONS,
        "count": len(SGP_NODE_DEFINITIONS),
    })


@app.route("/tracts", methods=["GET"])
def tracts():
    return jsonify({
        "white_matter_tracts": SGP_TRACT_DEFINITIONS,
        "count": len(SGP_TRACT_DEFINITIONS),
    })


@app.route("/results", methods=["GET"])
def results():
    return jsonify({
        "n_results": len(_stimulus_results),
        "results": _stimulus_results,
    })


@app.route("/coactivation_matrix", methods=["GET"])
def coactivation_matrix():
    """
    Compute the co-activation matrix across all stored stimulus results.
    This is the edge weight matrix for the Resonance Graph.
    """
    if len(_stimulus_results) < 2:
        return jsonify({
            "error": "Need at least 2 stimulus results to compute co-activation matrix.",
            "n_results": len(_stimulus_results),
        }), 400

    node_ids = list(SGP_NODE_DEFINITIONS.keys())
    n_nodes = len(node_ids)

    # Build activation matrix: (n_stimuli, n_nodes)
    activation_matrix = []
    stimulus_labels = []

    for sid, res in _stimulus_results.items():
        row = [res["sgp_nodes"].get(nid, 0.0) for nid in node_ids]
        activation_matrix.append(row)
        stimulus_labels.append(res.get("label", sid))

    A = np.array(activation_matrix)  # (n_stimuli, n_nodes)

    # Pearson correlation matrix across stimuli
    if A.shape[0] > 1:
        corr_matrix = np.corrcoef(A.T)  # (n_nodes, n_nodes)
    else:
        corr_matrix = np.eye(n_nodes)

    # Also compute mean activation per node across all stimuli
    mean_activation = A.mean(axis=0)

    return jsonify({
        "node_ids": node_ids,
        "n_stimuli": len(_stimulus_results),
        "stimulus_labels": stimulus_labels,
        "coactivation_matrix": corr_matrix.round(4).tolist(),
        "mean_activation_per_node": dict(zip(node_ids, mean_activation.round(4).tolist())),
        "interpretation": "coactivation_matrix[i][j] = Pearson correlation of node_i and node_j activation across all stimuli. Use as Resonance Graph edge weights.",
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=_load_model, daemon=True).start()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
