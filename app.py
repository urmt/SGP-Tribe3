"""
SGP-Tribe3 — Main API Application
==================================
Multimodal brain encoding API with SGP 9-node parcellation.
Supports video+audio, audio-only, and text-only inputs via TRIBE v2.

Endpoints:
    GET  /              - Service info
    GET  /health         - Model load status
    POST /warmup         - Trigger model loading
    POST /predict        - Run inference on video file (video + audio encoding)
    POST /predict_text   - Run inference on text input (text-only encoding)
    POST /predict_audio   - Run inference on audio file (audio-only encoding)
    GET  /nodes          - SGP node definitions
    GET  /tracts          - White matter tract definitions
    GET  /results        - All stored stimulus results
    GET  /coactivation_matrix - Cross-stimulus co-activation matrix

Reference: Harvard MLSysBook - Machine Learning Systems
https://github.com/harvard-edge/cs249r_book
"""

import os
import uuid
import math
import json
import warnings
import threading
import traceback
import tempfile
import subprocess
import numpy as np
import pandas as pd

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

# ─── Configuration ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CKPT = os.environ.get("TRIBE_CKPT", "facebook/tribev2")
MAX_VIDEO_DURATION = int(os.environ.get("MAX_VIDEO_DURATION", "120"))
MAX_AUDIO_DURATION = int(os.environ.get("MAX_AUDIO_DURATION", "120"))
CACHE_DIR = os.environ.get("SGP_CACHE_DIR", "/tmp/sgp_atlas")

os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_hub_cache")
os.environ.setdefault("WHISPER_CACHE_DIR", "/tmp/whisper_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["WHISPER_CACHE_DIR"], exist_ok=True)

# ─── Result storage (in-memory for now; extend to HF dataset for persistence) ───
_stimulus_results = {}

# ─── Metrics tracking (MLOps best practice) ──────────────────────────────────
_metrics = {
    "start_time": None,
    "total_predictions": 0,
    "predictions_by_modality": {"video": 0, "audio": 0, "text": 0},
    "inference_times": [],
}


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_model():
    global _model, _model_loaded, _model_loading, _model_error, _metrics

    with _model_lock:
        if _model_loaded or _model_loading:
            return
        _model_loading = True
        _model_error = None

    try:
        print("[SGP-Tribe3] Starting model load...", flush=True)
        _metrics["start_time"] = pd.Timestamp.now().isoformat()

        if HF_TOKEN:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            os.environ["HF_TOKEN"] = HF_TOKEN
            try:
                from huggingface_hub import login
                login(token=HF_TOKEN, add_to_git_credential=False)
                print(f"[SGP-Tribe3] HF login OK", flush=True)
            except Exception as e:
                print(f"[SGP-Tribe3] HF login warning: {e}", flush=True)
        else:
            print("[SGP-Tribe3] WARNING: No HF_TOKEN set — LLaMA encoder may fail", flush=True)

        import torch
        print(f"[SGP-Tribe3] PyTorch {torch.__version__}", flush=True)

        # CPU patch for HuggingFace text extractor (required for CPU-only inference)
        try:
            from neuralset.extractors.text import HuggingFaceText
            orig_load = HuggingFaceText._load_model

            def cpu_patched_load(self):
                object.__setattr__(self, 'device', 'cpu')
                return orig_load(self)

            HuggingFaceText._load_model = cpu_patched_load
            print("[SGP-Tribe3] CPU patch applied to HuggingFaceText", flush=True)
        except Exception as e:
            print(f"[SGP-Tribe3] CPU patch warning: {e}", flush=True)

        # Load TRIBE v2 model
        from tribev2 import TribeModel
        print("[SGP-Tribe3] Loading TribeModel...", flush=True)
        model = TribeModel.from_pretrained(CKPT, device='cpu')
        print("[SGP-Tribe3] TribeModel loaded!", flush=True)

        # Pre-warm the parcellator (downloads Schaefer atlas if needed)
        print("[SGP-Tribe3] Initializing SGP parcellator...", flush=True)
        parcellator = get_parcellator(CACHE_DIR)
        _ = parcellator.get_vertex_map()
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


# ─── Video/Audio preprocessing ────────────────────────────────────────────────

def _preprocess_video(video_path: str, max_duration: int = MAX_VIDEO_DURATION) -> str:
    """
    Trim video to max_duration and normalize to TRIBE v2 expected format.
    Returns path to processed video file.
    """
    output_path = video_path.replace(".mp4", "_processed.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", str(max_duration),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-vf", "scale=320:240",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg preprocessing failed: {result.stderr}")

    return output_path


def _preprocess_audio(audio_path: str, max_duration: int = MAX_AUDIO_DURATION) -> str:
    """
    Convert audio to wav format and normalize for TRIBE v2.
    Returns path to processed audio file.
    """
    output_path = audio_path.replace(audio_path.split(".")[-1], "wav")
    if output_path == audio_path:
        output_path = audio_path.rsplit(".", 1)[0] + "_processed.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-t", str(max_duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg audio preprocessing failed: {result.stderr}")

    return output_path


# ─── Core inference ──────────────────────────────────────────────────────────

def _run_inference_from_events(events_df: pd.DataFrame) -> dict:
    """
    Run TRIBE v2 inference from an events DataFrame and return SGP parcellation.
    This is the core function used by all three modality endpoints.
    """
    import time
    start_time = time.time()

    try:
        # Run TRIBE v2 prediction
        # Note: standardize_events is called inside get_loaders, so we need to ensure
        # our events have the right schema BEFORE calling predict
        preds, segments = _model.predict(events=events_df, verbose=False)

        # Convert to numpy
        if hasattr(preds, "numpy"):
            pred_array = preds.numpy()
        else:
            pred_array = np.array(preds)

        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(1, -1)

        inference_time = time.time() - start_time
        print(f"[SGP-Tribe3] Prediction shape: {pred_array.shape}, time: {inference_time:.1f}s", flush=True)

        # Apply SGP parcellation
        parcellator = get_parcellator(CACHE_DIR)
        result = parcellator.parcellate(pred_array)

        # Add activation timeline (mean activation per timestep)
        result["activation_timeline"] = [
            round(float(np.abs(pred_array[t]).mean()), 4)
            for t in range(pred_array.shape[0])
        ]

        # Add inference metadata
        result["inference_time_seconds"] = round(inference_time, 2)
        result["n_segments"] = pred_array.shape[0]
        result["n_vertices"] = pred_array.shape[1]

        # Update metrics
        _metrics["total_predictions"] += 1
        _metrics["inference_times"].append(inference_time)
        if len(_metrics["inference_times"]) > 100:
            _metrics["inference_times"] = _metrics["inference_times"][-100:]

        return result

    except Exception as e:
        raise RuntimeError(f"TRIBE v2 inference failed: {e}")


def _run_video_inference(video_path: str) -> dict:
    """
    Run inference on video file (video + audio modalities).
    Uses TRIBE v2's get_audio_and_text_events with audio_only=True.
    """
    from tribev2.demo_utils import get_audio_and_text_events

    processed_path = _preprocess_video(video_path)

    try:
        # Create initial video event with ALL required columns for TRIBE v2 schema
        event = pd.DataFrame([{
            "type": "Video",
            "filepath": processed_path,
            "start": 0.0,
            "timeline": "default",
            "subject": "default",
            "duration": MAX_VIDEO_DURATION,
            "offset": 0.0,
            "frequency": 1.0,
            "extra": {}
        }])
        
        # Use TRIBE v2 pipeline: extracts audio, chunks, but SKIPS whisperx
        events_df = get_audio_and_text_events(event, audio_only=True)

        # FIX: The get_audio_and_text_events may not preserve all required fields
        # We need to ensure every event has the required columns before calling predict
        required_base_cols = ["type", "filepath", "start", "timeline", "subject", "duration", "offset", "frequency", "extra"]
        
        for idx, row in events_df.iterrows():
            event_type = row.get("type", "")
            for col in required_base_cols:
                if col not in events_df.columns or pd.isna(events_df.at[idx, col]):
                    if col == "timeline":
                        events_df.at[idx, col] = "default"
                    elif col == "subject":
                        events_df.at[idx, col] = "default"
                    elif col == "duration":
                        events_df.at[idx, col] = MAX_VIDEO_DURATION
                    elif col == "offset":
                        events_df.at[idx, col] = 0.0
                    elif col == "frequency":
                        events_df.at[idx, col] = 1.0
                    elif col == "extra":
                        events_df.at[idx, col] = {}
                    elif col == "filepath":
                        if event_type in ["Video", "Audio"]:
                            events_df.at[idx, col] = row.get("filepath", processed_path)
                        else:
                            events_df.at[idx, col] = None

        event_types = events_df['type'].unique().tolist()
        print(f"[SGP-Tribe3] Video inference: {len(events_df)} events, types: {event_types}", flush=True)

        _metrics["predictions_by_modality"]["video"] += 1
        return _run_inference_from_events(events_df)

    finally:
        if os.path.exists(processed_path) and processed_path != video_path:
            os.remove(processed_path)


def _run_audio_inference(audio_path: str) -> dict:
    """
    Run inference on audio file (audio-only modality).
    Uses TRIBE v2's get_audio_and_text_events with audio_only=True.
    """
    from tribev2.demo_utils import get_audio_and_text_events

    processed_path = _preprocess_audio(audio_path)

    try:
        # Create initial audio event with ALL required columns
        event = pd.DataFrame([{
            "type": "Audio",
            "filepath": processed_path,
            "start": 0.0,
            "timeline": "default",
            "subject": "default",
            "duration": MAX_AUDIO_DURATION,
            "offset": 0.0,
            "frequency": 1.0,
            "extra": {}
        }])

        # Use TRIBE v2 pipeline with audio_only=True
        events_df = get_audio_and_text_events(event, audio_only=True)

        # Ensure all events have required columns
        required_base_cols = ["type", "filepath", "start", "timeline", "subject", "duration", "offset", "frequency", "extra"]
        
        for idx, row in events_df.iterrows():
            event_type = row.get("type", "")
            for col in required_base_cols:
                if col not in events_df.columns or pd.isna(events_df.at[idx, col]):
                    if col == "timeline":
                        events_df.at[idx, col] = "default"
                    elif col == "subject":
                        events_df.at[idx, col] = "default"
                    elif col == "duration":
                        events_df.at[idx, col] = MAX_AUDIO_DURATION
                    elif col == "offset":
                        events_df.at[idx, col] = 0.0
                    elif col == "frequency":
                        events_df.at[idx, col] = 1.0
                    elif col == "extra":
                        events_df.at[idx, col] = {}
                    elif col == "filepath":
                        if event_type in ["Video", "Audio"]:
                            events_df.at[idx, col] = row.get("filepath", processed_path)
                        else:
                            events_df.at[idx, col] = None

        event_types = events_df['type'].unique().tolist()
        print(f"[SGP-Tribe3] Audio inference: {len(events_df)} events, types: {event_types}", flush=True)

        _metrics["predictions_by_modality"]["audio"] += 1
        return _run_inference_from_events(events_df)

    finally:
        if os.path.exists(processed_path) and processed_path != audio_path:
            os.remove(processed_path)


def _run_text_inference(text: str) -> dict:
    """
    Run inference on text input (text-only modality).
    Creates Word events manually with accumulating context.
    """
    words = text.split()
    if not words:
        raise ValueError("Empty text provided")

    word_events = []
    context = ""
    for i, word in enumerate(words):
        context = f"{context} {word}" if context else word
        word_events.append({
            "type": "Word",
            "start": 0.0,
            "duration": 1.0,
            "text": word,
            "context": context,
            "timeline": "default",
            "subject": "default",
            "sequence_id": 0,
            "sentence": text,
            "language": "english",
            "offset": 0.0,
            "frequency": 1.0,
            "filepath": None,
            "extra": {}
        })

    events_df = pd.DataFrame(word_events)
    
    print(f"[SGP-Tribe3] Text inference: {len(words)} words", flush=True)

    _metrics["predictions_by_modality"]["text"] += 1
    result = _run_inference_from_events(events_df)
    result["text_length"] = len(text)
    result["word_count"] = len(words)

    return result


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "SGP-Tribe3",
        "version": "1.1.0",
        "description": "Sentient Generative Principal — Brain Encoding Calibration System",
        "status": "ok",
        "modality_support": {
            "video": "Video + audio encoding (V-JEPA2 + DINOv2 + Wav2Vec-BERT)",
            "audio": "Audio-only encoding (Wav2Vec-BERT)",
            "text": "Text-only encoding (LLaMA 3.2 embeddings)"
        },
        "endpoints": {
            "GET /health": "Model load status",
            "POST /warmup": "Trigger model loading",
            "POST /predict": "Run inference on video file (multipart/form-data, field: video)",
            "POST /predict_text": "Run inference on text (form field: text)",
            "POST /predict_audio": "Run inference on audio file (field: audio)",
            "GET /nodes": "SGP node definitions",
            "GET /tracts": "White matter tract definitions",
            "GET /results": "All stored stimulus results",
            "GET /coactivation_matrix": "Cross-stimulus co-activation matrix",
            "GET /metrics": "Service metrics and monitoring",
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
    """Run inference on video file (video + audio modalities)."""
    if not _model_loaded:
        return jsonify({
            "error": "Model not loaded. POST to /warmup first.",
            "model_loading": _model_loading,
            "load_error": _model_error,
        }), 503

    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Use multipart/form-data with 'video' field."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    stimulus_id = request.form.get("stimulus_id", str(uuid.uuid4()))
    stimulus_label = request.form.get("label", "unlabeled")
    target_node = request.form.get("target_node", "unknown")

    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        print(f"[SGP-Tribe3] Video inference: stimulus_id={stimulus_id}, label={stimulus_label}", flush=True)
        result = _run_video_inference(tmp_path)

        result["stimulus_id"] = stimulus_id
        result["label"] = stimulus_label
        result["target_node"] = target_node
        result["modality"] = "video"

        _stimulus_results[stimulus_id] = result
        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        err = traceback.format_exc()
        print(f"[SGP-Tribe3] Video inference error:\n{err}", flush=True)
        return jsonify({"error": str(e), "trace": err}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/predict_text", methods=["POST"])
def predict_text():
    """Run inference on plain text input (text-only modality)."""
    if not _model_loaded:
        return jsonify({
            "error": "Model not loaded. POST to /warmup first.",
            "model_loading": _model_loading,
            "load_error": _model_error,
        }), 503

    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided. Use form field 'text'."}), 400

    stimulus_id = request.form.get("stimulus_id", str(uuid.uuid4()))
    stimulus_label = request.form.get("label", "text_stimulus")
    target_node = request.form.get("target_node", "unknown")

    try:
        print(f"[SGP-Tribe3] Text inference: stimulus_id={stimulus_id}, label={stimulus_label}", flush=True)
        result = _run_text_inference(text)

        result["stimulus_id"] = stimulus_id
        result["label"] = stimulus_label
        result["target_node"] = target_node
        result["modality"] = "text"

        _stimulus_results[stimulus_id] = result
        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        err = traceback.format_exc()
        print(f"[SGP-Tribe3] Text inference error:\n{err}", flush=True)
        return jsonify({"error": str(e), "trace": err}), 500


@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    """Run inference on audio file (audio-only modality)."""
    if not _model_loaded:
        return jsonify({
            "error": "Model not loaded. POST to /warmup first.",
            "model_loading": _model_loading,
            "load_error": _model_error,
        }), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Use multipart/form-data with 'audio' field."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    stimulus_id = request.form.get("stimulus_id", str(uuid.uuid4()))
    stimulus_label = request.form.get("label", "audio_stimulus")
    target_node = request.form.get("target_node", "unknown")

    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        print(f"[SGP-Tribe3] Audio inference: stimulus_id={stimulus_id}, label={stimulus_label}", flush=True)
        result = _run_audio_inference(tmp_path)

        result["stimulus_id"] = stimulus_id
        result["label"] = stimulus_label
        result["target_node"] = target_node
        result["modality"] = "audio"

        _stimulus_results[stimulus_id] = result
        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        err = traceback.format_exc()
        print(f"[SGP-Tribe3] Audio inference error:\n{err}", flush=True)
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
    """Compute the co-activation matrix across all stored stimulus results."""
    if len(_stimulus_results) < 2:
        return jsonify({
            "error": "Need at least 2 stimulus results to compute co-activation matrix.",
            "n_results": len(_stimulus_results),
        }), 400

    node_ids = list(SGP_NODE_DEFINITIONS.keys())
    activation_matrix = []
    stimulus_labels = []
    modalities = []

    for sid, res in _stimulus_results.items():
        row = [res["sgp_nodes"].get(nid, 0.0) for nid in node_ids]
        activation_matrix.append(row)
        stimulus_labels.append(res.get("label", sid))
        modalities.append(res.get("modality", "unknown"))

    A = np.array(activation_matrix)

    if A.shape[0] > 1:
        corr_matrix = np.corrcoef(A.T)
    else:
        corr_matrix = np.eye(len(node_ids))

    mean_activation = A.mean(axis=0)

    return jsonify({
        "node_ids": node_ids,
        "n_stimuli": len(_stimulus_results),
        "stimulus_labels": stimulus_labels,
        "modalities": modalities,
        "coactivation_matrix": corr_matrix.round(4).tolist(),
        "mean_activation_per_node": dict(zip(node_ids, mean_activation.round(4).tolist())),
        "interpretation": "coactivation_matrix[i][j] = Pearson correlation of node_i and node_j activation across stimuli. Use as Resonance Graph edge weights.",
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Service metrics following MLOps best practices (MLSysBook Ch 13)."""
    mean_inference_time = 0.0
    if _metrics["inference_times"]:
        mean_inference_time = round(sum(_metrics["inference_times"]) / len(_metrics["inference_times"]), 2)

    uptime_seconds = None
    if _metrics["start_time"]:
        uptime_seconds = (pd.Timestamp.now() - pd.Timestamp(_metrics["start_time"])).total_seconds()

    return jsonify({
        "service_uptime_seconds": uptime_seconds,
        "total_predictions": _metrics["total_predictions"],
        "predictions_by_modality": _metrics["predictions_by_modality"],
        "mean_inference_time_seconds": mean_inference_time,
        "n_stored_results": len(_stimulus_results),
        "model_loaded": _model_loaded,
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=_load_model, daemon=True).start()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
