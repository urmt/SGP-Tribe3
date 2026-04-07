"""
SGP-Tribe3 — Main API Application
=================================
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
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

import sys
import warnings
import threading
import traceback
import tempfile
import subprocess
import uuid
import math
import json
import os
import numpy as np
import pandas as pd

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from sgp_parcellation import get_parcellator, SGP_NODE_DEFINITIONS, SGP_TRACT_DEFINITIONS

app = Flask(__name__)

_model = None
_model_loaded = False
_model_loading = False
_model_error = None
_model_lock = threading.Lock()

_ollama_client = None
_ollama_available = False
_adapter = None

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

_stimulus_results = {}

_metrics = {
    "start_time": None,
    "total_predictions": 0,
    "predictions_by_modality": {"video": 0, "audio": 0, "text": 0},
    "inference_times": [],
}


def _check_ollama_connection():
    """Check if Ollama is running and return client if available."""
    global _ollama_client, _ollama_available
    if not OLLAMA_AVAILABLE:
        print("[SGP-Tribe3] Ollama: package not installed", flush=True)
        return False
    try:
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        client.list()
        _ollama_client = client
        _ollama_available = True
        print("[SGP-Tribe3] Ollama: connected", flush=True)
        return True
    except Exception as e:
        print(f"[SGP-Tribe3] Ollama: not available ({e})", flush=True)
        return False


def _load_adapter():
    """Load the embedding adapter if available."""
    global _adapter
    import torch
    import torch.nn as nn
    
    adapter_path = os.environ.get("ADAPTER_PATH", "adapter_weights.pt")
    if not os.path.exists(adapter_path):
        print(f"[SGP-Tribe3] Adapter: not found at {adapter_path}", flush=True)
        return False
    
    try:
        class EmbeddingAdapter(nn.Module):
            def __init__(self, input_dim=4096, target_dim=9216, hidden_dim=8192):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, target_dim),
                )
            def forward(self, x):
                return self.network(x)
        
        adapter = EmbeddingAdapter()
        adapter.load_state_dict(torch.load(adapter_path, map_location='cpu', weights_only=True))
        adapter.eval()
        _adapter = adapter
        print(f"[SGP-Tribe3] Adapter: loaded from {adapter_path}", flush=True)
        return True
    except Exception as e:
        print(f"[SGP-Tribe3] Adapter: failed to load ({e})", flush=True)
        return False


def _get_ollama_embeddings(texts):
    """Get embeddings from Ollama for a list of texts.
    
    Returns:
        numpy array of shape (n_texts, 9216) after adapter, or (n_texts, 4096) raw
    """
    global _ollama_client, _adapter
    
    if _ollama_client is None:
        raise RuntimeError("Ollama client not available")
    
    model = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
    try:
        response = _ollama_client.embed(
            model=model,
            input=texts,
            truncate=True
        )
        embeddings = np.array(response['embeddings'], dtype=np.float32)
        
        # Apply adapter if available
        if _adapter is not None:
            import torch
            with torch.no_grad():
                embedding_tensor = torch.from_numpy(embeddings)
                adapted = _adapter(embedding_tensor).numpy()
            return adapted
        
        # Fallback: zero-pad to 9216
        if embeddings.shape[1] < 9216:
            padded = np.zeros((len(texts), 9216), dtype=np.float32)
            padded[:, :embeddings.shape[1]] = embeddings
            return padded
        
        return embeddings[:, :9216]
        
    except Exception as e:
        raise RuntimeError(f"Ollama embedding failed: {e}")


def _run_text_inference_ollama(text: str) -> dict:
    """Run text inference using Ollama embeddings.
    
    This creates a standalone pipeline:
    1. Gets Ollama embeddings for the full text
    2. Uses a simple projection to fMRI vertex space
    3. Returns SGP parcellation results
    """
    import time
    start_time = time.time()
    
    words = text.split()
    if not words:
        raise ValueError("Empty text provided")
    
    print(f"[SGP-Tribe3] Text inference (Ollama): {len(words)} words", flush=True)
    
    # Get Ollama embedding for the full text
    print(f"[SGP-Tribe3] Getting Ollama embedding...", flush=True)
    embeddings = _get_ollama_embeddings([text])  # Single embedding for full text
    print(f"[SGP-Tribe3] Ollama embedding shape: {embeddings.shape}", flush=True)
    
    # Project to fMRI vertex space (20,484 vertices)
    # Use the model's text projector if available, otherwise use a simple approach
    import torch
    
    # Get the text projector from the model
    text_projector = None
    try:
        # The model has a brain_model_config with projectors
        # We'll use a simple approach: normalize and scale
        embedding = embeddings[0]  # (9216,)
        
        # Normalize the embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Use the model's encoder to project if possible
        # The model expects input in a specific format
        # For now, use a simple random projection as placeholder
        n_vertices = 20484
        np.random.seed(42)  # Fixed seed for reproducibility
        projection_matrix = np.random.randn(9216, n_vertices).astype(np.float32) * 0.01
        
        # Project to vertex space
        vertex_activations = embedding @ projection_matrix  # (20484,)
        
        # Normalize to reasonable range
        vertex_activations = vertex_activations / (np.abs(vertex_activations).max() + 1e-8)
        
        # Reshape to (1, 20484) for parcellation
        pred_array = vertex_activations.reshape(1, -1)
        
    except Exception as e:
        import traceback
        print(f"[SGP-Tribe3] Projection error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise RuntimeError(f"Failed to project embeddings to fMRI space: {e}")
    
    inference_time = time.time() - start_time
    print(f"[SGP-Tribe3] Ollama prediction shape: {pred_array.shape}, time: {inference_time:.1f}s", flush=True)
    
    # Apply SGP parcellation
    parcellator = get_parcellator(CACHE_DIR)
    result = parcellator.parcellate(pred_array)
    
    result["activation_timeline"] = [
        round(float(np.abs(pred_array[t]).mean()), 4)
        for t in range(pred_array.shape[0])
    ]
    
    result["inference_time_seconds"] = round(inference_time, 2)
    result["n_segments"] = pred_array.shape[0]
    result["n_vertices"] = pred_array.shape[1]
    result["text_length"] = len(text)
    result["word_count"] = len(words)
    result["text_encoder"] = "ollama"
    
    # Update metrics
    _metrics["total_predictions"] += 1
    _metrics["inference_times"].append(inference_time)
    if len(_metrics["inference_times"]) > 100:
        _metrics["inference_times"] = _metrics["inference_times"][-100:]
    
    return result


def _run_text_inference(text: str) -> dict:
    """Run inference on text input (text-only modality).
    
    Uses Ollama if available, falls back to LLaMA on CPU.
    """
    if _ollama_available:
        return _run_text_inference_ollama(text)
    
    # Fallback: use LLaMA on CPU via TRIBE v2
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
    
    print(f"[SGP-Tribe3] Text inference (LLaMA CPU fallback): {len(words)} words", flush=True)

    _metrics["predictions_by_modality"]["text"] += 1
    result = _run_inference_from_events(events_df)
    result["text_length"] = len(text)
    result["word_count"] = len(words)
    result["text_encoder"] = "llama-cpu"

    return result


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
        print(f"[SGP-Tribe3] CUDA available: {torch.cuda.is_available()}", flush=True)

        print("[SGP-Tribe3] Loading TribeModel...", flush=True)
        from tribev2 import TribeModel
        model = TribeModel.from_pretrained(CKPT, device='cpu')
        print("[SGP-Tribe3] TribeModel loaded!", flush=True)

        # Check Ollama for fast text inference
        _check_ollama_connection()
        if _ollama_available:
            _load_adapter()

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


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    import json
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except:
            pass
    return 0.0


def _preprocess_video(video_path: str, max_duration: int = MAX_VIDEO_DURATION) -> str:
    """Trim video to max_duration and normalize to TRIBE v2 expected format."""
    actual_duration = _get_video_duration(video_path)
    clip_duration = min(max_duration, actual_duration) if actual_duration > 0 else max_duration
    
    output_path = video_path.replace(".mp4", "_processed.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", str(clip_duration),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-vf", "scale=320:240",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg preprocessing failed: {result.stderr}")

    return output_path


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    import json
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except:
            pass
    return 0.0


def _preprocess_audio(audio_path: str, max_duration: int = MAX_AUDIO_DURATION) -> str:
    """Convert audio to wav format and normalize for TRIBE v2."""
    actual_duration = _get_audio_duration(audio_path)
    clip_duration = min(max_duration, actual_duration) if actual_duration > 0 else max_duration
    
    output_path = audio_path.replace(audio_path.split(".")[-1], "wav")
    if output_path == audio_path:
        output_path = audio_path.rsplit(".", 1)[0] + "_processed.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-t", str(clip_duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg audio preprocessing failed: {result.stderr}")

    return output_path


def _run_inference_from_events(events_df: pd.DataFrame) -> dict:
    """Run TRIBE v2 inference from an events DataFrame and return SGP parcellation."""
    import time
    start_time = time.time()

    try:
        preds, segments = _model.predict(events=events_df, verbose=False)

        if hasattr(preds, "numpy"):
            pred_array = preds.numpy()
        else:
            pred_array = np.array(preds)

        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(1, -1)

        inference_time = time.time() - start_time
        print(f"[SGP-Tribe3] Prediction shape: {pred_array.shape}, time: {inference_time:.1f}s", flush=True)

        parcellator = get_parcellator(CACHE_DIR)
        result = parcellator.parcellate(pred_array)

        result["activation_timeline"] = [
            round(float(np.abs(pred_array[t]).mean()), 4)
            for t in range(pred_array.shape[0])
        ]

        result["inference_time_seconds"] = round(inference_time, 2)
        result["n_segments"] = pred_array.shape[0]
        result["n_vertices"] = pred_array.shape[1]

        _metrics["total_predictions"] += 1
        _metrics["inference_times"].append(inference_time)
        if len(_metrics["inference_times"]) > 100:
            _metrics["inference_times"] = _metrics["inference_times"][-100:]

        return result

    except Exception as e:
        raise RuntimeError(f"TRIBE v2 inference failed: {e}")


def _run_video_inference(video_path: str) -> dict:
    """Run inference on video file (video + audio modalities)."""
    from tribev2.demo_utils import get_audio_and_text_events

    processed_path = _preprocess_video(video_path)
    actual_duration = _get_video_duration(processed_path)
    clip_duration = int(actual_duration) if actual_duration > 0 else MAX_VIDEO_DURATION
    
    try:
        event = pd.DataFrame([{
            "type": "Video",
            "filepath": processed_path,
            "start": 0.0,
            "timeline": "default",
            "subject": "default",
            "duration": clip_duration,
            "offset": 0.0,
            "frequency": 1.0,
            "extra": {}
        }])
        
        events_df = get_audio_and_text_events(event, audio_only=True)

        if "timeline" not in events_df.columns:
            events_df["timeline"] = "default"
        events_df["timeline"] = events_df["timeline"].fillna("default")
        
        if "subject" not in events_df.columns:
            events_df["subject"] = "default"
        events_df["subject"] = events_df["subject"].fillna("default")
        
        if "duration" not in events_df.columns:
            events_df["duration"] = MAX_VIDEO_DURATION
        events_df["duration"] = events_df["duration"].fillna(MAX_VIDEO_DURATION)
        
        if "offset" not in events_df.columns:
            events_df["offset"] = 0.0
        events_df["offset"] = events_df["offset"].fillna(0.0)
        
        if "frequency" not in events_df.columns:
            events_df["frequency"] = 1.0
        events_df["frequency"] = events_df["frequency"].fillna(1.0)
        
        if "extra" not in events_df.columns:
            events_df["extra"] = {}
        events_df["extra"] = events_df["extra"].apply(lambda x: x if x is not None else {})

        event_types = events_df['type'].unique().tolist()
        print(f"[SGP-Tribe3] Video inference: {len(events_df)} events, types: {event_types}", flush=True)

        _metrics["predictions_by_modality"]["video"] += 1
        return _run_inference_from_events(events_df)

    finally:
        if os.path.exists(processed_path) and processed_path != video_path:
            os.remove(processed_path)


def _run_audio_inference(audio_path: str) -> dict:
    """Run inference on audio file (audio-only modality)."""
    from tribev2.demo_utils import get_audio_and_text_events

    processed_path = _preprocess_audio(audio_path)
    actual_duration = _get_audio_duration(processed_path)
    clip_duration = int(actual_duration) if actual_duration > 0 else MAX_AUDIO_DURATION
    
    try:
        event = pd.DataFrame([{
            "type": "Audio",
            "filepath": processed_path,
            "start": 0.0,
            "timeline": "default",
            "subject": "default",
            "duration": clip_duration,
            "offset": 0.0,
            "frequency": 1.0,
            "extra": {}
        }])

        events_df = get_audio_and_text_events(event, audio_only=True)

        if "timeline" not in events_df.columns:
            events_df["timeline"] = "default"
        events_df["timeline"] = events_df["timeline"].fillna("default")
        
        if "subject" not in events_df.columns:
            events_df["subject"] = "default"
        events_df["subject"] = events_df["subject"].fillna("default")
        
        if "duration" not in events_df.columns:
            events_df["duration"] = MAX_AUDIO_DURATION
        events_df["duration"] = events_df["duration"].fillna(MAX_AUDIO_DURATION)
        
        if "offset" not in events_df.columns:
            events_df["offset"] = 0.0
        events_df["offset"] = events_df["offset"].fillna(0.0)
        
        if "frequency" not in events_df.columns:
            events_df["frequency"] = 1.0
        events_df["frequency"] = events_df["frequency"].fillna(1.0)
        
        if "extra" not in events_df.columns:
            events_df["extra"] = {}
        events_df["extra"] = events_df["extra"].apply(lambda x: x if x is not None else {})

        event_types = events_df['type'].unique().tolist()
        print(f"[SGP-Tribe3] Audio inference: {len(events_df)} events, types: {event_types}", flush=True)

        _metrics["predictions_by_modality"]["audio"] += 1
        return _run_inference_from_events(events_df)

    finally:
        if os.path.exists(processed_path) and processed_path != audio_path:
            os.remove(processed_path)


def _run_text_inference(text: str) -> dict:
    """Run inference on text input (text-only modality)."""
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


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "SGP-Tribe3",
        "version": "1.3.0",
        "description": "Sentient Generative Principal — Brain Encoding Calibration System",
        "status": "ok",
        "modality_support": {
            "video": "Video + audio encoding (V-JEPA2 + DINOv2 + Wav2Vec-BERT)",
            "audio": "Audio-only encoding (Wav2Vec-BERT)",
            "text": "Text-only encoding (Ollama embeddings + adapter, or LLaMA 3.2 CPU fallback)"
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
        "ollama_available": _ollama_available,
        "adapter_loaded": _adapter is not None,
        "text_encoder": "ollama" if _ollama_available else "llama-cpu",
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


if __name__ == "__main__":
    threading.Thread(target=_load_model, daemon=True).start()
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)