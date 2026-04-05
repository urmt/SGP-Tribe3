"""
SGP Stimulus Pipeline
=====================
Runs on your LOCAL machine. Handles:
  1. Video acquisition (local files or YouTube via yt-dlp)
  2. Video preprocessing (trim, format via ffmpeg)
  3. Stimulus generation (for stimuli that need to be created)
  4. Sending to SGP-Tribe3 Space API
  5. Saving and analyzing results

Usage:
    python stimulus_pipeline.py --config stimuli.json --api https://YOUR-SPACE.hf.space

Requirements (install locally):
    pip install yt-dlp requests tqdm
    # ffmpeg must be installed on your system
"""

import os
import json
import time
import argparse
import subprocess
import tempfile
import requests
from pathlib import Path
from typing import Optional


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_API_URL = "https://YOUR-USERNAME-sgp-tribe3.hf.space"

# The 12 stimuli we designed, with YouTube search terms or local file paths
STIMULUS_MANIFEST = [
    # ── Category A: Ventral Stream (Comprehension/Meaning) ──
    {
        "id": "A1_semantic_richness",
        "label": "Semantic Richness — Nature Documentary",
        "target_node": "G2_wernicke",
        "stream": "ventral",
        "source_type": "youtube",
        "youtube_query": "BBC nature documentary narrated David Attenborough ocean 4K",
        "trim_start": 60,   # skip intro
        "trim_duration": 45,
        "notes": "Dense semantic content, simple syntax, naturalistic multimodal"
    },
    {
        "id": "A2_cross_modal_conflict",
        "label": "Cross-Modal Meaning Conflict",
        "target_node": "G8_atl",
        "stream": "ventral",
        "source_type": "generate",
        "generate_type": "conflict_av",
        "visual_query": "calm ocean waves relaxing",
        "audio_query": "busy city street noise",
        "text_overlay": "Everything is still.",
        "duration": 30,
        "notes": "Mismatched visual/audio forces ATL semantic arbitration"
    },
    {
        "id": "A3_abstract_grounding",
        "label": "Abstract Concept Grounding",
        "target_node": "G8_atl",
        "stream": "ventral",
        "source_type": "youtube",
        "youtube_query": "abstract philosophy consciousness mind lecture slow",
        "trim_start": 0,
        "trim_duration": 45,
        "notes": "Abstract language requiring semantic grounding"
    },
    # ── Category B: Dorsal Stream (Production/Phonology) ──
    {
        "id": "B1_phonological_load",
        "label": "Phonological Load — Nonwords",
        "target_node": "G1_broca",
        "stream": "dorsal",
        "source_type": "generate",
        "generate_type": "tts_nonwords",
        "nonwords": [
            "blictrix prenova stelofane",
            "cravontu flistep brenova",
            "spreltic vonamu clistrav",
            "tremfola spivonic blentaru"
        ],
        "duration": 40,
        "notes": "No semantic content — pure phonological encoding"
    },
    {
        "id": "B2_syntactic_complexity",
        "label": "Syntactic Complexity — Embedded Clauses",
        "target_node": "G1_broca",
        "stream": "dorsal",
        "source_type": "generate",
        "generate_type": "tts_complex_syntax",
        "sentences": [
            "The researcher that the committee that the dean appointed reviewed praised the student.",
            "The cat the dog the rat bit chased ran away.",
            "The man who the woman who the child liked admired left early.",
        ],
        "duration": 45,
        "notes": "Deep syntactic embedding taxes Broca/SLF dorsal stream"
    },
    {
        "id": "B3_inner_speech",
        "label": "Inner Speech — First Person Narration",
        "target_node": "G9_premotor",
        "stream": "dorsal",
        "source_type": "youtube",
        "youtube_query": "video diary first person narration daily life vlog talking camera",
        "trim_start": 30,
        "trim_duration": 45,
        "notes": "Self-narration taxes premotor/Broca dorsal pathway"
    },
    # ── Category C: Convergence Hubs ──
    {
        "id": "C1_stream_integration",
        "label": "Dual Stream Integration — Debate Counter-Argument",
        "target_node": "G3_tpj",
        "stream": "convergence",
        "source_type": "youtube",
        "youtube_query": "debate argument listening counterargument formulate response",
        "trim_start": 0,
        "trim_duration": 45,
        "notes": "Forces both streams to operate and converge at TPJ"
    },
    {
        "id": "C2_emotional_semantic",
        "label": "Emotional-Semantic Loading",
        "target_node": "G6_limbic",
        "stream": "modulatory",
        "source_type": "youtube",
        "youtube_query": "emotional personal story narration heartfelt testimonial",
        "trim_start": 0,
        "trim_duration": 45,
        "notes": "Strong emotional prosody + semantic content targets limbic-ATL pathway"
    },
    # ── Category D: Executive and Generative ──
    {
        "id": "D1_veto_conflict",
        "label": "Executive Veto — Conflicting Instructions",
        "target_node": "G4_pfc",
        "stream": "dorsal",
        "source_type": "generate",
        "generate_type": "conflict_instructions",
        "instruction_a": "Look to the left whenever you hear a high tone.",
        "instruction_b": "Look to the right whenever you see a blue circle.",
        "duration": 40,
        "notes": "Simultaneous conflicting instructions tax PFC/cingulum"
    },
    {
        "id": "D2_dmn_resting",
        "label": "Default Mode — Ambient Drift",
        "target_node": "G5_dmn",
        "stream": "generative",
        "source_type": "youtube",
        "youtube_query": "slow drifting clouds timelapse ambient sound no speech 4K",
        "trim_start": 0,
        "trim_duration": 60,
        "notes": "Minimal semantic/phonological content — targets DMN resting state"
    },
    {
        "id": "D3_memory_consolidation",
        "label": "Memory — Autobiographical Resonance",
        "target_node": "G6_limbic",
        "stream": "modulatory",
        "source_type": "youtube",
        "youtube_query": "nostalgic childhood memories narration old home footage family",
        "trim_start": 0,
        "trim_duration": 45,
        "notes": "Autobiographical content targets hippocampal-limbic memory system"
    },
    {
        "id": "D4_full_integration",
        "label": "Full Integration Baseline — Rich Film Clip",
        "target_node": "ALL",
        "stream": "all",
        "source_type": "youtube",
        "youtube_query": "short film emotional story dialogue rich visual audio",
        "trim_start": 0,
        "trim_duration": 90,
        "notes": "All modalities fully engaged. Used to normalize other activation maps."
    },
]


# ─── Acquisition ──────────────────────────────────────────────────────────────

def download_youtube(query: str, output_dir: str, max_duration: int = 120) -> Optional[str]:
    """
    Search YouTube for query and download the best matching video.
    Returns local file path or None on failure.
    """
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--match-filter", f"duration < {max_duration}",
        "--default-search", "ytsearch1:",
        "--output", output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        query,
    ]

    print(f"  [yt-dlp] Searching: {query}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.returncode != 0:
        print(f"  [yt-dlp] Error: {result.stderr[:200]}")
        return None

    # Find the downloaded file
    mp4_files = list(Path(output_dir).glob("*.mp4"))
    if not mp4_files:
        print("  [yt-dlp] No mp4 found after download")
        return None

    return str(sorted(mp4_files, key=os.path.getmtime)[-1])


def trim_video(input_path: str, output_path: str,
               start: int = 0, duration: int = 45) -> str:
    """Trim video to specified segment."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-vf", "scale=320:240",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr[:300]}")
    return output_path


def generate_tts_video(text: str, output_path: str, duration: int = 40) -> str:
    """
    Generate a video with TTS audio and neutral visual background.
    Uses espeak (free, offline) for TTS and ffmpeg for video.
    """
    audio_path = output_path.replace(".mp4", "_audio.wav")

    # TTS via espeak
    tts_cmd = ["espeak", "-w", audio_path, "-s", "140", text]
    result = subprocess.run(tts_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: generate silence
        print("  [TTS] espeak not available, using silence fallback")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r=16000:cl=mono",
            "-t", str(duration), audio_path
        ], capture_output=True)

    # Combine with neutral gray video background
    video_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=gray:size=320x240:rate=25",
        "-i", audio_path,
        "-shortest",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(video_cmd, capture_output=True)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return output_path


# ─── API interaction ──────────────────────────────────────────────────────────

def warmup_api(api_url: str, timeout: int = 300) -> bool:
    """Trigger model warmup and wait until ready."""
    print(f"[API] Warming up model at {api_url}...")

    try:
        requests.post(f"{api_url}/warmup", timeout=10)
    except Exception as e:
        print(f"[API] Warmup trigger failed: {e}")
        return False

    # Poll health endpoint
    for i in range(timeout // 10):
        time.sleep(10)
        try:
            resp = requests.get(f"{api_url}/health", timeout=10)
            data = resp.json()
            status = data.get("status", "")
            print(f"  [{i*10}s] Status: {status}")
            if status == "ready":
                print("[API] Model ready!")
                return True
            if data.get("error"):
                print(f"[API] Load error: {data['error']}")
                return False
        except Exception as e:
            print(f"  [{i*10}s] Health check failed: {e}")

    print("[API] Timeout waiting for model")
    return False


def send_to_api(api_url: str, video_path: str, stimulus: dict) -> Optional[dict]:
    """Upload video to SGP-Tribe3 API and return result."""
    url = f"{api_url}/predict"

    with open(video_path, "rb") as f:
        files = {"video": (os.path.basename(video_path), f, "video/mp4")}
        data = {
            "stimulus_id": stimulus["id"],
            "label": stimulus["label"],
            "target_node": stimulus["target_node"],
        }

        print(f"  [API] Uploading {os.path.getsize(video_path) // 1024}KB...")
        try:
            resp = requests.post(url, files=files, data=data, timeout=180)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  [API] Error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"  [API] Request failed: {e}")
            return None


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(api_url: str, output_dir: str = "./sgp_results",
                 stimuli_filter: Optional[list] = None):
    """
    Run the complete stimulus pipeline:
    1. Acquire/generate each stimulus video
    2. Send to SGP-Tribe3 API
    3. Save results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "activations"), exist_ok=True)

    # Warmup
    if not warmup_api(api_url):
        print("ERROR: Could not warm up API. Check Space is running.")
        return

    results = {}
    stimuli = STIMULUS_MANIFEST
    if stimuli_filter:
        stimuli = [s for s in stimuli if s["id"] in stimuli_filter]

    for i, stimulus in enumerate(stimuli):
        print(f"\n[{i+1}/{len(stimuli)}] Processing: {stimulus['id']}")
        print(f"  Label: {stimulus['label']}")
        print(f"  Target: {stimulus['target_node']} | Stream: {stimulus['stream']}")

        video_path = None
        video_dir = os.path.join(output_dir, "videos")
        final_path = os.path.join(video_dir, f"{stimulus['id']}.mp4")

        # Skip if already processed
        if os.path.exists(final_path):
            print(f"  [SKIP] Video already exists: {final_path}")
            video_path = final_path
        else:
            source_type = stimulus["source_type"]

            if source_type == "local":
                raw_path = stimulus["local_path"]
                trim_video(raw_path, final_path,
                          stimulus.get("trim_start", 0),
                          stimulus.get("trim_duration", 45))
                video_path = final_path

            elif source_type == "youtube":
                with tempfile.TemporaryDirectory() as tmpdir:
                    raw_path = download_youtube(
                        stimulus["youtube_query"], tmpdir,
                        max_duration=stimulus.get("trim_duration", 45) + stimulus.get("trim_start", 0) + 60
                    )
                    if raw_path is None:
                        print(f"  [SKIP] Could not download video for {stimulus['id']}")
                        continue
                    trim_video(raw_path, final_path,
                              stimulus.get("trim_start", 0),
                              stimulus.get("trim_duration", 45))
                video_path = final_path

            elif source_type == "generate":
                gen_type = stimulus["generate_type"]

                if gen_type in ("tts_nonwords", "tts_complex_syntax"):
                    if gen_type == "tts_nonwords":
                        text = " ... ".join(stimulus["nonwords"])
                    else:
                        text = " ... ".join(stimulus["sentences"])
                    generate_tts_video(text, final_path, stimulus.get("duration", 40))
                    video_path = final_path

                elif gen_type in ("conflict_av", "conflict_instructions"):
                    # For complex generation, download two YouTube clips and composite
                    print(f"  [GENERATE] {gen_type} — downloading source clips...")
                    with tempfile.TemporaryDirectory() as tmpdir:
                        q1 = stimulus.get("visual_query", stimulus.get("instruction_a", "abstract video"))
                        clip1 = download_youtube(q1, tmpdir, max_duration=60)
                        if clip1:
                            trim_video(clip1, final_path, 0, stimulus.get("duration", 40))
                            video_path = final_path
                        else:
                            print(f"  [SKIP] Could not generate {stimulus['id']}")
                            continue

        if video_path is None or not os.path.exists(video_path):
            print(f"  [SKIP] No video available")
            continue

        # Send to API
        result = send_to_api(api_url, video_path, stimulus)
        if result is None:
            print(f"  [FAIL] API returned no result")
            continue

        # Save individual result
        result_path = os.path.join(output_dir, "activations", f"{stimulus['id']}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        results[stimulus["id"]] = result
        print(f"  [OK] Result saved: {result_path}")

        # Print node activations
        if "result" in result and "sgp_nodes" in result["result"]:
            nodes = result["result"]["sgp_nodes"]
            top_nodes = sorted(nodes.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top activations: {', '.join(f'{n}={v}' for n, v in top_nodes)}")

    # Save combined results
    combined_path = os.path.join(output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] All results saved to {combined_path}")

    # Fetch and save co-activation matrix
    try:
        resp = requests.get(f"{api_url}/coactivation_matrix", timeout=30)
        if resp.status_code == 200:
            matrix_path = os.path.join(output_dir, "coactivation_matrix.json")
            with open(matrix_path, "w") as f:
                json.dump(resp.json(), f, indent=2)
            print(f"[DONE] Co-activation matrix saved to {matrix_path}")
    except Exception as e:
        print(f"[WARN] Could not fetch co-activation matrix: {e}")

    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGP Stimulus Pipeline")
    parser.add_argument("--api", default=DEFAULT_API_URL,
                        help="SGP-Tribe3 API base URL")
    parser.add_argument("--output", default="./sgp_results",
                        help="Output directory for results")
    parser.add_argument("--stimuli", nargs="*",
                        help="Specific stimulus IDs to run (default: all)")
    parser.add_argument("--local-video", 
                        help="Use a local video file for a single stimulus test")
    parser.add_argument("--stimulus-id", default="local_test",
                        help="Stimulus ID for local video test")

    args = parser.parse_args()

    if args.local_video:
        # Quick test with a local video file
        print(f"[TEST] Sending local video: {args.local_video}")
        warmup_api(args.api)
        stimulus = {
            "id": args.stimulus_id,
            "label": f"Local test: {os.path.basename(args.local_video)}",
            "target_node": "unknown",
        }
        result = send_to_api(args.api, args.local_video, stimulus)
        if result:
            print(json.dumps(result, indent=2))
    else:
        run_pipeline(args.api, args.output, args.stimuli)
