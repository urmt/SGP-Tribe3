"""
SGP-Tribe3 - Full 1000-Stimulus Research Battery
==================================================
Runs inference on 1000 stimuli (100 per category) with checkpointing.
Fully resumable - if interrupted, picks up where it left off.

Usage:
    python run_full_research_battery.py

Outputs (all NEW):
    - results/full_battery_1000/checkpoint_XXX.json
    - results/full_battery_1000/results.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

import sys
import json
import time
import glob
import numpy as np
from datetime import datetime

import app

RESULTS_DIR = "results/full_battery_1000"
CHECKPOINT_PREFIX = os.path.join(RESULTS_DIR, "checkpoint")
BATCH_SIZE = 50
NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn',
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']


def load_stimulus_bank(path="data/stimulus_bank.json"):
    """Load the 1000-stimulus bank."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['stimuli']


def get_last_checkpoint():
    """Find the last checkpoint file and return completed stimulus IDs."""
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_PREFIX}_*.json"))
    if not checkpoints:
        return [], 0

    last_checkpoint = checkpoints[-1]
    with open(last_checkpoint, 'r') as f:
        data = json.load(f)

    completed = data.get('results', [])
    last_id = data.get('last_stimulus_id', 0)
    print(f"  Resuming from checkpoint: {len(completed)} stimuli completed (last ID: {last_id})")
    return completed, last_id


def save_checkpoint(results, last_id, batch_num):
    """Save checkpoint."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = f"{CHECKPOINT_PREFIX}_{batch_num:03d}.json"

    with open(checkpoint_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'last_stimulus_id': last_id,
            'n_completed': len(results),
            'results': results,
        }, f, indent=2)

    print(f"  Checkpoint saved: {checkpoint_path} ({len(results)} stimuli)")


def run_full_battery():
    """Run the full 1000-stimulus battery."""
    print("=" * 70)
    print("SGP-Tribe3 - Full 1000-Stimulus Research Battery")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load stimulus bank
    print("\n[1/4] Loading stimulus bank...")
    stimuli = load_stimulus_bank()
    print(f"  Loaded {len(stimuli)} stimuli")

    # Load model
    print("\n[2/4] Loading model...")
    app._load_model()
    time.sleep(5)

    if not app._model_loaded:
        print(f"  ERROR: Model failed to load: {app._model_error}")
        return None

    # Force Ollama check
    if not app._ollama_available:
        print("  Forcing Ollama connection check...")
        app._check_ollama_connection()
        if app._ollama_available:
            app._load_adapter()

    print(f"  Model loaded: {app._model_loaded}")
    print(f"  Ollama available: {app._ollama_available}")
    print(f"  Adapter loaded: {app._adapter is not None}")

    # Resume from checkpoint
    print("\n[3/4] Running inference...")
    results, last_id = get_last_checkpoint()
    completed_ids = {r['stimulus_id'] for r in results}

    # Filter remaining stimuli
    remaining = [s for s in stimuli if s['id'] not in completed_ids]
    print(f"  Remaining stimuli: {len(remaining)}/{len(stimuli)}")

    total_start = time.time()
    batch_count = len(results) // BATCH_SIZE

    for i, stimulus in enumerate(remaining):
        print(f"\n  Stimulus {stimulus['id']}/{len(stimuli)} [{stimulus['category']}]: {stimulus['text'][:50]}...")

        try:
            result = app._run_text_inference(stimulus['text'])

            sgp_nodes = result.get('sgp_nodes', {})
            sorted_nodes = sorted([(n, sgp_nodes.get(n, 0)) for n in NODE_ORDER], key=lambda x: x[1], reverse=True)
            actual_top2 = [n[0] for n in sorted_nodes[:2]]

            test_result = {
                "stimulus_id": stimulus['id'],
                "category": stimulus['category'],
                "text": stimulus['text'],
                "expected_dominant": stimulus['expected_dominant'],
                "actual_dominant": actual_top2,
                "hypothesis_support": bool(set(actual_top2) & set(stimulus['expected_dominant'])),
                "sgp_nodes": sgp_nodes,
                "streams": result.get('streams', {}),
                "edge_weights": result.get('edge_weights', {}),
                "dominant_hemisphere": result.get('dominant_hemisphere', 'unknown'),
                "inference_time_seconds": result.get('inference_time_seconds', 0),
                "text_encoder": result.get('text_encoder', 'unknown'),
                "word_count": result.get('word_count', 0),
                "text_length": result.get('text_length', 0),
            }
            results.append(test_result)

            elapsed = time.time() - total_start
            avg_time = elapsed / len(results)
            remaining_time = avg_time * (len(stimuli) - len(results))
            print(f"    Time: {test_result['inference_time_seconds']:.1f}s, "
                  f"Match: {'Y' if test_result['hypothesis_support'] else 'N'}, "
                  f"ETA: {remaining_time/60:.1f}min")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "stimulus_id": stimulus['id'],
                "category": stimulus['category'],
                "text": stimulus['text'],
                "error": str(e),
            })

        # Checkpoint every BATCH_SIZE
        if len(results) % BATCH_SIZE == 0:
            batch_count += 1
            save_checkpoint(results, stimulus['id'], batch_count)

    # Final save
    print("\n[4/4] Saving final results...")
    save_checkpoint(results, stimuli[-1]['id'], batch_count + 1)

    # Combine into final results file
    final_path = os.path.join(RESULTS_DIR, "results.json")
    total_time = time.time() - total_start
    with open(final_path, 'w') as f:
        json.dump({
            "metadata": {
                "phase": "full_battery_1000",
                "date": datetime.now().isoformat(),
                "text_encoder": "ollama" if app._ollama_available else "llama-cpu",
                "adapter_loaded": app._adapter is not None,
                "total_stimuli": len(stimuli),
                "successful": len([r for r in results if 'error' not in r]),
                "failed": len([r for r in results if 'error' in r]),
                "total_time_minutes": round(total_time / 60, 2),
            },
            "results": results,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print("FULL BATTERY COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results: {final_path}")
    print(f"Total: {len(results)}/{len(stimuli)} stimuli")
    print(f"Time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    results = run_full_battery()
    if results:
        print("\nFull research battery complete!")
        sys.exit(0)
    else:
        print("\nFull research battery failed!")
        sys.exit(1)
