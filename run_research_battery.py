"""
SGP-Tribe3 — Research Text Battery (30 stimuli, 3 per category)
===============================================================
Focused battery for journal article production.
Tests the Hickok-Poeppel dual-stream hypothesis.

Usage:
    python run_research_battery.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import app

# 30 stimuli - 3 per category, carefully selected
TEXT_BATTERY = [
    # Simple (baseline)
    {"id": 1, "category": "simple", "text": "The cat sat on the mat.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 2, "category": "simple", "text": "The dog ran across the yard.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 3, "category": "simple", "text": "Rain falls on the green grass.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    
    # Logical
    {"id": 4, "category": "logical", "text": "If P implies Q, and Q implies R, then P implies R.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 5, "category": "logical", "text": "All humans are mortal. Socrates is human. Therefore Socrates is mortal.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 6, "category": "logical", "text": "If it rains, the ground gets wet. It is raining. Therefore the ground is wet.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    
    # Emotional
    {"id": 7, "category": "emotional", "text": "She felt the warmth of the sun on her skin as memories of childhood flooded back.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 8, "category": "emotional", "text": "Tears of joy streamed down her face when she saw her family again.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 9, "category": "emotional", "text": "The grief hit him like a wave when he heard the devastating news.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    
    # Factual
    {"id": 10, "category": "factual", "text": "The mitochondria is the powerhouse of the cell.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 11, "category": "factual", "text": "DNA contains the genetic instructions for the development of all living organisms.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 12, "category": "factual", "text": "The human brain contains approximately eighty-six billion neurons.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    
    # Abstract
    {"id": 13, "category": "abstract", "text": "What if the universe is a simulation and we're just characters in someone else's dream?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 14, "category": "abstract", "text": "If consciousness emerges from complexity, at what point does a system become aware?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 15, "category": "abstract", "text": "Perhaps free will is an illusion created by the brain's predictive mechanisms.", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    
    # Spatial
    {"id": 16, "category": "spatial", "text": "The mountain peak rose sharply above the clouds, casting a long shadow over the valley.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 17, "category": "spatial", "text": "The spiral staircase wound upward in a perfect golden ratio.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 18, "category": "spatial", "text": "Stars scattered across the night sky like diamonds on black velvet.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    
    # Social
    {"id": 19, "category": "social", "text": "She knew he was lying, but chose to pretend she believed him.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 20, "category": "social", "text": "He wondered what she was thinking as she stared out the window.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 21, "category": "social", "text": "The diplomat carefully considered how each nation would interpret the proposal.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    
    # Motor
    {"id": 22, "category": "motor", "text": "She grasped the heavy stone and hurled it across the river.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 23, "category": "motor", "text": "The dancer leaped across the stage with graceful precision.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 24, "category": "motor", "text": "The surgeon's hands moved with practiced precision during the operation.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    
    # Memory
    {"id": 25, "category": "memory", "text": "She remembered the day they first met as if it were yesterday.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 26, "category": "memory", "text": "The scent of fresh bread triggered vivid memories of his grandmother's kitchen.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 27, "category": "memory", "text": "The old journal revealed secrets from a century ago.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    
    # Auditory
    {"id": 28, "category": "auditory", "text": "The melody drifted through the air like a gentle stream of sound.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 29, "category": "auditory", "text": "The orchestra swelled to a crescendo as the final movement began.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 30, "category": "auditory", "text": "The violin's high notes pierced the silence with crystalline clarity.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
]


def run_research_battery():
    """Run the research text battery."""
    print("=" * 80)
    print("SGP-Tribe3 — Research Text Battery (30 Stimuli)")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Text encoder: {'ollama' if app._ollama_available else 'llama-cpu'}")
    print(f"Number of stimuli: {len(TEXT_BATTERY)}")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model...")
    app._load_model()
    time.sleep(5)
    
    if not app._model_loaded:
        print(f"ERROR: Model failed to load: {app._model_error}")
        return None
    
    if not app._ollama_available:
        print("Forcing Ollama connection check...")
        app._check_ollama_connection()
        if app._ollama_available:
            app._load_adapter()
    
    print(f"Model loaded: {app._model_loaded}")
    print(f"Ollama available: {app._ollama_available}")
    print(f"Adapter loaded: {app._adapter is not None}")
    
    # Run tests
    print("\n[2/3] Running research battery...")
    results = []
    total_start = time.time()
    
    for i, stimulus in enumerate(TEXT_BATTERY):
        print(f"\n{'─' * 60}")
        print(f"Stimulus {stimulus['id']}/{len(TEXT_BATTERY)} [{stimulus['category']}]: {stimulus['text'][:60]}...")
        
        try:
            result = app._run_text_inference(stimulus['text'])
            
            sgp_nodes = result.get('sgp_nodes', {})
            streams = result.get('streams', {})
            edge_weights = result.get('edge_weights', {})
            dominant = result.get('dominant_hemisphere', 'unknown')
            inference_time = result.get('inference_time_seconds', 0)
            text_encoder = result.get('text_encoder', 'unknown')
            
            sorted_nodes = sorted(sgp_nodes.items(), key=lambda x: x[1], reverse=True)
            actual_dominant = [n[0] for n in sorted_nodes[:2]]
            
            matches = set(actual_dominant) & set(stimulus['expected_dominant'])
            hypothesis_support = len(matches) > 0
            
            print(f"  Encoder: {text_encoder}, Time: {inference_time:.1f}s")
            print(f"  Top nodes: {', '.join(actual_dominant)}")
            print(f"  Expected: {', '.join(stimulus['expected_dominant'])}")
            print(f"  Match: {'✓' if matches else '✗'}")
            
            test_result = {
                "stimulus_id": stimulus['id'],
                "category": stimulus['category'],
                "text": stimulus['text'],
                "expected_dominant": stimulus['expected_dominant'],
                "actual_dominant": actual_dominant,
                "hypothesis_support": hypothesis_support,
                "matches": list(matches),
                "sgp_nodes": sgp_nodes,
                "streams": streams,
                "edge_weights": edge_weights,
                "dominant_hemisphere": dominant,
                "inference_time_seconds": inference_time,
                "text_encoder": text_encoder,
                "word_count": result.get('word_count', 0),
                "text_length": result.get('text_length', 0),
            }
            results.append(test_result)
            
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            results.append({
                "stimulus_id": stimulus['id'],
                "category": stimulus['category'],
                "text": stimulus['text'],
                "error": str(e),
            })
        
        # Progress
        if (i + 1) % 5 == 0:
            elapsed = time.time() - total_start
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(TEXT_BATTERY) - i - 1)
            print(f"\n  Progress: {i+1}/{len(TEXT_BATTERY)} ({(i+1)/len(TEXT_BATTERY)*100:.0f}%)")
            print(f"  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("RESEARCH BATTERY SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\nTotal stimuli: {len(TEXT_BATTERY)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per stimulus: {total_time/len(TEXT_BATTERY):.1f}s")
    
    if successful:
        hypothesis_supports = [r for r in successful if r.get('hypothesis_support')]
        print(f"\nHypothesis support: {len(hypothesis_supports)}/{len(successful)} ({len(hypothesis_supports)/len(successful)*100:.0f}%)")
        
        # Category analysis
        print(f"\nCategory Analysis:")
        categories = {}
        for r in successful:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'nodes': {}, 'supports': 0, 'total': 0}
            categories[cat]['total'] += 1
            if r.get('hypothesis_support'):
                categories[cat]['supports'] += 1
            for node, value in r['sgp_nodes'].items():
                if node not in categories[cat]['nodes']:
                    categories[cat]['nodes'][node] = []
                categories[cat]['nodes'][node].append(value)
        
        for cat in sorted(categories.keys()):
            data = categories[cat]
            support_rate = data['supports'] / data['total'] * 100
            print(f"\n  {cat.upper()} ({data['total']} stimuli, {support_rate:.0f}% support):")
            for node in sorted(data['nodes'].keys()):
                values = data['nodes'][node]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"    {node:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"research_battery_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "phase": "research_battery",
                "date": datetime.now().isoformat(),
                "text_encoder": "ollama" if app._ollama_available else "llama-cpu",
                "adapter_loaded": app._adapter is not None,
                "total_stimuli": len(TEXT_BATTERY),
                "successful": len(successful),
                "failed": len(failed),
                "total_time_minutes": round(total_time / 60, 2),
            },
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_research_battery()
    if results:
        print("\n✅ Research battery complete!")
        sys.exit(0)
    else:
        print("\n❌ Research battery failed!")
        sys.exit(1)
