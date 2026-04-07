"""
SGP-Tribe3 — Phase 1 Test Battery Runner
=========================================
Runs inference on 5 minimal texts and collects brain activation data.
Tests the Hickok-Poeppel dual-stream hypothesis.

Usage:
    python run_test_battery.py
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

# Import app components
import app

# Test battery from GAMEPLAN.md
TEST_BATTERY = [
    {
        "id": 1,
        "text": "The cat sat on the mat.",
        "expected_dominant": ["G2_wernicke", "G7_sensory"],
        "purpose": "Baseline simple sentence"
    },
    {
        "id": 2,
        "text": "If P implies Q, and Q implies R, then P implies R.",
        "expected_dominant": ["G4_pfc", "G1_broca"],
        "purpose": "Logical structure"
    },
    {
        "id": 3,
        "text": "She felt the warmth of the sun on her skin as memories of childhood flooded back.",
        "expected_dominant": ["G6_limbic", "G5_dmn"],
        "purpose": "Emotional + sensory"
    },
    {
        "id": 4,
        "text": "The mitochondria is the powerhouse of the cell.",
        "expected_dominant": ["G4_pfc", "G7_sensory"],
        "purpose": "Factual/technical"
    },
    {
        "id": 5,
        "text": "What if the universe is a simulation and we're just characters in someone else's dream?",
        "expected_dominant": ["G5_dmn", "G3_tpj"],
        "purpose": "Self-referential/abstract"
    }
]

def run_test_battery():
    """Run the Phase 1 test battery."""
    print("=" * 80)
    print("SGP-Tribe3 — Phase 1 Test Battery")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Text encoder: {'ollama' if app._ollama_available else 'llama-cpu'}")
    print(f"Number of tests: {len(TEST_BATTERY)}")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model...")
    app._load_model()
    time.sleep(5)
    
    if not app._model_loaded:
        print(f"ERROR: Model failed to load: {app._model_error}")
        return None
    
    print(f"Model loaded successfully")
    print(f"Ollama available: {app._ollama_available}")
    print(f"Adapter loaded: {app._adapter is not None}")
    
    # Force Ollama check if not already done
    if not app._ollama_available:
        print("Forcing Ollama connection check...")
        app._check_ollama_connection()
        if app._ollama_available:
            app._load_adapter()
        print(f"Ollama available after check: {app._ollama_available}")
    
    # Run tests
    print("\n[2/3] Running test battery...")
    results = []
    total_start = time.time()
    
    for test in TEST_BATTERY:
        print(f"\n{'─' * 60}")
        print(f"Test {test['id']}: {test['purpose']}")
        print(f"Text: \"{test['text']}\"")
        print(f"Expected dominant nodes: {', '.join(test['expected_dominant'])}")
        print(f"{'─' * 60}")
        
        try:
            result = app._run_text_inference(test['text'])
            
            # Collect key metrics
            sgp_nodes = result.get('sgp_nodes', {})
            streams = result.get('streams', {})
            edge_weights = result.get('edge_weights', {})
            dominant = result.get('dominant_hemisphere', 'unknown')
            inference_time = result.get('inference_time_seconds', 0)
            text_encoder = result.get('text_encoder', 'unknown')
            
            # Find actual dominant nodes
            sorted_nodes = sorted(sgp_nodes.items(), key=lambda x: x[1], reverse=True)
            actual_dominant = [n[0] for n in sorted_nodes[:2]]
            
            print(f"\nResults:")
            print(f"  Text encoder: {text_encoder}")
            print(f"  Inference time: {inference_time}s")
            print(f"  Dominant hemisphere: {dominant}")
            print(f"\n  SGP Node Activations:")
            for node, value in sorted_nodes:
                bar = '█' * int(value * 20)
                print(f"    {node:15s}: {value:.4f} {bar}")
            
            print(f"\n  Stream Activations:")
            for stream, value in streams.items():
                print(f"    {stream:15s}: {value:.4f}")
            
            print(f"\n  Edge Weights:")
            for edge, value in edge_weights.items():
                print(f"    {edge:15s}: {value:.4f}")
            
            print(f"\n  Actual dominant nodes: {', '.join(actual_dominant)}")
            print(f"  Expected dominant nodes: {', '.join(test['expected_dominant'])}")
            
            # Check if hypothesis is supported
            matches = set(actual_dominant) & set(test['expected_dominant'])
            hypothesis_support = len(matches) > 0
            
            print(f"  Hypothesis support: {'✓ PARTIAL MATCH' if matches else '✗ NO MATCH'} ({len(matches)}/2 nodes)")
            
            # Store result
            test_result = {
                "test_id": test['id'],
                "text": test['text'],
                "purpose": test['purpose'],
                "expected_dominant": test['expected_dominant'],
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
                "activation_timeline": result.get('activation_timeline', []),
            }
            results.append(test_result)
            
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            print(traceback.format_exc())
            results.append({
                "test_id": test['id'],
                "text": test['text'],
                "purpose": test['purpose'],
                "error": str(e),
            })
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\nTotal tests: {len(TEST_BATTERY)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per test: {total_time/len(TEST_BATTERY):.1f}s")
    
    if successful:
        hypothesis_supports = [r for r in successful if r.get('hypothesis_support')]
        print(f"\nHypothesis support: {len(hypothesis_supports)}/{len(successful)} tests ({len(hypothesis_supports)/len(successful)*100:.0f}%)")
        
        # Node activation summary
        print(f"\nNode Activation Summary (mean across tests):")
        all_nodes = {}
        for r in successful:
            for node, value in r['sgp_nodes'].items():
                if node not in all_nodes:
                    all_nodes[node] = []
                all_nodes[node].append(value)
        
        for node in sorted(all_nodes.keys()):
            values = all_nodes[node]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {node:15s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Stream summary
        print(f"\nStream Activation Summary (mean across tests):")
        all_streams = {}
        for r in successful:
            for stream, value in r['streams'].items():
                if stream not in all_streams:
                    all_streams[stream] = []
                all_streams[stream].append(value)
        
        for stream in sorted(all_streams.keys()):
            values = all_streams[stream]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {stream:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"phase1_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "phase": 1,
                "date": datetime.now().isoformat(),
                "text_encoder": "ollama" if app._ollama_available else "llama-cpu",
                "adapter_loaded": app._adapter is not None,
                "total_tests": len(TEST_BATTERY),
                "successful_tests": len(successful),
                "failed_tests": len(failed),
                "total_time_seconds": round(total_time, 2),
            },
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_test_battery()
    if results:
        print("\n✅ Phase 1 test battery complete!")
        sys.exit(0)
    else:
        print("\n❌ Phase 1 test battery failed!")
        sys.exit(1)
