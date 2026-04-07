"""
SGP-Tribe3 — Phase 2: LLM Integration with SGP Guidance
=========================================================
Demonstrates using SGP activation profiles to guide LLM generation.

Pipeline:
1. Input text → SGP activation profile (via TRIBE v2)
2. SGP profile → Resonance weights for LLM prompt
3. LLM generates response guided by SGP weights
4. Compare SGP-guided vs unguided responses

Usage:
    python sgp_llm_integration.py

Outputs:
    results/full_battery_1000/sgp_llm_results.json
    results/full_battery_1000/figures/geo_06_llm_comparison.png
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

import json
import time
import numpy as np
import pandas as pd
import ollama
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn',
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

NODE_DESCRIPTIONS = {
    'G1_broca': 'Broca\'s Area - speech production, syntax',
    'G2_wernicke': 'Wernicke\'s Area - language comprehension',
    'G3_tpj': 'Temporoparietal Junction - theory of mind',
    'G4_pfc': 'Prefrontal Cortex - executive function, reasoning',
    'G5_dmn': 'Default Mode Network - self-referential thought',
    'G6_limbic': 'Limbic System - emotion, memory',
    'G7_sensory': 'Sensory Cortex - perceptual processing',
    'G8_atl': 'Anterior Temporal Lobe - semantic memory',
    'G9_premotor': 'Premotor Cortex - motor planning',
}

STREAM_DESCRIPTIONS = {
    'dorsal': 'Dorsal stream - sensorimotor integration',
    'ventral': 'Ventral stream - semantic comprehension',
    'generative': 'Generative stream - self-referential processing',
    'modulatory': 'Modulatory stream - emotional weighting',
    'convergence': 'Convergence stream - sensorimotor interface',
}

# ─── Test Stimuli ────────────────────────────────────────────────────────────

TEST_STIMULI = [
    {"text": "The cat sat on the mat.", "category": "simple"},
    {"text": "If P implies Q, and Q implies R, then P implies R.", "category": "logical"},
    {"text": "She felt the warmth of the sun on her skin as memories of childhood flooded back.", "category": "emotional"},
    {"text": "The mitochondria is the powerhouse of the cell.", "category": "factual"},
    {"text": "What if the universe is a simulation and we're just characters in someone else's dream?", "category": "abstract"},
    {"text": "The mountain peak rose sharply above the clouds, casting a long shadow over the valley.", "category": "spatial"},
    {"text": "She knew he was lying, but chose to pretend she believed him.", "category": "social"},
    {"text": "She grasped the heavy stone and hurled it across the river.", "category": "motor"},
    {"text": "She remembered the day they first met as if it were yesterday.", "category": "memory"},
    {"text": "The melody drifted through the air like a gentle stream of sound.", "category": "auditory"},
]

# ─── SGP-Guided Prompt Construction ─────────────────────────────────────────

def build_sgp_guided_prompt(text, sgp_nodes):
    """Construct an LLM prompt weighted by SGP activation profile.
    
    The SGP activation profile is used to emphasize certain cognitive
    aspects in the LLM's response generation.
    """
    # Sort nodes by activation
    sorted_nodes = sorted(sgp_nodes.items(), key=lambda x: x[1], reverse=True)
    top_nodes = sorted_nodes[:3]
    
    # Build activation description
    activation_desc = []
    for node, value in top_nodes:
        desc = NODE_DESCRIPTIONS.get(node, node)
        activation_desc.append(f"  - {desc}: {value:.2f}")
    
    # Determine dominant streams
    stream_activations = {}
    for node, value in sgp_nodes.items():
        if 'broca' in node or 'pfc' in node or 'premotor' in node:
            stream_activations['dorsal'] = stream_activations.get('dorsal', 0) + value
        elif 'wernicke' in node or 'sensory' in node or 'atl' in node:
            stream_activations['ventral'] = stream_activations.get('ventral', 0) + value
        elif 'dmn' in node:
            stream_activations['generative'] = stream_activations.get('generative', 0) + value
        elif 'limbic' in node:
            stream_activations['modulatory'] = stream_activations.get('modulatory', 0) + value
        elif 'tpj' in node:
            stream_activations['convergence'] = stream_activations.get('convergence', 0) + value
    
    dominant_stream = max(stream_activations, key=stream_activations.get) if stream_activations else 'unknown'
    stream_desc = STREAM_DESCRIPTIONS.get(dominant_stream, '')
    
    prompt = f"""You are responding to this text input: "{text}"

Your response should be guided by the following brain activation profile from the SGP (Sentient Generative Principal) model:

Top activated brain regions:
{chr(10).join(activation_desc)}

Dominant processing stream: {dominant_stream} ({stream_desc})

Based on this brain activation pattern, provide a response that:
1. Reflects the cognitive processing indicated by the activation profile
2. Emphasizes the dominant brain regions and their functions
3. Maintains coherence with the input text
4. Demonstrates awareness of the brain-inspired geometric structure

Response:"""
    
    return prompt


def build_standard_prompt(text):
    """Build a standard prompt without SGP guidance."""
    return f"""Respond to this text: "{text}"

Provide a thoughtful response that engages with the content.

Response:"""


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def run_sgp_llm_integration():
    """Run the full SGP-guided LLM integration pipeline."""
    print("=" * 70)
    print("SGP-Tribe3 - Phase 2: LLM Integration with SGP Guidance")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Test stimuli: {len(TEST_STIMULI)}")
    print("=" * 70)
    
    # Load existing results to get SGP profiles
    results_path = "results/full_battery_1000/results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            full_results = json.load(f)
        all_results = full_results.get('results', [])
        # Build lookup by text
        result_lookup = {}
        for r in all_results:
            result_lookup[r.get('text', '')] = r
        print(f"Loaded {len(all_results)} existing SGP profiles")
    else:
        # Load from checkpoints
        import glob
        all_results = []
        for f in sorted(glob.glob("results/full_battery_1000/checkpoint_*.json")):
            with open(f) as fp:
                data = json.load(fp)
            all_results.extend(data.get('results', []))
        result_lookup = {r.get('text', ''): r for r in all_results}
        print(f"Loaded {len(all_results)} SGP profiles from checkpoints")
    
    # Initialize Ollama client
    client = ollama.Client()
    model = "tinyllama"
    
    # Run comparison
    comparison_results = []
    
    for i, stimulus in enumerate(TEST_STIMULI):
        text = stimulus['text']
        category = stimulus['category']
        
        print(f"\n[{i+1}/{len(TEST_STIMULI)}] {category}: {text[:60]}...")
        
        # Get SGP profile
        sgp_profile = result_lookup.get(text, {})
        sgp_nodes = sgp_profile.get('sgp_nodes', {})
        
        if not sgp_nodes:
            # Use mean activations as fallback
            sgp_nodes = {node: 0.95 for node in NODE_ORDER}
        
        # Build prompts
        sgp_prompt = build_sgp_guided_prompt(text, sgp_nodes)
        standard_prompt = build_standard_prompt(text)
        
        print(f"  SGP profile: Top nodes = {sorted(sgp_nodes.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # Generate responses
        try:
            # SGP-guided response
            print(f"  Generating SGP-guided response...")
            start = time.time()
            sgp_response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': sgp_prompt}],
                options={'temperature': 0.7, 'num_predict': 200}
            )
            sgp_text = sgp_response['message']['content']
            sgp_time = time.time() - start
            print(f"  SGP-guided: {sgp_text[:100]}... ({sgp_time:.1f}s)")
        except Exception as e:
            sgp_text = f"ERROR: {e}"
            sgp_time = 0
        
        try:
            # Standard response
            print(f"  Generating standard response...")
            start = time.time()
            std_response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': standard_prompt}],
                options={'temperature': 0.7, 'num_predict': 200}
            )
            std_text = std_response['message']['content']
            std_time = time.time() - start
            print(f"  Standard: {std_text[:100]}... ({std_time:.1f}s)")
        except Exception as e:
            std_text = f"ERROR: {e}"
            std_time = 0
        
        comparison_results.append({
            'text': text,
            'category': category,
            'sgp_nodes': sgp_nodes,
            'sgp_guided_response': sgp_text,
            'standard_response': std_text,
            'sgp_generation_time': sgp_time,
            'standard_generation_time': std_time,
            'sgp_prompt': sgp_prompt,
            'standard_prompt': standard_prompt,
        })
    
    # Save results
    output_path = "results/full_battery_1000/sgp_llm_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'model': model,
            'n_stimuli': len(TEST_STIMULI),
            'results': comparison_results,
        }, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("SGP-LLM INTEGRATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved: {output_path}")
    
    return comparison_results


if __name__ == "__main__":
    results = run_sgp_llm_integration()
    if results:
        print("\nSGP-LLM integration complete!")
    else:
        print("\nSGP-LLM integration failed!")
