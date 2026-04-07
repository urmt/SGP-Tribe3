"""
SGP-Tribe3 — Full Research Text Battery (100 stimuli)
=====================================================
Comprehensive text stimuli across 10 categories for brain encoding analysis.
Tests the Hickok-Poeppel dual-stream hypothesis and SGP node differentiation.

Categories:
1. Simple sentences (baseline)
2. Logical/mathematical reasoning
3. Emotional/narrative
4. Factual/technical
5. Self-referential/abstract
6. Spatial/visual
7. Auditory/musical
8. Social/theory of mind
9. Motor/action
10. Memory/temporal

Usage:
    python run_full_battery.py
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

# Full text battery - 100 stimuli across 10 categories
TEXT_BATTERY = [
    # Category 1: Simple sentences (baseline)
    {"id": 1, "category": "simple", "text": "The cat sat on the mat.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 2, "category": "simple", "text": "The dog ran across the yard.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 3, "category": "simple", "text": "Birds fly in the blue sky.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 4, "category": "simple", "text": "The sun shines brightly today.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 5, "category": "simple", "text": "She opened the door slowly.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 6, "category": "simple", "text": "The book rests on the table.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 7, "category": "simple", "text": "Rain falls on the green grass.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 8, "category": "simple", "text": "He drank water from the glass.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 9, "category": "simple", "text": "The clock ticked on the wall.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    {"id": 10, "category": "simple", "text": "Flowers bloom in the spring.", "expected_dominant": ["G2_wernicke", "G7_sensory"]},
    
    # Category 2: Logical/mathematical reasoning
    {"id": 11, "category": "logical", "text": "If P implies Q, and Q implies R, then P implies R.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 12, "category": "logical", "text": "All humans are mortal. Socrates is human. Therefore Socrates is mortal.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 13, "category": "logical", "text": "The sum of angles in a triangle equals one hundred eighty degrees.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 14, "category": "logical", "text": "If it rains, the ground gets wet. It is raining. Therefore the ground is wet.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 15, "category": "logical", "text": "Two plus two equals four, and four plus four equals eight.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 16, "category": "logical", "text": "The square root of sixteen is four, and the square root of nine is three.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 17, "category": "logical", "text": "If all A are B, and all B are C, then all A must be C.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 18, "category": "logical", "text": "The probability of rolling a six on a fair die is one in six.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 19, "category": "logical", "text": "A implies B does not mean B implies A; the converse is not necessarily true.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    {"id": 20, "category": "logical", "text": "The product of two negative numbers yields a positive result.", "expected_dominant": ["G4_pfc", "G1_broca"]},
    
    # Category 3: Emotional/narrative
    {"id": 21, "category": "emotional", "text": "She felt the warmth of the sun on her skin as memories of childhood flooded back.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 22, "category": "emotional", "text": "His heart raced with fear as the shadow moved closer in the darkness.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 23, "category": "emotional", "text": "Tears of joy streamed down her face when she saw her family again.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 24, "category": "emotional", "text": "The old man smiled sadly as he remembered his youth and lost friends.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 25, "category": "emotional", "text": "Anger burned within him as he read the unfair accusations.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 26, "category": "emotional", "text": "She hugged her child tightly, feeling overwhelming love and protectiveness.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 27, "category": "emotional", "text": "The grief hit him like a wave when he heard the devastating news.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 28, "category": "emotional", "text": "Hope flickered in her heart despite all the hardships she had endured.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 29, "category": "emotional", "text": "Pride swelled in his chest as he watched his student succeed.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    {"id": 30, "category": "emotional", "text": "The loneliness of the empty house echoed through every silent room.", "expected_dominant": ["G6_limbic", "G5_dmn"]},
    
    # Category 4: Factual/technical
    {"id": 31, "category": "factual", "text": "The mitochondria is the powerhouse of the cell.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 32, "category": "factual", "text": "DNA contains the genetic instructions for the development of all living organisms.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 33, "category": "factual", "text": "Photosynthesis converts light energy into chemical energy stored in glucose.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 34, "category": "factual", "text": "The Earth orbits the Sun at an average distance of ninety-three million miles.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 35, "category": "factual", "text": "Water boils at one hundred degrees Celsius at standard atmospheric pressure.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 36, "category": "factual", "text": "The human brain contains approximately eighty-six billion neurons.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 37, "category": "factual", "text": "Gravity is the force that attracts objects with mass toward each other.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 38, "category": "factual", "text": "The speed of light in vacuum is approximately three hundred thousand kilometers per second.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 39, "category": "factual", "text": "Plate tectonics explains the movement of Earth's lithospheric plates.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    {"id": 40, "category": "factual", "text": "The periodic table organizes elements by atomic number and chemical properties.", "expected_dominant": ["G4_pfc", "G7_sensory"]},
    
    # Category 5: Self-referential/abstract
    {"id": 41, "category": "abstract", "text": "What if the universe is a simulation and we're just characters in someone else's dream?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 42, "category": "abstract", "text": "I think therefore I am, but what does it truly mean to think?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 43, "category": "abstract", "text": "If consciousness emerges from complexity, at what point does a system become aware?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 44, "category": "abstract", "text": "The nature of reality may be fundamentally different from our perception of it.", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 45, "category": "abstract", "text": "Perhaps free will is an illusion created by the brain's predictive mechanisms.", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 46, "category": "abstract", "text": "What is the relationship between language and thought? Does one shape the other?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 47, "category": "abstract", "text": "If we could upload our minds to computers, would we still be ourselves?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 48, "category": "abstract", "text": "The concept of infinity challenges our ability to comprehend the universe.", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 49, "category": "abstract", "text": "Time may be an emergent property rather than a fundamental dimension.", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    {"id": 50, "category": "abstract", "text": "Can we ever truly know another person's subjective experience?", "expected_dominant": ["G5_dmn", "G3_tpj"]},
    
    # Category 6: Spatial/visual
    {"id": 51, "category": "spatial", "text": "The mountain peak rose sharply above the clouds, casting a long shadow over the valley.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 52, "category": "spatial", "text": "She navigated through the narrow alleyways of the ancient city.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 53, "category": "spatial", "text": "The spiral staircase wound upward in a perfect golden ratio.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 54, "category": "spatial", "text": "Stars scattered across the night sky like diamonds on black velvet.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 55, "category": "spatial", "text": "The river curved gracefully through the landscape, forming wide meanders.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 56, "category": "spatial", "text": "He visualized the three-dimensional structure rotating in his mind.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 57, "category": "spatial", "text": "The cathedral's vaulted ceiling soared upward toward the heavens.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 58, "category": "spatial", "text": "The map showed the territory extending from the coast to the mountains.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 59, "category": "spatial", "text": "Geometric patterns adorned the walls in intricate repeating designs.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    {"id": 60, "category": "spatial", "text": "The horizon stretched endlessly where the ocean met the sky.", "expected_dominant": ["G7_sensory", "G9_premotor"]},
    
    # Category 7: Auditory/musical
    {"id": 61, "category": "auditory", "text": "The melody drifted through the air like a gentle stream of sound.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 62, "category": "auditory", "text": "Thunder rumbled in the distance, growing louder with each passing moment.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 63, "category": "auditory", "text": "The orchestra swelled to a crescendo as the final movement began.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 64, "category": "auditory", "text": "Birdsong filled the morning air with cheerful chirping and warbling.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 65, "category": "auditory", "text": "The rhythm of the drums echoed through the crowded street.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 66, "category": "auditory", "text": "She hummed a familiar tune while walking through the garden.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 67, "category": "auditory", "text": "The violin's high notes pierced the silence with crystalline clarity.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 68, "category": "auditory", "text": "Waves crashed against the shore with a rhythmic roaring sound.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 69, "category": "auditory", "text": "The choir's harmonies blended together in perfect resonance.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    {"id": 70, "category": "auditory", "text": "A whisper carried secrets across the quiet room.", "expected_dominant": ["G7_sensory", "G2_wernicke"]},
    
    # Category 8: Social/theory of mind
    {"id": 71, "category": "social", "text": "She knew he was lying, but chose to pretend she believed him.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 72, "category": "social", "text": "He wondered what she was thinking as she stared out the window.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 73, "category": "social", "text": "The teacher understood that each student learned at their own pace.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 74, "category": "social", "text": "They negotiated carefully, each trying to understand the other's position.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 75, "category": "social", "text": "She felt embarrassed when she realized everyone was watching her.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 76, "category": "social", "text": "He predicted his friend's reaction before even delivering the news.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 77, "category": "social", "text": "The diplomat carefully considered how each nation would interpret the proposal.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 78, "category": "social", "text": "She recognized the subtle hint of sarcasm in his seemingly complimentary remark.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 79, "category": "social", "text": "The children learned to share by understanding each other's feelings.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    {"id": 80, "category": "social", "text": "He imagined how his ancestors would view the modern world.", "expected_dominant": ["G3_tpj", "G4_pfc"]},
    
    # Category 9: Motor/action
    {"id": 81, "category": "motor", "text": "She grasped the heavy stone and hurled it across the river.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 82, "category": "motor", "text": "The dancer leaped across the stage with graceful precision.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 83, "category": "motor", "text": "He typed rapidly on the keyboard, fingers flying across the keys.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 84, "category": "motor", "text": "The athlete sprinted toward the finish line with determined strides.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 85, "category": "motor", "text": "She carefully threaded the needle with steady hands.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 86, "category": "motor", "text": "The carpenter hammered the nail into the wooden beam.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 87, "category": "motor", "text": "He caught the ball with one hand while running at full speed.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 88, "category": "motor", "text": "The surgeon's hands moved with practiced precision during the operation.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 89, "category": "motor", "text": "She climbed the rope hand over hand with practiced ease.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    {"id": 90, "category": "motor", "text": "The pianist's fingers danced across the keys in a blur of motion.", "expected_dominant": ["G9_premotor", "G1_broca"]},
    
    # Category 10: Memory/temporal
    {"id": 91, "category": "memory", "text": "She remembered the day they first met as if it were yesterday.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 92, "category": "memory", "text": "The scent of fresh bread triggered vivid memories of his grandmother's kitchen.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 93, "category": "memory", "text": "He recalled the sequence of events leading up to the accident.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 94, "category": "memory", "text": "The photograph brought back forgotten memories of a summer long past.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 95, "category": "memory", "text": "She tried to remember where she had placed the important documents.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 96, "category": "memory", "text": "The melody reminded him of a song his mother used to sing.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 97, "category": "memory", "text": "He reflected on the lessons learned from years of experience.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 98, "category": "memory", "text": "The old journal revealed secrets from a century ago.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 99, "category": "memory", "text": "She anticipated the upcoming event with a mixture of excitement and anxiety.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
    {"id": 100, "category": "memory", "text": "The passage of time had faded the sharp edges of his grief.", "expected_dominant": ["G5_dmn", "G6_limbic"]},
]


def run_full_battery():
    """Run the full research text battery."""
    print("=" * 80)
    print("SGP-Tribe3 — Full Research Text Battery (100 Stimuli)")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Text encoder: {'ollama' if app._ollama_available else 'llama-cpu'}")
    print(f"Number of stimuli: {len(TEXT_BATTERY)}")
    print(f"Categories: {len(set(t['category'] for t in TEXT_BATTERY))}")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model...")
    app._load_model()
    time.sleep(5)
    
    if not app._model_loaded:
        print(f"ERROR: Model failed to load: {app._model_error}")
        return None
    
    # Force Ollama check
    if not app._ollama_available:
        print("Forcing Ollama connection check...")
        app._check_ollama_connection()
        if app._ollama_available:
            app._load_adapter()
    
    print(f"Model loaded: {app._model_loaded}")
    print(f"Ollama available: {app._ollama_available}")
    print(f"Adapter loaded: {app._adapter is not None}")
    
    # Run tests
    print("\n[2/3] Running full battery...")
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
                "activation_timeline": result.get('activation_timeline', []),
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
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(TEXT_BATTERY) - i - 1)
            print(f"\n  Progress: {i+1}/{len(TEXT_BATTERY)} ({(i+1)/len(TEXT_BATTERY)*100:.0f}%)")
            print(f"  Elapsed: {elapsed/60:.1f}min, Estimated remaining: {remaining/60:.1f}min")
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("FULL BATTERY SUMMARY")
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
    output_file = os.path.join(output_dir, f"full_battery_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "phase": "full_battery",
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
    results = run_full_battery()
    if results:
        print("\n✅ Full research battery complete!")
        sys.exit(0)
    else:
        print("\n❌ Full research battery failed!")
        sys.exit(1)
