"""
SGP-Tribe3 — Phase 2: LLM Integration Visualization
=====================================================
Demonstrates SGP-guided prompt construction and compares with standard prompts.
Creates visualizations showing how SGP activation profiles guide LLM generation.

Since LLM chat may be slow on CPU, this script:
1. Constructs SGP-guided vs standard prompts for 10 test stimuli
2. Visualizes the prompt differences
3. Shows how SGP activations map to cognitive guidance weights

Usage:
    python visualize_llm_integration.py

Outputs:
    results/full_battery_1000/figures/geo_06_llm_comparison.png
    results/full_battery_1000/sgp_prompt_examples.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ─── Load Results ────────────────────────────────────────────────────────────

results_files = sorted(glob.glob("results/full_battery_1000/checkpoint_*.json"))
all_results = []
for f in results_files:
    with open(f) as fp:
        data = json.load(fp)
    all_results.extend(data.get('results', []))

seen = set()
results = []
for r in all_results:
    sid = r.get('stimulus_id')
    if sid and sid not in seen:
        seen.add(sid)
        results.append(r)

df = pd.DataFrame(results)
print(f"Loaded {len(df)} stimuli")

NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn',
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

for node in NODE_ORDER:
    df[node] = df['sgp_nodes'].apply(lambda x: x.get(node, 0) if isinstance(x, dict) else 0)

output_dir = "results/full_battery_1000/figures"
os.makedirs(output_dir, exist_ok=True)

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

NODE_DESCRIPTIONS = {
    'G1_broca': 'Broca\'s Area',
    'G2_wernicke': 'Wernicke\'s Area',
    'G3_tpj': 'TPJ',
    'G4_pfc': 'PFC',
    'G5_dmn': 'DMN',
    'G6_limbic': 'Limbic',
    'G7_sensory': 'Sensory',
    'G8_atl': 'ATL',
    'G9_premotor': 'Premotor',
}

# ─── Build SGP-guided prompts ───────────────────────────────────────────────

def build_sgp_prompt(text, sgp_nodes):
    """Build SGP-guided prompt from activation profile."""
    sorted_nodes = sorted(sgp_nodes.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_nodes[:3]
    
    activation_parts = []
    for node, value in top3:
        activation_parts.append(f"    {NODE_DESCRIPTIONS.get(node, node)}: {value:.2f}")
    
    prompt = (
        f"SGP-Guided Response\n"
        f"==================\n"
        f"Input: \"{text}\"\n\n"
        f"Brain Activation Profile (SGP Model):\n"
        + "\n".join(activation_parts) +
        f"\n\nGuidance: Respond emphasizing the cognitive functions\n"
        f"indicated by the most activated brain regions above."
    )
    return prompt


def build_standard_prompt(text):
    """Build standard prompt without SGP guidance."""
    return f"Standard Response\n===============\nInput: \"{text}\"\n\nGuidance: Respond thoughtfully to the input."


# ─── Generate prompt examples ────────────────────────────────────────────────

print("\nGenerating SGP-guided prompt examples...")

prompt_examples = []
for stimulus in TEST_STIMULI:
    text = stimulus['text']
    category = stimulus['category']
    
    # Find matching result
    match = df[df['text'] == text]
    if len(match) > 0:
        sgp_nodes = match.iloc[0]['sgp_nodes']
    else:
        # Use category mean
        cat_data = df[df['category'] == category]
        sgp_nodes = {node: cat_data[node].mean() for node in NODE_ORDER}
    
    sgp_prompt = build_sgp_prompt(text, sgp_nodes)
    std_prompt = build_standard_prompt(text)
    
    prompt_examples.append({
        'text': text,
        'category': category,
        'sgp_nodes': sgp_nodes,
        'sgp_prompt': sgp_prompt,
        'standard_prompt': std_prompt,
    })

# Save prompt examples
with open('results/full_battery_1000/sgp_prompt_examples.json', 'w') as f:
    json.dump(prompt_examples, f, indent=2)
print(f"Saved: results/full_battery_1000/sgp_prompt_examples.json")

# ─── Figure: LLM Comparison ─────────────────────────────────────────────────

print("\nGenerating Figure: SGP-guided vs Standard LLM prompts...")

fig, axes = plt.subplots(2, 5, figsize=(22, 10))
axes = axes.flatten()

for idx, example in enumerate(prompt_examples):
    ax = axes[idx]
    
    sgp_nodes = example['sgp_nodes']
    category = example['category']
    
    # Create radar chart for SGP activation
    angles = np.linspace(0, 2 * np.pi, len(NODE_ORDER), endpoint=False).tolist()
    values = [sgp_nodes.get(node, 0) for node in NODE_ORDER]
    values += values[:1]  # Close the radar chart
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    
    # Add node labels
    for i, node in enumerate(NODE_ORDER):
        angle = angles[i]
        r = values[i]
        ax.text(angle, r + 0.02, NODE_DESCRIPTIONS.get(node, node),
                ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    ax.set_ylim(0.8, 1.05)
    ax.set_title(f'{category.capitalize()}', fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

plt.suptitle('SGP Activation Profiles Used for LLM Guidance\n'
             'Each radar chart shows the brain activation profile that guides the LLM response',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_06_llm_guidance_profiles.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_06_llm_guidance_profiles.png")

# ─── Figure: Prompt Comparison ──────────────────────────────────────────────

print("Generating Figure: Prompt comparison visualization...")

fig, axes = plt.subplots(5, 2, figsize=(16, 20))

for idx in range(5):
    example = prompt_examples[idx]
    
    # Left: SGP-guided prompt
    ax1 = axes[idx, 0]
    ax1.text(0.05, 0.95, example['sgp_prompt'], transform=ax1.transAxes,
             fontsize=7, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fd', edgecolor='#1f77b4'))
    ax1.set_title(f"SGP-Guided Prompt: {example['category'].capitalize()}", fontsize=10, fontweight='bold', color='#1f77b4')
    ax1.axis('off')
    
    # Right: Standard prompt
    ax2 = axes[idx, 1]
    ax2.text(0.05, 0.95, example['standard_prompt'], transform=ax2.transAxes,
             fontsize=7, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#999999'))
    ax2.set_title(f"Standard Prompt: {example['category'].capitalize()}", fontsize=10, fontweight='bold', color='#666666')
    ax2.axis('off')

plt.suptitle('SGP-Guided vs Standard LLM Prompts\n'
             'SGP prompts include brain activation profiles to guide response generation',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_07_prompt_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_07_prompt_comparison.png")

# ─── Figure: SGP-to-LLM Pipeline ────────────────────────────────────────────

print("Generating Figure: SGP-to-LLM pipeline diagram...")

fig, ax = plt.subplots(figsize=(16, 8))

# Pipeline stages
stages = [
    {'x': 0.05, 'y': 0.5, 'w': 0.18, 'h': 0.6, 'label': 'Input Text', 'color': '#2ecc71',
     'detail': 'Raw text stimulus\n(e.g., "The cat sat\non the mat.")'},
    {'x': 0.28, 'y': 0.5, 'w': 0.18, 'h': 0.6, 'label': 'SGP Encoding', 'color': '#3498db',
     'detail': 'TRIBE v2 →\n9-node activation\nprofile'},
    {'x': 0.51, 'y': 0.5, 'w': 0.18, 'h': 0.6, 'label': 'Resonance\nWeights', 'color': '#9b59b6',
     'detail': 'Activation →\nprompt weights\n(top 3 nodes)'},
    {'x': 0.74, 'y': 0.5, 'w': 0.18, 'h': 0.6, 'label': 'LLM\nGeneration', 'color': '#e74c3c',
     'detail': 'SGP-guided\nprompt →\nbrain-informed response'},
]

for stage in stages:
    rect = plt.Rectangle((stage['x'], stage['y'] - stage['h']/2), stage['w'], stage['h'],
                         facecolor=stage['color'], alpha=0.8, edgecolor='black', linewidth=2,
                         zorder=2)
    ax.add_patch(rect)
    ax.text(stage['x'] + stage['w']/2, stage['y'] + 0.15, stage['label'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=3)
    ax.text(stage['x'] + stage['w']/2, stage['y'] - 0.1, stage['detail'],
            ha='center', va='center', fontsize=8, color='white', zorder=3, family='monospace')

# Arrows between stages
for i in range(len(stages) - 1):
    x1 = stages[i]['x'] + stages[i]['w']
    x2 = stages[i+1]['x']
    y = stages[i]['y']
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))

# Add example activation profile
example_nodes = {'G5_dmn': 1.000, 'G4_pfc': 0.953, 'G1_broca': 0.931,
                 'G9_premotor': 0.944, 'G7_sensory': 0.941, 'G6_limbic': 0.937,
                 'G3_tpj': 0.946, 'G8_atl': 0.903, 'G2_wernicke': 0.914}

ax.text(0.5, 0.08, 'Example: "If P implies Q, then Q implies R" → SGP Profile:\n'
        'G5_DMN: 1.000 | G4_PFC: 0.953 | G9_Premotor: 0.944 | ...\n'
        '→ Guides LLM to emphasize self-referential + executive reasoning',
        ha='center', va='center', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#ffc107'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('SGP-to-LLM Guidance Pipeline\n'
             'Text → Brain Activation → Resonance Weights → SGP-Guided LLM Response',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/geo_08_sgp_llm_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_08_sgp_llm_pipeline.png")

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SGP-LLM INTEGRATION VISUALIZATION COMPLETE")
print(f"{'=' * 60}")
print(f"Figures saved to: {output_dir}/")
print(f"  - geo_06_llm_guidance_profiles.png")
print(f"  - geo_07_prompt_comparison.png")
print(f"  - geo_08_sgp_llm_pipeline.png")
print(f"\nPrompt examples saved to: results/full_battery_1000/sgp_prompt_examples.json")
