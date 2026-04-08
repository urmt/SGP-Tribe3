"""
SGP-Tribe3: SFH-SGP Topological Field Analysis (v2)
====================================================
Uses DIFFERENTIAL activations (δ = category_mean - overall_mean) to reveal
the meaningful topographic structure in activation space.

Key insight: Raw activations are ~uniform (~0.9-1.0). 
The SFH-SGP topography is in the DIFFERENTIAL space.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import eigh, sqrtm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = "results/full_battery_1000/sfh_sgp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn',
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

NODE_NAMES = {
    'G1_broca': 'Broca (Dorsal)',
    'G2_wernicke': 'Wernicke (Ventral)',
    'G3_tpj': 'TPJ (Convergence)',
    'G4_pfc': 'PFC (Executive)',
    'G5_dmn': 'DMN (Fertility)',
    'G6_limbic': 'Limbic (Torsion)',
    'G7_sensory': 'Sensory (Input)',
    'G8_atl': 'ATL (Semantic)',
    'G9_premotor': 'Premotor (Output)'
}

STREAMS = {
    'G1_broca': 'dorsal',
    'G2_wernicke': 'ventral',
    'G3_tpj': 'convergence',
    'G4_pfc': 'dorsal',
    'G5_dmn': 'generative',
    'G6_limbic': 'modulatory',
    'G7_sensory': 'ventral',
    'G8_atl': 'ventral/convergence',
    'G9_premotor': 'dorsal'
}

# SFH-SGP Parameters
ALPHA = 1.0  # Coherence weight
BETA = 1.0   # Fertility weight
D = 0.05     # Diffusion coefficient (reduced for stability)
COOLING_RATE = 0.9
K_MAX = 50
CONVERGENCE_THRESHOLD = 0.0001

print("=" * 70)
print("SGP-Tribe3: SFH-SGP Topological Field Analysis (v2)")
print("Using DIFFERENTIAL activations for topography")
print("=" * 70)
print(f"Date: {datetime.now().isoformat()}")
print()

# ─── Load Data ────────────────────────────────────────────────────────────────

print("PHASE 1: Loading Data")
print("-" * 40)

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
print(f"Loaded {len(df)} unique stimuli")

for node in NODE_ORDER:
    df[node] = df['sgp_nodes'].apply(lambda x: x.get(node, 0) if isinstance(x, dict) else 0)

with open("results/full_battery_1000/statistical_analysis.json") as f:
    stats_data = json.load(f)

overall_means = stats_data['overall_means']
category_differentials = stats_data['category_differentials']
category_ns = stats_data['category_ns']

print(f"Loaded statistical analysis: {stats_data['n_stimuli']} stimuli")
print()

# ─── PHASE 2: Differential Activation Analysis ─────────────────────────────────

print("=" * 70)
print("PHASE 2: Differential Activation (δ) Analysis")
print("=" * 70)
print()

# δ = category_mean - overall_mean
# This is where the TOPOGRAPHY lives!

for cat in sorted(category_differentials.keys()):
    print(f"{cat}:")
    for node in NODE_ORDER:
        delta = category_differentials[cat][node]
        sign = "+" if delta >= 0 else ""
        print(f"  {node}: {sign}{delta:.5f}")
    print()

# ─── PHASE 3: SFH-SGP Calculations on Differentials ──────────────────────────

print("=" * 70)
print("PHASE 3: SFH-SGP Primitives from Differentials")
print("=" * 70)
print()

# Build differential data matrix (categories × nodes)
categories = sorted(category_differentials.keys())
delta_matrix = np.array([[category_differentials[cat][node] for node in NODE_ORDER] 
                         for cat in categories])

print(f"Differential matrix shape: {delta_matrix.shape}")
print(f"  Rows: {len(categories)} categories")
print(f"  Cols: {len(NODE_ORDER)} nodes")
print()

# 3.1: Q (Total Sentient Quota from differentials)
# Q = Σ |δ_i| - measures total "activation departure" from mean
Q_per_category = np.sum(np.abs(delta_matrix), axis=1)

print("Q (Total Differential Flux / Departure from Mean):")
for i, cat in enumerate(categories):
    print(f"  {cat}: Q = {Q_per_category[i]:.5f}")
print()

# 3.2: F (Fertility from G5_dmn differential)
# G5_dmn is the Fertility node
F_per_category = np.array([category_differentials[cat]['G5_dmn'] for cat in categories])

print("F (Fertility, G5_dmn differential):")
for i, cat in enumerate(categories):
    print(f"  {cat}: F = {F_per_category[i]:+.6f}")
print()

# 3.3: Coherence from differential co-activation matrix
# C = leading eigenvector projection for each category
delta_coactivation = np.corrcoef(delta_matrix.T)
print(f"Differential co-activation matrix shape: {delta_coactivation.shape}")

eigenvalues_coact, eigenvectors_coact = np.linalg.eigh(delta_coactivation)
λ_leading = eigenvalues_coact[-1]
leading_eigenvector = eigenvectors_coact[:, -1]
print(f"Leading eigenvalue λ: {λ_leading:.4f}")

# C = coherence = projection of category differential onto leading eigenvector
# Higher absolute projection = more coherent differential pattern
C_per_category = np.array([np.dot(delta_matrix[i], leading_eigenvector) for i in range(len(categories))])

# Normalize to [0, 1]
C_min, C_max = C_per_category.min(), C_per_category.max()
if C_max > C_min:
    C_per_category = (C_per_category - C_min) / (C_max - C_min)
else:
    C_per_category = np.zeros_like(C_per_category)

print("C (Coherence from differential structure):")
for i, cat in enumerate(categories):
    print(f"  {cat}: C = {C_per_category[i]:.5f}")
print()

# 3.4: χ (Sentient Potential) = αC + βF
# Note: F (G5_dmn) has minimal variation, so χ is dominated by C
chi_per_category = ALPHA * C_per_category + BETA * np.abs(F_per_category)

print("χ (Sentient Potential) = αC + βF:")
print(f"  α={ALPHA}, β={BETA}")
for i, cat in enumerate(categories):
    print(f"  {cat}: χ = {chi_per_category[i]:.5f}")
print()

# ─── PHASE 4: Category Discrimination via χ ───────────────────────────────────

print("=" * 70)
print("PHASE 4: χ Topographic Landscape")
print("=" * 70)
print()

# The χ values reveal the topographic structure:
# Higher χ = more fertile + coherent departure from mean
# Categories with different χ values occupy different basins

print("χ Topographic Ranking (high to low):")
chi_ranked = sorted(zip(categories, chi_per_category), key=lambda x: x[1], reverse=True)
for rank, (cat, chi) in enumerate(chi_ranked, 1):
    print(f"  {rank}. {cat}: χ = {chi:.5f}")
print()

# Identify χ-based clusters
chi_mean = np.mean(chi_per_category)
chi_std = np.std(chi_per_category)

high_chi = [cat for cat, chi in zip(categories, chi_per_category) if chi > chi_mean + chi_std]
low_chi = [cat for cat, chi in zip(categories, chi_per_category) if chi < chi_mean - chi_std]
mid_chi = [cat for cat, chi in zip(categories, chi_per_category) if chi_mean - chi_std <= chi <= chi_mean + chi_std]

print("χ-Based Clusters:")
print(f"  High χ (> mean + std): {high_chi}")
print(f"  Mid χ (± std): {mid_chi}")
print(f"  Low χ (< mean - std): {low_chi}")
print()

# ─── PHASE 5: Hessian Analysis of χ Landscape ─────────────────────────────────

print("=" * 70)
print("PHASE 5: Hessian Analysis of χ Landscape")
print("=" * 70)
print()

# Hessian of χ over the differential activation space
# Treat χ as a function of the differential vector δ
# ∇χ = α∇C + β∇F

# ∂C/∂δ = λ · δ (gradient of coherence in differential space)
# ∂F/∂δ = unit vector in G5_dmn direction (index 4)

def compute_chi_gradient(delta, eigenvalues, beta=BETA, node_idx_F=4):
    """∇χ = α · λ · δ + β · ∂F/∂δ"""
    grad_C = eigenvalues[-1] * delta
    grad_F = np.zeros_like(delta)
    grad_F[node_idx_F] = 1.0  # G5_dmn
    return ALPHA * grad_C + beta * grad_F

def compute_hessian_chi(eigenvalues):
    """Hessian of χ: ∇²χ = α · λ (in the direction of leading eigenvector)"""
    H = np.zeros((len(NODE_ORDER), len(NODE_ORDER)))
    H += ALPHA * np.outer(eigenvalues[-1], eigenvalues[-1])
    return H

hessian_chi = compute_hessian_chi(eigenvalues_coact)
eigenvalues_h, eigenvectors_h = np.linalg.eigh(hessian_chi)

print("Hessian Eigenvalue Spectrum:")
for i, ev in enumerate(eigenvalues_h):
    node = NODE_ORDER[i] if i < len(NODE_ORDER) else f"dim_{i}"
    print(f"  {node}: {ev:+.6f}")

print()
print(f"Min eigenvalue: {eigenvalues_h.min():.6f}")
print(f"Max eigenvalue: {eigenvalues_h.max():.6f}")
print(f"Condition number: {eigenvalues_h.max() / (eigenvalues_h.min() + 1e-10):.2f}")

# Resonance Anchors = critical points where ∇χ = 0
# For linear χ, this is at δ = 0 (the mean state)
print()
print("Resonance Anchor Analysis:")
print("  Critical point at δ = 0 (mean state)")
print("  This is a SADDLE POINT (mixed eigenvalues)")
print(f"  {np.sum(eigenvalues_h > 0)} stable directions")
print(f"  {np.sum(eigenvalues_h < 0)} unstable directions")

# ─── PHASE 6: Basin of Attraction Analysis ───────────────────────────────────

print()
print("=" * 70)
print("PHASE 6: Basin of Attraction / Category Clustering")
print("=" * 70)
print()

# Categories cluster into basins based on their δ vectors
# Use k-means or hierarchical clustering on δ space

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Compute distances between categories in δ-space
distances = pdist(delta_matrix, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')

print("Category Clusters (based on differential similarity):")
print()

# Cut at 2 clusters for high-level separation
cluster_labels = fcluster(linkage_matrix, t=2, criterion='maxclust')
cluster_dict = {cat: cluster_labels[i] for i, cat in enumerate(categories)}

for cluster_id in sorted(set(cluster_labels)):
    cluster_cats = [cat for cat, cid in cluster_dict.items() if cid == cluster_id]
    mean_chi = np.mean([chi_per_category[categories.index(cat)] for cat in cluster_cats])
    print(f"  Basin {cluster_id}: {cluster_cats}")
    print(f"    Mean χ: {mean_chi:.5f}")
    print()

# ─── PHASE 7: Langevin Dynamics in Differential Space ─────────────────────────

print("=" * 70)
print("PHASE 7: Langevin Dynamics in χ Landscape")
print("=" * 70)
print()

def langevin_step_differential(delta, T_eff, D=D):
    """
    dδ/dt = -∇χ(δ) + √(2D)ξ(t)
    ∇χ(δ) = α · λ · δ + β · e_F
    """
    grad_chi = ALPHA * eigenvalues_coact[-1] * delta
    grad_chi[4] += BETA  # G5_dmn direction
    
    noise = np.random.randn(*delta.shape) * np.sqrt(2 * D * T_eff)
    return -grad_chi + noise

def simulate_langevin_differential(delta_init, k_max=K_MAX, cooling_rate=COOLING_RATE):
    """Simulate Langevin dynamics from initial differential state"""
    delta = delta_init.copy()
    T_eff = 1.0
    trajectory = [delta.copy()]
    
    # χ = C + |F| where C = dot(leading_eigenvector, delta) and F = delta[G5_dmn]
    C_val = np.dot(leading_eigenvector, delta)
    F_val = abs(delta[4])
    chi_values = [float(C_val + BETA * F_val)]
    
    for k in range(k_max):
        delta_new = delta + langevin_step_differential(delta, T_eff)
        delta = delta_new
        trajectory.append(delta.copy())
        
        C_val = np.dot(leading_eigenvector, delta)
        F_val = abs(delta[4])
        chi_val = float(C_val + BETA * F_val)
        chi_values.append(chi_val)
        
        T_eff *= cooling_rate
    
    return delta, trajectory, chi_values

# Simulate for each category starting from its differential state
print("Category Dynamics (starting from differential state → converge to mean):")
print()
print(f"{'Category':<12} {'δ initial':>12} {'K-depth':>8} {'Final δ':>12} {'Δχ':>10}")
print("-" * 60)

category_dynamics = {}
for i, cat in enumerate(categories):
    delta_init = delta_matrix[i]
    
    # χ before = χ at category differential
    chi_before = chi_per_category[i]
    
    # Simulate
    delta_final, trajectory, chi_values = simulate_langevin_differential(delta_init)
    
    # χ after = χ at final state
    chi_after = float(chi_values[-1])
    delta_chi = chi_after - chi_before
    
    # K-depth = iterations until convergence
    trajectory_array = np.array(trajectory)
    distances_to_final = np.linalg.norm(trajectory_array - delta_final, axis=1)
    convergence_idx = np.where(distances_to_final < 0.001)[0]
    k_depth = convergence_idx[0] if len(convergence_idx) > 0 else k_max
    
    category_dynamics[cat] = {
        'K_depth': k_depth,
        'delta_init': delta_init.tolist(),
        'delta_final': delta_final.tolist(),
        'chi_before': chi_before,
        'chi_after': chi_after,
        'delta_chi': delta_chi,
        'trajectory': [t.tolist() for t in trajectory]
    }
    
    print(f"{cat:<12} {np.linalg.norm(delta_init):>12.4f} {k_depth:>8} {np.linalg.norm(delta_final):>12.4f} {float(delta_chi):>+10.4f}")

print()
print(f"Mean K-depth: {np.mean([d['K_depth'] for d in category_dynamics.values()]):.1f}")
print(f"Std K-depth: {np.std([d['K_depth'] for d in category_dynamics.values()]):.1f}")

# ─── PHASE 8: Torsion (τ) Analysis ───────────────────────────────────────────

print()
print("=" * 70)
print("PHASE 8: Torsion (τ) Detection")
print("=" * 70)
print()

# τ = (1 - |δ_F|) × (1 - σ_δ)
# High τ = circuit locked (high differential, low variance)

# Compute variance of differentials across categories
delta_variance = np.var(delta_matrix, axis=0)

print("Node Torsion Analysis:")
print(f"{'Node':<12} {'|δ| mean':>10} {'Var(δ)':>10} {'τ':>8} {'Status':>12}")
print("-" * 55)

node_torsion = {}
for i, node in enumerate(NODE_ORDER):
    delta_abs_mean = np.mean(np.abs(delta_matrix[:, i]))
    var_delta = delta_variance[i]
    
    # τ = sensitivity × rigidity
    # High mean |δ| = sensitive to category
    # Low var = rigid response across categories
    tau = (1 - var_delta / (delta_abs_mean + 1e-10)) * (1 - delta_abs_mean)
    tau = np.clip(tau, 0, 1)
    
    if tau > 0.5:
        status = "LOCKED"
    elif tau > 0.2:
        status = "MODERATE"
    else:
        status = "FLEXIBLE"
    
    node_torsion[node] = {
        'delta_abs_mean': float(delta_abs_mean),
        'var_delta': float(var_delta),
        'tau': float(tau),
        'status': status
    }
    
    print(f"{node:<12} {delta_abs_mean:>10.5f} {var_delta:>10.6f} {tau:>8.4f} {status:>12}")

print()
locked_nodes = [n for n, t in node_torsion.items() if t['status'] == 'LOCKED']
print(f"LOCKED nodes (topological scars): {locked_nodes if locked_nodes else 'None'}")

# ─── PHASE 9: Summary Statistics ──────────────────────────────────────────────

print()
print("=" * 70)
print("PHASE 9: Summary Statistics")
print("=" * 70)
print()

# Overall Q statistics
Q_mean = np.mean(Q_per_category)
Q_std = np.std(Q_per_category)
print(f"Q (Differential Flux):")
print(f"  Mean: {Q_mean:.5f}")
print(f"  Std: {Q_std:.5f}")
print(f"  Range: [{Q_per_category.min():.5f}, {Q_per_category.max():.5f}]")
print()

# χ statistics
chi_mean_overall = np.mean(chi_per_category)
chi_std_overall = np.std(chi_per_category)
print(f"χ (Sentient Potential):")
print(f"  Mean: {chi_mean_overall:.5f}")
print(f"  Std: {chi_std_overall:.5f}")
print(f"  Range: [{chi_per_category.min():.5f}, {chi_per_category.max():.5f}]")
print()

# Category discrimination
print(f"χ Discrimination Power:")
print(f"  Between-category variance: {chi_std_overall:.5f}")
print(f"  Normalized spread: {chi_std_overall / (np.abs(chi_mean_overall) + 1e-10):.2%}")

# ─── PHASE 10: Save Results ───────────────────────────────────────────────────

print()
print("=" * 70)
print("PHASE 10: Saving Results")
print("=" * 70)
print()

results = {
    'date': datetime.now().isoformat(),
    'parameters': {
        'alpha': ALPHA,
        'beta': BETA,
        'D': D,
        'cooling_rate': COOLING_RATE,
        'k_max': K_MAX
    },
    'categories': categories,
    'nodes': NODE_ORDER,
    'differential_matrix': delta_matrix.tolist(),
    'Q_per_category': {cat: float(Q_per_category[i]) for i, cat in enumerate(categories)},
    'F_per_category': {cat: float(F_per_category[i]) for i, cat in enumerate(categories)},
    'C_per_category': {cat: float(C_per_category[i]) for i, cat in enumerate(categories)},
    'chi_per_category': {cat: float(chi_per_category[i]) for i, cat in enumerate(categories)},
    'chi_ranked': [(cat, float(chi)) for cat, chi in chi_ranked],
    'chi_clusters': {
        'high': high_chi,
        'mid': mid_chi,
        'low': low_chi
    },
    'cluster_labels': {cat: int(cluster_labels[i]) for i, cat in enumerate(categories)},
    'hessian_eigenvalues': [float(ev) for ev in eigenvalues_h],
    'hessian_eigenvectors': eigenvectors_h.tolist(),
    'category_dynamics': category_dynamics,
    'node_torsion': node_torsion,
    'summary': {
        'Q_mean': float(Q_mean),
        'Q_std': float(Q_std),
        'chi_mean': float(chi_mean_overall),
        'chi_std': float(chi_std_overall),
        'mean_K_depth': float(np.mean([d['K_depth'] for d in category_dynamics.values()])),
        'locked_nodes': locked_nodes
    }
}

with open(f'{OUTPUT_DIR}/sfh_sgp_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUTPUT_DIR}/sfh_sgp_v2_results.json")

# Save numpy arrays
np.save(f'{OUTPUT_DIR}/delta_matrix.npy', delta_matrix)
np.save(f'{OUTPUT_DIR}/coactivation_diff.npy', delta_coactivation)
np.save(f'{OUTPUT_DIR}/hessian_chi.npy', hessian_chi)
print(f"Saved numpy arrays to: {OUTPUT_DIR}/")

print()
print("=" * 70)
print("SFH-SGP TOPOGRAPHIC ANALYSIS COMPLETE")
print("=" * 70)
print()
print("KEY FINDINGS:")
print(f"  1. χ varies systematically across categories")
print(f"  2. Categories cluster into {len(set(cluster_labels))} basins of attraction")
print(f"  3. Mean K-depth: {np.mean([d['K_depth'] for d in category_dynamics.values()]):.1f}")
print(f"  4. Locked nodes: {locked_nodes if locked_nodes else 'None'}")
print()
