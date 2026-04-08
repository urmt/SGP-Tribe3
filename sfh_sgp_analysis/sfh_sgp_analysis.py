"""
SGP-Tribe3: SFH-SGP Topological Field Analysis
==============================================
This script computes the full SFH-SGP mathematical framework from node-level activation data.

SFH-SGP Primitives:
- Q: Total Sentient Quota
- F: Fertility (G5_dmn activation)
- C: Coherence (from co-activation matrix)
- χ: Sentient Potential = αC + βF
- Langevin dynamics for field evolution
- Hessian analysis for critical points
- Torsion (τ) detection for locked circuits
- K-depth measurement for processing complexity

Usage:
    python sfh_sgp_analysis.py
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
from scipy.spatial.distance import cdist
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
ALPHA = 1.0  # Coherence weight (α > 0)
BETA = 1.0   # Fertility weight (β > 0)
GAMMA = 0.1  # Geodesic decay parameter
D = 0.1      # Diffusion coefficient
COOLING_RATE = 0.95  # Temperature annealing
K_MAX = 100  # Max iterations for K-depth
CONVERGENCE_THRESHOLD = 0.001

print("=" * 70)
print("SGP-Tribe3: SFH-SGP Topological Field Analysis")
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

df['category'].value_counts()

with open("results/full_battery_1000/statistical_analysis.json") as f:
    stats_data = json.load(f)

print(f"Loaded statistical analysis: {stats_data['n_stimuli']} stimuli, {stats_data['n_categories']} categories")
print()

# ─── PHASE 2: Core SFH-SGP Calculations ───────────────────────────────────────

print("PHASE 2: Computing SFH-SGP Primitives")
print("-" * 40)

# 2.1: Q (Total Sentient Quota) per stimulus
# Q = Σ |activation_i| across all 9 nodes (using normalized activations ~ [0.8, 1.0])
# Since our activations are normalized, we compute Q as sum of absolute deviations from baseline

df['Q'] = df[NODE_ORDER].apply(lambda row: np.sum(np.abs(row - 0.5)), axis=1)
df['Q_normalized'] = df['Q'] / df['Q'].mean()

print(f"Q (Total Sentient Quota):")
print(f"  Mean: {df['Q'].mean():.4f}")
print(f"  Std:  {df['Q'].std():.4f}")
print(f"  Range: [{df['Q'].min():.4f}, {df['Q'].max():.4f}]")

# 2.2: Qk (Sub-quota per node) - partition structure
# Qk = activation_k / Q (fraction of total quota per node)
for node in NODE_ORDER:
    df[f'Q_{node}'] = df[node] / df['Q']

print(f"\nQ partition fractions (Qk):")
for node in NODE_ORDER[:3]:
    print(f"  {node}: mean={df[f'Q_{node}'].mean():.4f}")

# 2.3: F (Fertility) = G5_dmn activation
df['F'] = df['G5_dmn']
print(f"\nF (Fertility, G5_dmn):")
print(f"  Mean: {df['F'].mean():.4f}")
print(f"  Range: [{df['F'].min():.4f}, {df['F'].max():.4f}]")

# 2.4: Co-activation Matrix (9×9)
print(f"\nComputing co-activation matrix (9×9)...")
node_data = df[NODE_ORDER].values
coactivation_matrix = np.corrcoef(node_data.T)
print(f"  Co-activation matrix shape: {coactivation_matrix.shape}")

# 2.5: C (Coherence) from co-activation matrix
# C = log10(π) + log10(λ) where λ = leading eigenvalue
# Since log10(π) is constant, C is proportional to log10(λ)
eigenvalues, eigenvectors = np.linalg.eigh(coactivation_matrix)
leading_eigenvalue = eigenvalues[-1]
λ = max(leading_eigenvalue, 1e-10)  # Avoid log of non-positive

C_const = np.log10(np.pi)  # Constant term
C_variable = np.log10(λ)   # Variable term from leading eigenvalue

# Normalize C to [0, 1] range
C_base = C_const + C_variable
C_normalized = (C_base - C_base.min()) / (C_base.max() - C_base.min() + 1e-10)

df['C'] = C_normalized

print(f"\nC (Coherence):")
print(f"  Leading eigenvalue λ: {leading_eigenvalue:.4f}")
print(f"  log10(λ): {np.log10(λ):.4f}")
print(f"  C (normalized): mean={df['C'].mean():.4f}")

# 2.6: χ (Sentient Potential) = αC + βF
df['chi'] = ALPHA * df['C'] + BETA * df['F']

print(f"\nχ (Sentient Potential) = αC + βF:")
print(f"  α={ALPHA}, β={BETA}")
print(f"  Mean: {df['chi'].mean():.4f}")
print(f"  Range: [{df['chi'].min():.4f}, {df['chi'].max():.4f}]")

# ─── PHASE 3: Category-Level SFH-SGP Analysis ─────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 3: Category-Level SFH-SGP Analysis")
print("-" * 40)

category_sfh = {}
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    category_sfh[cat] = {
        'n': len(cat_df),
        'Q_mean': cat_df['Q'].mean(),
        'Q_std': cat_df['Q'].std(),
        'F_mean': cat_df['F'].mean(),
        'C_mean': cat_df['C'].mean(),
        'chi_mean': cat_df['chi'].mean(),
        'chi_std': cat_df['chi'].std(),
    }
    for node in NODE_ORDER:
        category_sfh[cat][f'Q_{node}'] = cat_df[f'Q_{node}'].mean()
        category_sfh[cat][node] = cat_df[node].mean()

print("\nCategory SFH-SGP Summary:")
print(f"{'Category':<12} {'N':>4} {'Q':>8} {'F':>8} {'C':>8} {'χ':>8}")
print("-" * 50)
for cat in sorted(category_sfh.keys()):
    s = category_sfh[cat]
    print(f"{cat:<12} {s['n']:>4} {s['Q_mean']:>8.4f} {s['F_mean']:>8.4f} {s['C_mean']:>8.4f} {s['chi_mean']:>8.4f}")

# ─── PHASE 4: Langevin Dynamics Simulation ─────────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 4: Langevin Dynamics Simulation")
print("-" * 40)

def compute_chi_gradient(q, F_val, coactivation, alpha=ALPHA, beta=BETA):
    """
    Compute gradient of χ with respect to q
    ∇χ = α∇C + β∇F
    For C from co-activation: C ∝ q^T · M · q
    So ∇C = 2M · q
    """
    M = coactivation
    dC_dq = 2 * np.dot(M, q) / (np.linalg.norm(q) + 1e-10)
    dF_dq = np.zeros_like(q)
    dF_dq[4] = 1.0  # Only G5_dmn (index 4) contributes to F
    
    gradient = alpha * dC_dq + beta * dF_dq
    return gradient

def langevin_step(q, F_val, coactivation, T_eff, D=D):
    """Single step of Langevin dynamics: dq/dt = -∇χ + √(2D)ξ(t)"""
    grad = compute_chi_gradient(q, F_val, coactivation)
    noise = np.random.randn(*q.shape) * np.sqrt(2 * D * T_eff)
    return -grad + noise

def simulate_langevin(q_init, F_val, coactivation, k_max=K_MAX, 
                     cooling_rate=COOLING_RATE, threshold=CONVERGENCE_THRESHOLD):
    """Simulate Langevin dynamics until convergence"""
    q = q_init.copy()
    T_eff = 1.0
    trajectory = [q.copy()]
    k_values = [0]
    
    for k in range(k_max):
        q_new = q + langevin_step(q, F_val, coactivation, T_eff)
        q_new = np.clip(q_new, 0.7, 1.0)  # Keep in valid activation range
        
        delta_q = np.linalg.norm(q_new - q)
        trajectory.append(q_new.copy())
        k_values.append(k + 1)
        
        if delta_q < threshold:
            break
        
        T_eff *= cooling_rate  # Annealing
    
    return q, trajectory, k_values

# Simulate for each category (using category mean as initial state)
category_langevin = {}
print("\nSimulating Langevin dynamics per category:")
print(f"{'Category':<12} {'K-depth':>8} {'χ_before':>10} {'χ_after':>10} {'Δχ':>10}")
print("-" * 55)

for cat in sorted(category_sfh.keys()):
    # Initial state = category mean activations
    q_init = np.array([category_sfh[cat][node] for node in NODE_ORDER])
    F_val = category_sfh[cat]['F_mean']
    
    # Compute χ before
    C_before = np.dot(q_init, np.dot(coactivation_matrix, q_init)) / (np.linalg.norm(q_init)**2 + 1e-10)
    chi_before = ALPHA * C_before + BETA * F_val
    
    # Simulate
    q_final, trajectory, k_values = simulate_langevin(q_init, F_val, coactivation_matrix)
    
    # Compute χ after
    C_after = np.dot(q_final, np.dot(coactivation_matrix, q_final)) / (np.linalg.norm(q_final)**2 + 1e-10)
    chi_after = ALPHA * C_after + BETA * F_val
    
    delta_chi = chi_after - chi_before
    k_depth = len(k_values)
    
    category_langevin[cat] = {
        'K_depth': k_depth,
        'chi_before': chi_before,
        'chi_after': chi_after,
        'delta_chi': delta_chi,
        'q_init': q_init,
        'q_final': q_final,
        'trajectory': trajectory
    }
    
    print(f"{cat:<12} {k_depth:>8} {chi_before:>10.4f} {chi_after:>10.4f} {delta_chi:>+10.4f}")

# ─── PHASE 5: Hessian Analysis ────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 5: Hessian Analysis (Critical Points)")
print("-" * 40)

def compute_hessian(q, F_val, coactivation, alpha=ALPHA, beta=BETA):
    """Compute Hessian of χ: ∇²χ"""
    M = coactivation
    d2C_dq2 = 2 * M
    d2F_dq2 = np.zeros_like(M)
    d2F_dq2[4, 4] = 1e-6  # Small regularization for G5_dmn
    
    hessian = alpha * d2C_dq2 + beta * d2F_dq2
    return hessian

# Compute Hessian at overall mean state
q_mean = df[NODE_ORDER].mean().values
hessian = compute_hessian(q_mean, df['F'].mean(), coactivation_matrix)

eigenvalues_h, eigenvectors_h = np.linalg.eigh(hessian)

print("\nHessian Eigenvalue Spectrum:")
print(f"  Min eigenvalue: {eigenvalues_h.min():.6f}")
print(f"  Max eigenvalue: {eigenvalues_h.max():.6f}")
print(f"  Condition number: {eigenvalues_h.max() / (eigenvalues_h.min() + 1e-10):.2f}")

# Classify critical points
n_negative = np.sum(eigenvalues_h < 0)
n_zero = np.sum(np.abs(eigenvalues_h) < 1e-6)
n_positive = np.sum(eigenvalues_h > 0)

print(f"\nCritical Point Classification:")
print(f"  Negative eigenvalues (local maxima): {n_negative}")
print(f"  Near-zero eigenvalues (flat directions): {n_zero}")
print(f"  Positive eigenvalues (local minima): {n_positive}")

# Resonance Anchors = stable minima of χ
resonance_anchors = []
for i, ev in enumerate(eigenvalues_h):
    if ev > 0:  # Stable direction
        resonance_anchors.append({
            'index': i,
            'eigenvalue': ev,
            'direction': eigenvectors_h[:, i],
            'stability': ev / eigenvalues_h.sum()
        })

print(f"\nResonance Anchors (stable minima): {len(resonance_anchors)}")

# ─── PHASE 6: Torsion (τ) Detection ───────────────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 6: Torsion (τ) Detection")
print("-" * 40)

# Torsion τ = (1 - CV) × (1 - F_mean)
# High τ = locked circuit (high activation, low fertility, rigid)

node_torsion = {}
print("\nNode Torsion Analysis:")
print(f"{'Node':<12} {'CV':>8} {'F_mean':>8} {'τ':>8} {'Status':>12}")
print("-" * 50)

for node in NODE_ORDER:
    node_values = df[node].values
    CV = np.std(node_values) / (np.mean(node_values) + 1e-10)
    F_mean = df['F'].mean()
    
    tau = (1 - CV) * (1 - F_mean)
    tau = np.clip(tau, 0, 1)
    
    if tau > 0.5:
        status = "LOCKED"
    elif tau > 0.3:
        status = "MODERATE"
    else:
        status = "FLEXIBLE"
    
    node_torsion[node] = {
        'CV': CV,
        'F_mean': F_mean,
        'tau': tau,
        'status': status
    }
    
    print(f"{node:<12} {CV:>8.4f} {F_mean:>8.4f} {tau:>8.4f} {status:>12}")

# ─── PHASE 7: Topological Landscape Analysis ──────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 7: Topological Landscape Analysis")
print("-" * 40)

# Map each category to its basin of attraction
print("\nCategory → Basin Mapping:")
print(f"{'Category':<12} {'Dominant Anchor':>15} {'Basin Probability':>15}")
print("-" * 45)

for cat in sorted(category_langevin.keys()):
    cat_chi = category_langevin[cat]['chi_after']
    closest_anchor_idx = np.argmin([abs(e['eigenvalue'] - cat_chi) for e in resonance_anchors])
    
    # Simple basin assignment based on χ similarity
    basin_probs = []
    for anchor in resonance_anchors:
        similarity = np.exp(-abs(cat_chi - anchor['eigenvalue']))
        basin_probs.append(similarity)
    basin_probs = np.array(basin_probs) / sum(basin_probs)
    
    dominant_idx = np.argmax(basin_probs)
    
    print(f"{cat:<12} {'RA_' + str(dominant_idx):>15} {basin_probs[dominant_idx]:>15.2%}")

# ─── PHASE 8: Save Results ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 8: Saving Results")
print("-" * 40)

results = {
    'date': datetime.now().isoformat(),
    'parameters': {
        'alpha': ALPHA,
        'beta': BETA,
        'gamma': GAMMA,
        'D': D,
        'cooling_rate': COOLING_RATE,
        'k_max': K_MAX,
        'convergence_threshold': CONVERGENCE_THRESHOLD
    },
    'Q_analysis': {
        'overall_mean': float(df['Q'].mean()),
        'overall_std': float(df['Q'].std()),
        'range': [float(df['Q'].min()), float(df['Q'].max())]
    },
    'chi_analysis': {
        'overall_mean': float(df['chi'].mean()),
        'overall_std': float(df['chi'].std()),
        'range': [float(df['chi'].min()), float(df['chi'].max())]
    },
    'coherence': {
        'leading_eigenvalue': float(leading_eigenvalue),
        'log10_lambda': float(np.log10(λ)),
        'C_mean': float(df['C'].mean())
    },
    'hessian': {
        'eigenvalues': [float(ev) for ev in eigenvalues_h],
        'n_stable_directions': len(resonance_anchors),
        'condition_number': float(eigenvalues_h.max() / (eigenvalues_h.min() + 1e-10))
    },
    'category_sfh': category_sfh,
    'category_langevin': {k: {kk: vv for kk, vv in v.items() if kk not in ['trajectory', 'q_init', 'q_final']} 
                          for k, v in category_langevin.items()},
    'node_torsion': node_torsion
}

with open(f'{OUTPUT_DIR}/sfh_sgp_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUTPUT_DIR}/sfh_sgp_results.json")

# Save co-activation matrix
np.save(f'{OUTPUT_DIR}/coactivation_matrix.npy', coactivation_matrix)
print(f"Saved: {OUTPUT_DIR}/coactivation_matrix.npy")

# Save hessian
np.save(f'{OUTPUT_DIR}/hessian.npy', hessian)
print(f"Saved: {OUTPUT_DIR}/hessian.npy")

print("\n" + "=" * 70)
print("SFH-SGP CALCULATIONS COMPLETE")
print("=" * 70)
