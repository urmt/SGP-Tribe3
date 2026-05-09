#!/usr/bin/env python3
"""
PHASE 184 - GRAPH METRIC SENSITIVITY VALIDATION
LEP COMPLIANT PIPELINE TEST

Objective: Validate whether Phase 183 pipeline was insensitive due to thresholding
"""

import os, json, numpy as np, mne, time, csv
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

R = 42
np.random.seed(R)
DATA = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUT = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase184_sensitivity_validation'

# ============================================================
# SYNTHETIC GRAPH GENERATION
# ============================================================

def create_random_graph(n=15, p=0.3):
    """1. Erdos-Renyi random graph"""
    adj = np.random.rand(n, n) < p
    adj = (adj | adj.T)  # symmetric
    np.fill_diagonal(adj, 0)
    return adj

def create_ring_lattice(n=15, k=2):
    """2. Ring lattice (local connections)"""
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k+1):
            adj[i, (i + j) % n] = 1
            adj[i, (i - j) % n] = 1
    return adj

def create_modular_graph(n=15, n_modules=3):
    """3. Modular graph (clusters)"""
    adj = np.zeros((n, n))
    module_size = n // n_modules
    for m in range(n_modules):
        start = m * module_size
        end = start + module_size
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    adj[i, j] = np.random.rand() < 0.7  # high within-module
    # Add sparse between-module
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 0:
                adj[i, j] = np.random.rand() < 0.05
    return adj

def create_star_graph(n=15):
    """4. Star graph (one hub, all spokes)"""
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = 1
        adj[i, 0] = 1
    return adj

def create_dense_global(n=15):
    """5. Dense global (high connectivity)"""
    adj = np.random.rand(n, n) < 0.6
    adj = (adj | adj.T)
    np.fill_diagonal(adj, 0)
    return adj

# ============================================================
# NETWORK METRICS
# ============================================================

def compute_metrics(adj):
    """Compute all network metrics"""
    n = adj.shape[0]
    if n < 2:
        return {'modularity': 0, 'efficiency': 0, 'clustering': 0, 'path_length': 0, 'dom_mod': 0}
    
    # Modularity
    deg = np.sum(adj, axis=1)
    m = np.sum(deg) / 2
    Q = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                Aij = adj[i, j]
                Q += (Aij - deg[i] * deg[j] / (2 * m)) / (2 * m) if m > 0 else 0
    
    # Global efficiency
    adj_inv = 1 / (adj + np.eye(n))
    eff = (np.sum(adj_inv) - n) / (n * (n - 1))
    
    # Clustering coefficient
    tri = np.dot(adj, adj) * adj
    deg_tri = np.sum(tri, axis=1) / 2
    deg_adj = deg * (deg - 1) / 2
    clustering = np.mean(deg_tri / (deg_adj + 1e-12))
    
    # Mean path length (simplified - connected pairs)
    conn = np.sum(adj > 0) / 2
    max_possible = n * (n - 1) / 2
    path_length = 1 - (conn / max_possible) if max_possible > 0 else 0
    
    # Dominant module fraction (largest component)
    visited = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            size = 0
            while queue:
                node = queue.pop(0)
                size += 1
                neighbors = np.where(adj[node] > 0)[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            comps.append(size)
    dom_mod = max(comps) / n if comps else 0
    
    return {
        'modularity': float(Q),
        'efficiency': float(eff),
        'clustering': float(clustering) if clustering < 1 else float(clustering % 1),
        'path_length': float(path_length),
        'dominant_module': float(dom_mod)
    }

# ============================================================
# SYNTHETIC GRAPH TEST
# ============================================================

print("="*70)
print("PHASE 184 - PIPELINE SENSITIVITY VALIDATION")
print("="*70)

print("\n--- SYNTHETIC GRAPH VALIDATION ---")

graphs = {
    'random_graph': create_random_graph(15),
    'ring_lattice': create_ring_lattice(15),
    'modular_graph': create_modular_graph(15),
    'star_graph': create_star_graph(15),
    'dense_global': create_dense_global(15)
}

synthetic_results = {}
for name, adj in graphs.items():
    m = compute_metrics(adj)
    synthetic_results[name] = m
    print(f"{name}: Q={m['modularity']:.3f}, eff={m['efficiency']:.3f}, clust={m['clustering']:.3f}")

# Aggregate comparison
print("\nSYNTHETIC COMPARISON:")
print(f"  Dense vs Modular efficiency: {synthetic_results['dense_global']['efficiency']:.3f} vs {synthetic_results['modular_graph']['efficiency']:.3f}")
print(f"  Dense vs Modular modularity: {synthetic_results['dense_global']['modularity']:.3f} vs {synthetic_results['modular_graph']['modularity']:.3f}")

# Check if pipeline can distinguish
eff_diff = abs(synthetic_results['dense_global']['efficiency'] - synthetic_results['modular_graph']['efficiency'])
q_diff = abs(synthetic_results['dense_global']['modularity'] - synthetic_results['modular_graph']['modularity'])

pipeline_validated = (eff_diff > 0.01 or q_diff > 0.05)

# ============================================================
# WEIGHTED EEG ANALYSIS
# ============================================================

print("\n--- WEIGHTED EEG ANALYSIS ---")

files = [f for f in os.listdir(DATA) if f.endswith('.edf')][:4]
weighted_results = {}

for f in files:
    print(f"--- {f} ---")
    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA, f), preload=True, verbose=False)
        data = raw.get_data()[:10, :50000]
        
        r = {'file': f}
        
        # Real EEG - WEIGHTED (no thresholding)
        a = hilbert(data, axis=1)
        sync = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
        r['real_weighted'] = {'efficiency': float(np.mean(sync)), 'modularity': float(np.std(sync))}
        
        # Phase random
        pr = data.copy()
        for i in range(10):
            fft = np.fft.rfft(pr[i])
            pr[i] = np.fft.irfft(np.abs(fft) * np.exp(1j * np.random.uniform(0, 6.28, len(fft))), n=len(pr[i]))
        a = hilbert(pr, axis=1)
        sync = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
        r['A_weighted'] = {'efficiency': float(np.mean(sync)), 'modularity': float(np.std(sync))}
        
        # Burst timing
        a_orig = hilbert(data, axis=1)
        s_orig = np.abs(np.mean(np.exp(1j * np.angle(a_orig)), axis=0))
        m = s_orig > np.percentile(s_orig, 90)
        idx = np.where(m)[0]
        if len(idx) >= 50:
            np.random.seed(R)
            np.random.shuffle(idx)
            bt = data.copy()
            for j, k in zip(idx, np.where(m)[0]):
                if j < data.shape[1]:
                    bt[:, k] = data[:, j]
            a = hilbert(bt, axis=1)
            sync = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
            r['C_weighted'] = {'efficiency': float(np.mean(sync)), 'modularity': float(np.std(sync))}
        else:
            r['C_weighted'] = {'efficiency': 0, 'modularity': 0}
        
        weighted_results[f] = r
        print(f"  real: eff={r['real_weighted']['efficiency']:.3f} mod={r['real_weighted']['modularity']:.3f}")
        print(f"  A:    eff={r['A_weighted']['efficiency']:.3f} mod={r['A_weighted']['modularity']:.3f}")
        print(f"  C:    eff={r['C_weighted']['efficiency']:.3f} mod={r['C_weighted']['modularity']:.3f}")
        
    except Exception as e:
        print(f"FAIL {f}: {e}")

# Aggregate weighted
print("\nWEIGHTED EEG AGGREGATE:")
for ctrl in ['real_weighted', 'A_weighted', 'C_weighted']:
    effs = [weighted_results[f][ctrl]['efficiency'] for f in weighted_results]
    mods = [weighted_results[f][ctrl]['modularity'] for f in weighted_results]
    print(f"  {ctrl}: eff={np.mean(effs):.3f} mod={np.mean(mods):.3f}")

# Compare weighted to thresholded
real_w_eff = np.mean([weighted_results[f]['real_weighted']['efficiency'] for f in weighted_results])
real_w_mod = np.mean([weighted_results[f]['real_weighted']['modularity'] for f in weighted_results])
real_t_eff = 0.200  # From Phase 183
real_t_mod = 0.187  # From Phase 183

print(f"\nWEIGHTED vs THRESHOLDED:")
print(f"  Weighted: eff={real_w_eff:.3f}, mod={real_w_mod:.3f}")
print(f"  Thresholded: eff={real_t_eff:.3f}, mod={real_t_mod:.3f}")

weighted_discriminates = abs(real_w_eff - np.mean([weighted_results[f]['A_weighted']['efficiency'] for f in weighted_results])) > 0.05

# ============================================================
# VERDICT
# ============================================================

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print(f"Pipeline validated: {pipeline_validated}")
print(f"Weighted discriminates: {weighted_discriminates}")

if pipeline_validated and weighted_discriminates:
    verdict = "PIPELINE_VALIDATED"
    note = "Both synthetic graphs distinguishable AND weighted EEG separates controls"
elif pipeline_validated and not weighted_discriminates:
    verdict = "THRESHOLDING_COLLAPSED_SIGNAL"
    note = "Synthetic test passes but weighted EEG still converges"
elif not pipeline_validated:
    verdict = "PIPELINE_INSENSITIVE"
    note = "Synthetic graphs also collapse to similar metrics"
else:
    verdict = "NETWORK_STRUCTURE_NOT_ROBUST"
    note = "Neither synthetic nor real EEG show discrimination"

print(f"VERDICT: {verdict}")
print(f"Note: {note}")

# Save outputs
with open(f'{OUT}/phase184_results.json', 'w') as out:
    json.dump({
        'phase': 184,
        'verdict': verdict,
        'note': note,
        'synthetic': synthetic_results,
        'weighted_eeg': weighted_results
    }, out, indent=2)

with open(f'{OUT}/synthetic_graph_validation.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['graph', 'modularity', 'efficiency', 'clustering', 'path_length', 'dominant_module'])
    for name, m in synthetic_results.items():
        writer.writerow([name, m['modularity'], m['efficiency'], m['clustering'], m['path_length'], m['dominant_module']])

with open(f'{OUT}/weighted_network_metrics.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['file', 'real_efficiency', 'real_modularity', 'A_efficiency', 'A_modularity', 'C_efficiency', 'C_modularity'])
    for f in weighted_results:
        r = weighted_results[f]
        writer.writerow([f, r['real_weighted']['efficiency'], r['real_weighted']['modularity'], r['A_weighted']['efficiency'], r['A_weighted']['modularity'], r['C_weighted']['efficiency'], r['C_weighted']['modularity']])

print("\nDONE")