#!/usr/bin/env python3
"""
PHASE 251 — SYNCHRONIZATION THRESHOLD UNIVERSALITY AUDIT

Investigates what determines the critical coupling threshold K* discovered in
Phase 250 (where organizational regeneration emerges).

Core question:
    "Is the critical synchronization threshold K* universal across network
     topologies, oscillator heterogeneity, noise levels, and system sizes?"

EPISTEMIC STATUS: TIER 1 VALIDATION — SYNCHRONIZATION MECHANISM
COMPLIANCE: LEP | NO consciousness | NO SFH metaphysics | NO observer theory

PREDECESSORS:
    Phase 250: Discovered K* = 0.02 for all-to-all Kuramoto with default params
    Phase 249: Established synchronization-specific generation
    Phase 248: Established dynamical attractor necessity (DNI)

HYPOTHESES:
    H0: K* is universal (~0.02) regardless of topology, heterogeneity, noise, size
    H1: K* depends on network topology (topology-specific threshold)
    H2: K* depends on oscillator heterogeneity (broader frequencies → higher K*)
    H3: K* depends on noise level (more noise → higher K* needed)
    H4: K* depends on system size (more oscillators → lower K*)

DIRECTOR: Mark Rowe Traver
DATE: 2026-05-11

ASSUMPTIONS:
    1. DNI threshold for regeneration = 0.25 (from Phase 248-250 standard)
    2. K* defined as smallest K where DNI > 0.25
    3. Default parameters: noise=0.01, freq_range=[0.1, 0.5], N=8
    4. Networks normalized by mean degree for comparability
    5. 10000 time steps per simulation, dt=0.01
    6. Fixed random seed R=42 with per-condition offsets
"""

import os, sys, json, time, csv, warnings
import numpy as np
from scipy import signal, stats
from scipy.optimize import curve_fit
warnings.filterwarnings('ignore')

R = 42
np.random.seed(R)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = SCRIPT_DIR
FIGURES = os.path.join(OUT, 'phase251_figures')
EXPORTS = os.path.join(OUT, 'outputs')
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(EXPORTS, exist_ok=True)

# ====================================================================
# GLOBAL PARAMETERS — ALL EXPLICITLY DECLARED
# ====================================================================
DT = 0.01                              # Integration timestep
N_TIME = 10000                         # Time steps per simulation
N_CH_DEFAULT = 8                       # Default system size
K_SWEEP = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
DNI_THRESHOLD = 0.25                   # Minimum DNI for "regeneration"
DEFAULT_NOISE = 0.01                   # Default noise amplitude
DEFAULT_FREQ_MIN = 0.1                 # Default min natural frequency
DEFAULT_FREQ_MAX = 0.5                 # Default max natural frequency
FREQ_HETEROGENEITY_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]  # Spread σ_ω
NOISE_VALUES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SIZE_VALUES = [4, 6, 8, 12, 16, 24, 32]

# ====================================================================
# SERIALIZATION
# ====================================================================
def json_serial(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):
        if obj.ndim == 0: return float(obj)
        return obj.tolist()
    if isinstance(obj, set): return sorted(x for x in obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# ====================================================================
# TRUE OPERATORS (Phase 201, inline)
# ====================================================================
def apply_all_destroyers(data):
    d = data.copy()
    for c in range(d.shape[0]):
        h = max(1, 64//4); w = 64
        for i in range(0, d.shape[1]-w, w//2):
            s = d[c, i:i+w].copy()
            if len(s) >= 2: d[c, i:i+w] = np.roll(s, np.random.randint(-h, h))
        d[c] = np.roll(d[c], np.random.randint(-min(200, d.shape[1]//2), min(200, d.shape[1]//2)))
        for seg_start in range(0, d.shape[1], 500):
            seg = d[c, seg_start:seg_start+500].copy()
            if len(seg) >= 3:
                f = np.fft.rfft(seg)
                d[c, seg_start:seg_start+500] = np.fft.irfft(f * np.exp(2j*np.pi*np.random.uniform(0,1,len(f))), n=len(seg))
        ss = max(1, d.shape[1]//4)
        segs = [d[c, i*ss:min((i+1)*ss, d.shape[1])].copy() for i in range(4)]
        v = [s for s in segs if len(s) > 0]; np.random.shuffle(v)
        d[c, :sum(len(s) for s in v)] = np.concatenate(v) if v else d[c]
        mr = min(2000, d.shape[1]-1)
        d[c] = np.roll(d[c], 500 if mr <= 500 else np.random.randint(500, mr))
    return d

# ====================================================================
# NETWORK GENERATORS (inline, no external dependencies)
# ====================================================================
def make_alltoall(N):
    K = np.ones((N, N), dtype=np.float64) / (N - 1)
    np.fill_diagonal(K, 0)
    return K

def make_erdos_renyi(N, p=0.4):
    K = (np.random.uniform(0, 1, (N, N)) < p).astype(np.float64)
    np.fill_diagonal(K, 0)
    deg = np.maximum(np.sum(K, axis=1), 1)
    K = K / deg[:, None]
    return K

def make_small_world(N, k_neighbors=2, p_rewire=0.3):
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(1, k_neighbors+1):
            K[i, (i+j)%N] = 1.0
            K[i, (i-j)%N] = 1.0
    edges = np.where(K > 0)
    for idx in range(len(edges[0])):
        i, j = edges[0][idx], edges[1][idx]
        if i < j and np.random.uniform() < p_rewire:
            K[i, j] = 0; K[j, i] = 0
            new_j = np.random.randint(N)
            while new_j == i or K[i, new_j] > 0:
                new_j = np.random.randint(N)
            K[i, new_j] = 1.0; K[new_j, i] = 1.0
    deg = np.maximum(np.sum(K, axis=1), 1)
    K = K / deg[:, None]
    return K

def make_scale_free(N, m_attach=2):
    K = np.zeros((N, N), dtype=np.float64)
    degrees = np.zeros(N, dtype=int)
    for i in range(1, N):
        probs = (degrees[:i] + 1) / np.sum(degrees[:i] + 1)
        targets = np.random.choice(i, size=min(m_attach, i), replace=False, p=probs)
        for t in targets:
            K[i, t] = 1.0; K[t, i] = 1.0
            degrees[i] += 1; degrees[t] += 1
    deg = np.maximum(np.sum(K, axis=1), 1)
    K = K / deg[:, None]
    return K

def make_ring_1d(N, k_neighbors=1):
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(1, k_neighbors+1):
            K[i, (i+j)%N] = 1.0; K[i, (i-j)%N] = 1.0
    deg = np.maximum(np.sum(K, axis=1), 1)
    K = K / deg[:, None]
    return K

NETWORK_GENERATORS = {
    'alltoall': make_alltoall,
    'erdos_renyi': lambda N: make_erdos_renyi(N, p=0.4),
    'small_world': lambda N: make_small_world(N, k_neighbors=2, p_rewire=0.3),
    'scale_free': lambda N: make_scale_free(N, m_attach=2),
    'ring_1d': lambda N: make_ring_1d(N, k_neighbors=1),
}

# ====================================================================
# KURAMOTO SIMULATOR
# ====================================================================
def simulate_kuramoto(N, K_matrix, noise=DEFAULT_NOISE, freq_min=DEFAULT_FREQ_MIN,
                       freq_max=DEFAULT_FREQ_MAX, n_t=N_TIME, dt=DT, seed=None):
    if seed is not None: np.random.seed(seed)
    omega = np.random.uniform(freq_min, freq_max, N)
    phases = np.random.uniform(0, 2*np.pi, N)
    traj = np.zeros((N, n_t), dtype=np.float64)
    for t in range(n_t):
        dphi = omega + np.sum(K_matrix * np.sin(phases - phases[:, None]), axis=1)
        phases = phases + dphi * dt + np.random.randn(N) * noise * np.sqrt(dt)
        traj[:, t] = np.sin(phases)
    return traj

# ====================================================================
# MEASUREMENT FUNCTIONS
# ====================================================================
def compute_org(data):
    nc, nt = data.shape
    w, step = 200, 50
    if nt < w: return 0.0
    nw = (nt - w) // step + 1
    orgs = []
    for i in range(nw):
        seg = data[:, i*step:i*step+w]
        c = np.corrcoef(seg)
        np.fill_diagonal(c, 0)
        try: orgs.append(float(np.linalg.eigvalsh(c)[-1]))
        except: orgs.append(0.0)
    return float(np.mean(orgs)) if orgs else 0.0

def compute_corr(data):
    nc, nt = data.shape; w, step = 200, 50
    if nt < w: return np.corrcoef(data)
    nw = (nt-w)//step+1; cs = np.zeros((nc,nc)); cnt = 0
    for i in range(nw):
        try:
            c = np.corrcoef(data[:, i*step:i*step+w])
            cs += np.nan_to_num(c, nan=0.0); cnt += 1
        except: pass
    return cs/cnt if cnt > 0 else np.eye(nc)

def corr_sim(c1, c2):
    t = np.triu_indices(c1.shape[0], k=1)
    v1, v2 = c1[t], c2[t]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10: return 0.0
    return float(np.dot(v1, v2)/(n1*n2))

def compute_order_R(data):
    nc, nt = data.shape
    analytic = signal.hilbert(data, axis=1)
    phases = np.angle(analytic)
    R_vals = np.array([abs(np.mean(np.exp(1j * phases[:, t]))) for t in range(nt)])
    return float(np.mean(R_vals))

# ====================================================================
# CONDITIONS (inline per-system)
# ====================================================================
def run_condition_A(N, K_mat, noise, freq_min, freq_max, n_t, seed):
    """Dynamics preserved: run standard Kuramoto with different seed."""
    return simulate_kuramoto(N, K_mat, noise, freq_min, freq_max, n_t, dt=DT, seed=seed)

def run_condition_B(data):
    """Structure preserved, dynamics destroyed: temporal shuffle each channel."""
    r = data.copy()
    for c in range(r.shape[0]):
        np.random.shuffle(r[c])
    return r

def run_condition_D(pre_data, N, K_mat, noise, freq_min, freq_max, n_t, seed_recovery):
    """True recovery: destroy then regenerate."""
    apply_all_destroyers(pre_data)
    return simulate_kuramoto(N, K_mat, noise, freq_min, freq_max, n_t, dt=DT, seed=seed_recovery)

# ====================================================================
# DNI COMPUTATION
# ====================================================================
def compute_dni(N, K_mat, noise, freq_min, freq_max, seed_base=R):
    """Full DNI pipeline for a given configuration."""
    pre = simulate_kuramoto(N, K_mat, noise, freq_min, freq_max, N_TIME, DT, seed_base)

    rec_d = run_condition_D(pre.copy(), N, K_mat, noise, freq_min, freq_max, N_TIME, seed_base+1)
    corr_d = compute_corr(rec_d)

    rec_a = run_condition_A(N, K_mat, noise, freq_min, freq_max, N_TIME, seed_base+200)
    sim_a = corr_sim(corr_d, compute_corr(rec_a))

    rec_b = run_condition_B(rec_d)
    sim_b = corr_sim(corr_d, compute_corr(rec_b))

    dni = sim_a - sim_b
    r_val = compute_order_R(rec_d)
    org_d = compute_org(rec_d)

    return {
        'DNI': float(dni),
        'R': float(r_val),
        'baseline_org': float(org_d),
        'A_sim': float(sim_a),
        'B_sim': float(sim_b),
    }

# ====================================================================
# FIND CRITICAL K*
# ====================================================================
def find_critical_K(N, K_gen_func, noise=DEFAULT_NOISE, freq_min=DEFAULT_FREQ_MIN,
                    freq_max=DEFAULT_FREQ_MAX, seed_base=R):
    """Sweep coupling strengths and find K* where DNI first exceeds threshold."""
    results = []
    k_star = K_SWEEP[-1]
    for K in K_SWEEP:
        K_mat = K_gen_func(N) * K
        res = compute_dni(N, K_mat, noise, freq_min, freq_max, seed_base + int(K*10000))
        res['K'] = float(K)
        results.append(res)
    # Find first K where DNI > threshold
    for r in results:
        if r['DNI'] > DNI_THRESHOLD:
            k_star = r['K']
            break
    return results, k_star

# ====================================================================
# EXPERIMENT 1: TOPOLOGY SWEEP
# ====================================================================
def experiment_topology():
    print("\n" + "="*65)
    print("  EXPERIMENT 1: NETWORK TOPOLOGY SWEEP")
    print("="*65)
    topology_results = {}
    for topo_name, topo_gen in sorted(NETWORK_GENERATORS.items()):
        K_gen = lambda N: topo_gen(N)
        results, k_star = find_critical_K(N_CH_DEFAULT, K_gen, seed_base=R+100)
        topology_results[topo_name] = {
            'k_star': float(k_star),
            'sweep': results,
        }
        print(f"  {topo_name:15s}: K*={k_star:.4f}")
    return topology_results

# ====================================================================
# EXPERIMENT 2: FREQUENCY HETEROGENEITY SWEEP
# ====================================================================
def experiment_heterogeneity():
    print("\n" + "="*65)
    print("  EXPERIMENT 2: FREQUENCY HETEROGENEITY SWEEP")
    print("="*65)
    het_results = {}
    for sigma in FREQ_HETEROGENEITY_VALUES:
        f_min = max(0.01, 0.3 - sigma/2)
        f_max = 0.3 + sigma/2
        K_gen = lambda N: make_alltoall(N)
        results, k_star = find_critical_K(N_CH_DEFAULT, K_gen, freq_min=f_min, freq_max=f_max, seed_base=R+200)
        het_results[sigma] = {
            'sigma': float(sigma),
            'freq_min': float(f_min),
            'freq_max': float(f_max),
            'k_star': float(k_star),
            'sweep': results,
        }
        print(f"  σ_ω={sigma:.3f} (f=[{f_min:.3f},{f_max:.3f}]): K*={k_star:.4f}")
    return het_results

# ====================================================================
# EXPERIMENT 3: NOISE SWEEP
# ====================================================================
def experiment_noise():
    print("\n" + "="*65)
    print("  EXPERIMENT 3: NOISE LEVEL SWEEP")
    print("="*65)
    noise_results = {}
    for noise_val in NOISE_VALUES:
        K_gen = lambda N: make_alltoall(N)
        results, k_star = find_critical_K(N_CH_DEFAULT, K_gen, noise=noise_val, seed_base=R+300)
        noise_results[noise_val] = {
            'noise': float(noise_val),
            'k_star': float(k_star),
            'sweep': results,
        }
        print(f"  η={noise_val:.4f}: K*={k_star:.4f}")
    return noise_results

# ====================================================================
# EXPERIMENT 4: SYSTEM SIZE SWEEP
# ====================================================================
def experiment_size():
    print("\n" + "="*65)
    print("  EXPERIMENT 4: SYSTEM SIZE SWEEP")
    print("="*65)
    size_results = {}
    for N in SIZE_VALUES:
        K_gen = lambda n: make_alltoall(n)
        results, k_star = find_critical_K(N, K_gen, seed_base=R+400)
        size_results[N] = {
            'N': int(N),
            'k_star': float(k_star),
            'sweep': results,
        }
        print(f"  N={N:2d}: K*={k_star:.4f}")
    return size_results

# ====================================================================
# VERDICT DETERMINATION
# ====================================================================
def determine_verdict(topo_res, het_res, noise_res, size_res):
    evidence = {}

    # Topology: compute K* range
    k_stars_topo = [v['k_star'] for v in topo_res.values()]
    k_range_topo = max(k_stars_topo) - min(k_stars_topo)
    k_mean_topo = float(np.mean(k_stars_topo))
    evidence['topology_k_star_range'] = float(k_range_topo)
    evidence['topology_k_star_mean'] = k_mean_topo
    evidence['topology_k_stars'] = {k: v['k_star'] for k, v in topo_res.items()}

    # Heterogeneity: does K* increase with sigma?
    sigmas = sorted(het_res.keys())
    k_stars_het = [het_res[s]['k_star'] for s in sigmas]
    if len(sigmas) >= 2 and np.std(sigmas) > 0:
        slope, _, r_val, _, _ = stats.linregress(sigmas, k_stars_het)
    else:
        slope, r_val = 0.0, 0.0
    evidence['heterogeneity_K_vs_sigma_slope'] = float(slope)
    evidence['heterogeneity_K_vs_sigma_r'] = float(r_val)
    evidence['heterogeneity_k_stars'] = {str(s): het_res[s]['k_star'] for s in sigmas}

    # Noise: does K* increase with noise?
    noises = sorted(noise_res.keys())
    k_stars_noise = [noise_res[n]['k_star'] for n in noises]
    if len(noises) >= 2 and np.std(noises) > 0:
        n_slope, _, n_r_val, _, _ = stats.linregress(noises, k_stars_noise)
    else:
        n_slope, n_r_val = 0.0, 0.0
    evidence['noise_K_vs_eta_slope'] = float(n_slope)
    evidence['noise_K_vs_eta_r'] = float(n_r_val)

    # Size: does K* decrease with N?
    ns = sorted(size_res.keys())
    k_stars_size = [size_res[n]['k_star'] for n in ns]
    if len(ns) >= 2 and np.std(ns) > 0:
        s_slope, _, s_r_val, _, _ = stats.linregress(np.log(ns), k_stars_size)
    else:
        s_slope, s_r_val = 0.0, 0.0
    evidence['size_K_vs_logN_slope'] = float(s_slope)
    evidence['size_K_vs_logN_r'] = float(s_r_val)

    evidence['topology_range'] = k_range_topo
    evidence['heterogeneity_sensitivity'] = slope
    evidence['noise_sensitivity'] = n_slope
    evidence['size_sensitivity'] = s_slope

    # Default baseline K* from all-to-all with default params
    baseline_k_star = topo_res.get('alltoall', {}).get('k_star', 0.02)

    # Verdict logic
    # Universal if K* varies by < 0.03 across topologies AND
    # heterogeneity/noise/size effects are weak
    if k_range_topo < 0.03 and abs(slope) < 0.1 and abs(n_slope) < 0.3:
        verdict = 'UNIVERSAL_SYNCHRONIZATION_THRESHOLD'
        confidence = 'HIGH'
    elif k_range_topo < 0.05:
        # Weak topology dependence
        if abs(slope) > 0.3 or abs(n_slope) > 0.5:
            verdict = 'HETEROGENEITY_MODULATED_THRESHOLD'
            confidence = 'MODERATE'
        else:
            verdict = 'APPROXIMATELY_UNIVERSAL_THRESHOLD'
            confidence = 'MODERATE'
    else:
        # Strong topology dependence
        verdict = 'TOPOLOGY_DEPENDENT_THRESHOLD'
        confidence = 'HIGH'

    return verdict, confidence, evidence

# ====================================================================
# PLOTTING
# ====================================================================
def plot_all(topo_res, het_res, noise_res, size_res):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1. Topology comparison: DNI(K) for each topology
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'alltoall': 'blue', 'erdos_renyi': 'green', 'small_world': 'orange',
                   'scale_free': 'red', 'ring_1d': 'purple'}
        for topo_name, tres in sorted(topo_res.items()):
            ks = [r['K'] for r in tres['sweep']]
            dnis = [r['DNI'] for r in tres['sweep']]
            ax.semilogx(ks, dnis, 'o-', color=colors.get(topo_name, 'gray'), lw=1.5, label=topo_name, markersize=4)
        ax.axhline(DNI_THRESHOLD, color='black', ls='--', lw=1, alpha=0.5, label=f'Threshold={DNI_THRESHOLD}')
        ax.set_xlabel('Coupling K'); ax.set_ylabel('DNI')
        ax.set_title('DNI(K) by Network Topology')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIGURES}/topology_dni_curves.png', dpi=150); plt.close()

        # 2. K* by topology
        fig, ax = plt.subplots(figsize=(8, 5))
        names = sorted(topo_res.keys())
        kstars = [topo_res[n]['k_star'] for n in names]
        colors_b = ['blue', 'green', 'orange', 'red', 'purple']
        ax.bar(range(len(names)), kstars, color=colors_b, alpha=0.7)
        ax.axhline(0.02, color='gray', ls='--', alpha=0.5, label='All-to-all baseline')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Critical K*'); ax.set_title('K* by Network Topology'); ax.legend()
        plt.tight_layout(); plt.savefig(f'{FIGURES}/k_star_by_topology.png', dpi=150); plt.close()

        # 3. K* vs heterogeneity
        fig, ax = plt.subplots(figsize=(8, 5))
        sigmas = sorted(het_res.keys())
        kstars = [het_res[s]['k_star'] for s in sigmas]
        ax.plot(sigmas, kstars, 'o-', color='purple', lw=2, markersize=6)
        ax.set_xlabel('Frequency Spread σ_ω'); ax.set_ylabel('Critical K*')
        ax.set_title('K* vs Oscillator Heterogeneity'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIGURES}/k_star_vs_heterogeneity.png', dpi=150); plt.close()

        # 4. K* vs noise
        fig, ax = plt.subplots(figsize=(8, 5))
        noises = sorted(noise_res.keys())
        kstars_n = [noise_res[n]['k_star'] for n in noises]
        ax.plot(noises, kstars_n, 'o-', color='red', lw=2, markersize=6)
        ax.set_xlabel('Noise η'); ax.set_ylabel('Critical K*')
        ax.set_title('K* vs Noise Level'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIGURES}/k_star_vs_noise.png', dpi=150); plt.close()

        # 5. K* vs system size
        fig, ax = plt.subplots(figsize=(8, 5))
        ns = sorted(size_res.keys())
        kstars_s = [size_res[n]['k_star'] for n in ns]
        ax.plot(ns, kstars_s, 'o-', color='green', lw=2, markersize=6)
        ax.set_xlabel('System Size N'); ax.set_ylabel('Critical K*')
        ax.set_title('K* vs System Size'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIGURES}/k_star_vs_size.png', dpi=150); plt.close()

        # 6. Synthesis: DNI(K) at selected noise levels
        fig, ax = plt.subplots(figsize=(10, 5))
        for eta in [0.001, 0.01, 0.05, 0.2]:
            if eta in noise_res:
                ks = [r['K'] for r in noise_res[eta]['sweep']]
                dnis = [r['DNI'] for r in noise_res[eta]['sweep']]
                ax.semilogx(ks, dnis, 'o-', lw=1.5, label=f'η={eta}', markersize=4)
        ax.axhline(DNI_THRESHOLD, color='black', ls='--', alpha=0.5)
        ax.set_xlabel('Coupling K'); ax.set_ylabel('DNI')
        ax.set_title('DNI(K) at Different Noise Levels')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIGURES}/dni_by_noise.png', dpi=150); plt.close()

        # 7. Synthesis: DNI(K) at selected sizes
        fig, ax = plt.subplots(figsize=(10, 5))
        for sel_n in [4, 8, 16, 32]:
            if sel_n in size_res:
                ks = [r['K'] for r in size_res[sel_n]['sweep']]
                dnis = [r['DNI'] for r in size_res[sel_n]['sweep']]
                ax.semilogx(ks, dnis, 'o-', lw=1.5, label=f'N={sel_n}', markersize=4)
        ax.axhline(DNI_THRESHOLD, color='black', ls='--', alpha=0.5)
        ax.set_xlabel('Coupling K'); ax.set_ylabel('DNI')
        ax.set_title('DNI(K) at Different System Sizes')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{FIGURES}/dni_by_size.png', dpi=150); plt.close()

        print(f"  Figures saved to {FIGURES}/")
    except Exception as e:
        print(f"  WARNING: Figure generation failed: {e}")

# ====================================================================
# OUTPUT WRITERS
# ====================================================================
def write_all_outputs(topo_res, het_res, noise_res, size_res, verdict, conf, evidence):
    # CSV: sweep data
    with open(f'{EXPORTS}/phase251_results.csv', 'w', newline='') as f:
        f.write('experiment,condition,K,DNI,R,baseline_org\n')
        for exp_name, exp_data in [('topology', topo_res), ('heterogeneity', het_res),
                                    ('noise', noise_res), ('size', size_res)]:
            for cond_name, cond_data in exp_data.items():
                for r in cond_data['sweep']:
                    f.write(f"{exp_name},{cond_name},{r['K']:.6f},{r['DNI']:.6f},{r['R']:.6f},{r['baseline_org']:.6f}\n")
    # CSV: K* summary
    with open(f'{EXPORTS}/k_star_summary.csv', 'w', newline='') as f:
        f.write('experiment,condition,k_star\n')
        for exp_name, exp_data in [('topology', topo_res), ('heterogeneity', het_res),
                                    ('noise', noise_res), ('size', size_res)]:
            for cond_name, cond_data in exp_data.items():
                f.write(f"{exp_name},{cond_name},{cond_data['k_star']:.6f}\n")

    # JSON: full results
    with open(f'{EXPORTS}/phase251_verdict.json', 'w') as f:
        out = {
            'phase': 251, 'name': 'Synchronization Threshold Universality Audit',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'verdict': verdict, 'confidence': conf, 'evidence': evidence,
            'parameters': {
                'K_sweep': K_SWEEP, 'DNI_threshold': DNI_THRESHOLD,
                'default_noise': DEFAULT_NOISE, 'default_freq_range': [DEFAULT_FREQ_MIN, DEFAULT_FREQ_MAX],
                'default_N': N_CH_DEFAULT, 'N_time': N_TIME, 'dt': DT,
            },
            'compliance': {'lep': True, 'no_consciousness': True, 'no_sfh_metaphysics': True},
        }
        json.dump(out, f, indent=2, default=json_serial)

    # Summary markdown
    with open(f'{OUT}/phase251_summary.md', 'w') as f:
        f.write("# Phase 251: Synchronization Threshold Universality Audit\n\n")
        f.write(f"**Verdict:** {verdict}\n**Confidence:** {conf}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
        f.write("## Baseline (All-to-All, Default Params)\n\n")
        f.write(f"Phase 250 found K* ≈ 0.02 for all-to-all Kuramoto (N=8, η=0.01, σ_ω≈0.12)\n\n")
        f.write("## Topology Results\n\n")
        f.write("| Topology | K* |\n|----------|-----|\n")
        for n, v in sorted(topo_res.items()):
            f.write(f"| {n} | {v['k_star']:.4f} |\n")
        f.write(f"\nRange: {evidence['topology_k_star_range']:.4f} | Mean: {evidence['topology_k_star_mean']:.4f}\n\n")
        f.write("## Heterogeneity Results\n\n")
        f.write("| σ_ω | K* |\n|-----|-----|\n")
        for s in sorted(het_res.keys()):
            f.write(f"| {s:.3f} | {het_res[s]['k_star']:.4f} |\n")
        f.write(f"\nSlope: {evidence['heterogeneity_K_vs_sigma_slope']:.3f}\n\n")
        f.write("## Noise Results\n\n")
        f.write("| η | K* |\n|-----|-----|\n")
        for n in sorted(noise_res.keys()):
            f.write(f"| {n:.4f} | {noise_res[n]['k_star']:.4f} |\n")
        f.write(f"\nSlope: {evidence['noise_K_vs_eta_slope']:.3f}\n\n")
        f.write("## Size Results\n\n")
        f.write("| N | K* |\n|-----|-----|\n")
        for n in sorted(size_res.keys()):
            f.write(f"| {n} | {size_res[n]['k_star']:.4f} |\n")
        f.write(f"\nSlope (log N): {evidence['size_K_vs_logN_slope']:.3f}\n\n---\n\n")
        f.write("## Interpretation\n\n")
        f.write(f"1. Topology dependence: K* range = {evidence['topology_k_star_range']:.4f}\n")
        f.write(f"2. Heterogeneity sensitivity: slope = {evidence['heterogeneity_sensitivity']:.3f}\n")
        f.write(f"3. Noise sensitivity: slope = {evidence['noise_sensitivity']:.3f}\n")
        f.write(f"4. Size scaling: slope(log N) = {evidence['size_sensitivity']:.3f}\n\n")
        f.write("ASSUMPTIONS:\n")
        f.write(f"- DNI threshold = {DNI_THRESHOLD}\n")
        f.write(f"- Default noise = {DEFAULT_NOISE}\n")
        f.write(f"- Default frequency range = [{DEFAULT_FREQ_MIN}, {DEFAULT_FREQ_MAX}]\n")
        f.write(f"- Default N = {N_CH_DEFAULT}\n")
        f.write("- Network coupling matrices normalized by mean degree\n\nCOMPLIANCE: LEP\n")

    # Replication status
    with open(f'{EXPORTS}/replication_status.json', 'w') as f:
        json.dump({
            'phase': 251, 'name': 'threshold_universality',
            'verdict': verdict, 'runtime': 'COMPLETED', 'tier': 'VALIDATION',
            'compliance': 'FULL', 'parameters_used': {
                'K_sweep': K_SWEEP, 'threshold': DNI_THRESHOLD,
                'noise_values': NOISE_VALUES, 'heterogeneity_values': FREQ_HETEROGENEITY_VALUES,
                'size_values': SIZE_VALUES,
            }
        }, f, indent=2)

    # Audit chain
    with open(f'{EXPORTS}/audit_chain.txt', 'w') as f:
        f.write(f"# PHASE 251 AUDIT CHAIN\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nVerdict: {verdict}\n\n")
        f.write(f"Topology K* range: {evidence['topology_k_star_range']:.4f}\n")
        f.write(f"Heterogeneity slope: {evidence['heterogeneity_K_vs_sigma_slope']:.3f}\n")
        f.write(f"Noise slope: {evidence['noise_K_vs_eta_slope']:.3f}\n")
        f.write(f"Size slope (log N): {evidence['size_K_vs_logN_slope']:.3f}\n\n")
        f.write("All parameters, thresholds, and assumptions explicitly declared in script header.\n")

# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    t_start = time.time()
    print("="*65)
    print("  PHASE 251: SYNCHRONIZATION THRESHOLD UNIVERSALITY AUDIT")
    print("  TIER 1 VALIDATION — SYNCHRONIZATION MECHANISM")
    print("  Investigating what determines critical K*")
    print("="*65)
    print(f"\n  Global thresholds:")
    print(f"    DNI threshold for regeneration = {DNI_THRESHOLD}")
    print(f"    Default noise η = {DEFAULT_NOISE}")
    print(f"    Default freq range = [{DEFAULT_FREQ_MIN}, {DEFAULT_FREQ_MAX}]")
    print(f"    Default system size N = {N_CH_DEFAULT}")
    print(f"    K sweep = [{K_SWEEP[0]}, ..., {K_SWEEP[-1]}] ({len(K_SWEEP)} values)")

    print(f"\n  Running Experiment 1: Topology sweep (5 topologies, {len(K_SWEEP)} coupling values each)...")
    topo_res = experiment_topology()

    print(f"\n  Running Experiment 2: Frequency heterogeneity sweep ({len(FREQ_HETEROGENEITY_VALUES)} values)...")
    het_res = experiment_heterogeneity()

    print(f"\n  Running Experiment 3: Noise level sweep ({len(NOISE_VALUES)} values)...")
    noise_res = experiment_noise()

    print(f"\n  Running Experiment 4: System size sweep ({len(SIZE_VALUES)} values)...")
    size_res = experiment_size()

    verdict, conf, evidence = determine_verdict(topo_res, het_res, noise_res, size_res)

    print(f"\n{'='*65}")
    print(f"  FINAL VERDICT: {verdict}")
    print(f"  CONFIDENCE: {conf}")
    print(f"  K* range across topologies: {evidence['topology_k_star_range']:.4f}")
    print(f"  Heterogeneity sensitivity: {evidence['heterogeneity_K_vs_sigma_slope']:.3f}")
    print(f"  Noise sensitivity: {evidence['noise_K_vs_eta_slope']:.3f}")
    print(f"  Size scaling: {evidence['size_K_vs_logN_slope']:.3f}")
    print(f"{'='*65}")

    print(f"\n  Writing outputs to {EXPORTS}/ and {FIGURES}/...")
    plot_all(topo_res, het_res, noise_res, size_res)
    write_all_outputs(topo_res, het_res, noise_res, size_res, verdict, conf, evidence)

    elapsed = time.time() - t_start
    print(f"\n  Phase 251 complete. Elapsed: {elapsed:.1f}s")
    print(f"  Verdict: {verdict}")
    print(f"  Output directory: {OUT}")
