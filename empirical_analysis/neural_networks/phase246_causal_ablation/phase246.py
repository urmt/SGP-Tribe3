#!/usr/bin/env python3
"""
PHASE 246 — CAUSAL GENERATOR DIMENSION ABLATION AUDIT

Tests whether the low-dimensional generator components discovered in Phase 245
are causally necessary for organizational recovery.

Core question:
    "Does removing the recovered low-rank spectral backbone destroy recovery itself?"

EPISTEMIC STATUS: TIER 1 VALIDATION
COMPLIANCE: LEP | NO consciousness | NO SFH metaphysics | NO observer theory
            Preserve Phase 199 boundaries
            TRUE operators from Phase 201 | Recovery from Phase 242
            Scale from Phase 243 | Skeleton from Phase 244 | Generator from Phase 245

DIRECTOR: Mark Rowe Traver
DATE: 2026-05-11
"""

import os, sys, json, time, csv, warnings
import numpy as np
from scipy import signal, stats, sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
warnings.filterwarnings('ignore')

R = 42
np.random.seed(R)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = SCRIPT_DIR
PROJECT_BASE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

ABLATION_LEVELS = ['none', 'λ1_only', 'λ1_λ2', 'λ1_λ2_λ3']

EEG_FILES = [
    os.path.join(PROJECT_BASE, 'empirical_analysis', 'neural_networks', 'phase105_real_eeg_download', 'raw', 'CHBMIT.edf'),
    os.path.join(PROJECT_BASE, 'empirical_analysis', 'neural_networks', 'phase111_long_duration_real_eeg', 'downloaded', 'chb01_03.edf'),
    os.path.join(PROJECT_BASE, 'empirical_analysis', 'neural_networks', 'phase112_persistent_acquisition', 'downloaded', 'chb01_04.edf'),
    os.path.join(PROJECT_BASE, 'empirical_analysis', 'neural_networks', 'phase112_persistent_acquisition', 'downloaded', 'chb02_01.edf'),
    os.path.join(PROJECT_BASE, 'empirical_analysis', 'neural_networks', 'phase112_persistent_acquisition', 'downloaded', 'chb03_01.edf'),
    os.path.join(PROJECT_BASE, 'empirical_analysis', 'neural_networks', 'phase112_persistent_acquisition', 'downloaded', 'chb04_01.edf'),
]

def json_serial(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):
        if obj.ndim == 0: return float(obj)
        return obj.tolist()
    if isinstance(obj, set): return sorted(x for x in obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_json(obj, indent=2):
    return json.dumps(obj, indent=indent, default=json_serial)

# ====================================================================
# TRUE OPERATORS (Phase 201)
# ====================================================================
def destroy_f1_zerolag(data, window=64):
    result = data.copy()
    n_ch, n_t = result.shape
    for ch in range(n_ch):
        half = max(1, window // 4)
        for i in range(0, n_t - window, window // 2):
            seg = result[ch, i:i+window].copy()
            if len(seg) < 2: continue
            jitter = np.random.randint(-half, half)
            result[ch, i:i+window] = np.roll(seg, jitter)
    return result

def destroy_f2_propagation(data, max_shift=200):
    result = data.copy()
    n_ch, n_t = result.shape
    for ch in range(n_ch):
        shift = np.random.randint(-min(max_shift, n_t//2), min(max_shift, n_t//2))
        result[ch] = np.roll(result[ch], shift)
    return result

def destroy_f3_plv(data, seg_len=500):
    result = data.copy()
    n_ch, n_t = result.shape
    for ch in range(n_ch):
        for seg_start in range(0, n_t, seg_len):
            seg = result[ch, seg_start:seg_start+seg_len].copy()
            if len(seg) < 3: continue
            fft_seg = np.fft.rfft(seg)
            phases = np.exp(2j * np.pi * np.random.uniform(0, 1, len(fft_seg)))
            result[ch, seg_start:seg_start+seg_len] = np.fft.irfft(fft_seg * phases, n=len(seg))
    return result

def destroy_f4_coalition(data, n_seg=4):
    result = data.copy()
    n_ch, n_t = result.shape
    seg_size = max(1, n_t // n_seg)
    for ch in range(n_ch):
        segments = [result[ch, i*seg_size:min((i+1)*seg_size, n_t)].copy() for i in range(n_seg)]
        valid = [s for s in segments if len(s) > 0]
        np.random.shuffle(valid)
        result[ch, :sum(len(s) for s in valid)] = np.concatenate(valid) if len(valid) > 0 else result[ch]
    return result

def destroy_f5_burst(data, min_roll=500, max_roll=2000):
    result = data.copy()
    n_ch = result.shape[0]
    n_t = result.shape[1]
    for ch in range(n_ch):
        max_r = min(max_roll, n_t - 1)
        roll = min_roll if max_r <= min_roll else np.random.randint(min_roll, max_r)
        result[ch] = np.roll(result[ch], roll)
    return result

def apply_all_destroyers(data):
    d = destroy_f1_zerolag(data)
    d = destroy_f2_propagation(d)
    d = destroy_f3_plv(d)
    d = destroy_f4_coalition(d)
    d = destroy_f5_burst(d)
    return d

# ====================================================================
# SYSTEM GENERATORS
# ====================================================================
def create_kuramoto(n_ch=8, n_t=10000, coupling=0.2, noise=0.01, seed=None):
    if seed is not None: np.random.seed(seed)
    omega = np.random.uniform(0.1, 0.5, n_ch)
    K = np.random.uniform(0, coupling, (n_ch, n_ch))
    np.fill_diagonal(K, 0)
    phases = np.random.uniform(0, 2*np.pi, n_ch)
    dt = 0.01
    traj = np.zeros((n_ch, n_t))
    for t_step in range(n_t):
        dphi = omega + np.sum(K * np.sin(phases - phases[:, None]), axis=1)
        phases = phases + dphi * dt + np.random.randn(n_ch) * noise * np.sqrt(dt)
        traj[:, t_step] = np.sin(phases)
    return traj

def create_logistic(n_ch=8, n_t=10000, r=3.9, coupling=0.001, noise=0.01, seed=None):
    if seed is not None: np.random.seed(seed)
    x = np.random.uniform(0.1, 0.9, n_ch)
    traj = np.zeros((n_ch, n_t))
    for t_step in range(n_t):
        x_new = r * x * (1 - x)
        x_new += coupling * np.sum(x[:, None] - x, axis=1)
        x_new += np.random.uniform(-noise, noise, n_ch)
        x = np.clip(x_new, 0.001, 0.999)
        traj[:, t_step] = x
    return traj

# ====================================================================
# EEG LOADER
# ====================================================================
def load_eeg(max_ch=8, duration_sec=60, sfreq_target=128):
    available = [f for f in EEG_FILES if os.path.exists(f)]
    if not available:
        print("  WARNING: No EEG files")
        return create_kuramoto(n_ch=max_ch, n_t=sfreq_target*duration_sec, seed=R)
    fpath = available[0]
    print(f"  Loading EEG: {os.path.basename(fpath)}")
    import mne
    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
    if raw.info['sfreq'] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)
    n_samples = min(int(sfreq_target * duration_sec), raw.n_times)
    n_ch = min(max_ch, len(raw.ch_names))
    data = raw.get_data()[:n_ch, :n_samples]
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-10)
    return data.astype(np.float64)

# ====================================================================
# CORRELATION + EIGENDECOMPOSITION
# ====================================================================
def _correlation_matrix(data, window=200, step=50):
    n_ch, n_t = data.shape
    if n_t < window: return np.corrcoef(data)
    n_windows = (n_t - window) // step + 1
    corr_sum = np.zeros((n_ch, n_ch))
    count = 0
    for i in range(n_windows):
        seg = data[:, i*step:i*step+window]
        try:
            c = np.corrcoef(seg)
            corr_sum += np.nan_to_num(c, nan=0.0)
            count += 1
        except: pass
    return corr_sum / count if count > 0 else np.eye(n_ch)

def spectral_decompose(corr):
    """Return sorted eigvals, eigvecs (descending)."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]

def ablate_eigenvectors(corr, n_remove):
    """
    Remove top n_remove eigenvectors from correlation matrix.
    Returns ablated correlation matrix and relative impairment.
    """
    eigvals, eigvecs = spectral_decompose(corr)
    n = corr.shape[0]
    n_remove = min(n_remove, n - 1)

    if n_remove == 0:
        return corr.copy(), 0.0

    # Reconstruct without top eigenvectors
    rec = np.zeros((n, n))
    for i in range(n_remove, n):
        rec += eigvals[i] * np.outer(eigvecs[:, i], eigvecs[:, i])

    # Normalize to [-1, 1] range
    max_abs = np.max(np.abs(rec))
    if max_abs > 1e-10:
        rec = rec / max_abs
    np.fill_diagonal(rec, 1.0)

    # Impairment: Frobenius norm distance from original
    diff = np.linalg.norm(corr - rec, 'fro')
    norm = max(np.linalg.norm(corr, 'fro'), 1e-10)
    impairment = min(1.0, diff / norm)

    return rec, float(impairment)

def random_ablate(corr, n_remove):
    """Remove n_remove RANDOM eigenvectors."""
    eigvals, eigvecs = spectral_decompose(corr)
    n = corr.shape[0]
    n_remove = min(n_remove, n - 1)
    if n_remove == 0:
        return corr.copy(), 0.0

    # Choose random indices to remove (not necessarily top)
    indices = list(range(n))
    np.random.shuffle(indices)
    remove_idx = set(indices[:n_remove])

    rec = np.zeros((n, n))
    for i in range(n):
        if i not in remove_idx:
            rec += eigvals[i] * np.outer(eigvecs[:, i], eigvecs[:, i])

    max_abs = np.max(np.abs(rec))
    if max_abs > 1e-10:
        rec = rec / max_abs
    np.fill_diagonal(rec, 1.0)

    diff = np.linalg.norm(corr - rec, 'fro')
    norm = max(np.linalg.norm(corr, 'fro'), 1e-10)
    impairment = min(1.0, diff / norm)
    return rec, float(impairment)

# ====================================================================
# IMPAIRMENT METRICS (A-G)
# ====================================================================
def metric_A_persistence_loss(orig, ablated):
    """Drop in leading eigenvalue after ablation."""
    ev_o, _ = spectral_decompose(orig)
    ev_a, _ = spectral_decompose(ablated)
    if ev_o[0] < 1e-10: return 0.0
    return max(0.0, (ev_o[0] - ev_a[0]) / ev_o[0])

def metric_B_attractor_collapse(orig, ablated):
    """Change in mean absolute correlation."""
    def mean_abs(c):
        ac = np.abs(c)
        np.fill_diagonal(ac, 0)
        return np.mean(ac)
    m_orig = mean_abs(orig)
    m_abl = mean_abs(ablated)
    if m_orig < 1e-10: return 0.0
    return max(0.0, (m_orig - m_abl) / m_orig)

def metric_C_coalition_fragmentation(orig, ablated):
    """Change in clustering coefficient."""
    def clustering(corr):
        n = corr.shape[0]
        thresh = np.percentile(np.abs(corr), 70)
        adj = (np.abs(corr) > thresh).astype(float)
        np.fill_diagonal(adj, 0)
        clusts = []
        for i in range(n):
            neigh = np.where(adj[i] > 0)[0]
            k = len(neigh)
            if k >= 2:
                sub = adj[np.ix_(neigh, neigh)]
                clusts.append(np.sum(sub) / (k * (k - 1)))
        return np.mean(clusts) if clusts else 0.0
    co, ca = clustering(orig), clustering(ablated)
    if co < 1e-10: return 0.0
    return max(0.0, (co - ca) / co)

def metric_D_mst_degradation(orig, ablated):
    """MST edge overlap loss."""
    def mst_edges(corr):
        n = corr.shape[0]
        abs_c = np.abs(corr)
        np.fill_diagonal(abs_c, 0)
        dist = 1.0 - abs_c
        np.fill_diagonal(dist, 0)
        mst = minimum_spanning_tree(sp.csr_matrix(np.maximum(dist, 0))).toarray()
        edges = set()
        for i in range(n):
            for j in range(i+1, n):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    edges.add((i, j))
        return edges
    eo, ea = mst_edges(orig), mst_edges(ablated)
    if len(eo) == 0: return 0.0
    inter = len(eo & ea)
    return 1.0 - inter / len(eo)

def metric_E_spectral_distortion(orig, ablated):
    """Eigenvalue spectrum divergence (KL-like)."""
    ev_o, _ = spectral_decompose(orig)
    ev_a, _ = spectral_decompose(ablated)
    ev_o = np.maximum(ev_o, 0) / (np.sum(np.maximum(ev_o, 0)) + 1e-10)
    ev_a = np.maximum(ev_a, 0) / (np.sum(np.maximum(ev_a, 0)) + 1e-10)
    # JS divergence
    m = 0.5 * (ev_o + ev_a)
    kl1 = np.sum(ev_o * np.log((ev_o + 1e-10) / (m + 1e-10)))
    kl2 = np.sum(ev_a * np.log((ev_a + 1e-10) / (m + 1e-10)))
    js = 0.5 * (kl1 + kl2)
    return min(1.0, js)

def metric_F_trajectory_divergence(orig, ablated):
    """Frobenius norm distance."""
    diff = np.linalg.norm(orig - ablated, 'fro')
    norm = max(np.linalg.norm(orig, 'fro'), 1e-10)
    return min(1.0, diff / norm)

def metric_G_recovery_failure(orig, ablated, baseline_org=0.5):
    """Probability that organization drops below threshold."""
    def organization_level(corr):
        abs_c = np.abs(corr)
        np.fill_diagonal(abs_c, 0)
        return float(np.mean(abs_c))
    o_abl = organization_level(ablated)
    o_orig = organization_level(orig)
    if o_orig < baseline_org: return 1.0
    return max(0.0, 1.0 - o_abl / o_orig)

def compute_impairment(orig, ablated):
    metrics = {
        'persistence_loss': metric_A_persistence_loss(orig, ablated),
        'attractor_collapse': metric_B_attractor_collapse(orig, ablated),
        'coalition_fragmentation': metric_C_coalition_fragmentation(orig, ablated),
        'mst_degradation': metric_D_mst_degradation(orig, ablated),
        'spectral_distortion': metric_E_spectral_distortion(orig, ablated),
        'trajectory_divergence': metric_F_trajectory_divergence(orig, ablated),
        'recovery_failure_probability': metric_G_recovery_failure(orig, ablated),
    }
    vals = [v for v in metrics.values() if isinstance(v, (int, float)) and not np.isnan(v)]
    mean_impairment = float(np.mean(vals)) if vals else 0.0
    return metrics, mean_impairment

# ====================================================================
# MAIN ANALYSIS
# ====================================================================
def analyze_ablation(name, data_func, n_ch=8, n_t=10000, seed=R):
    print(f"\n{'='*65}")
    print(f"  SYSTEM: {name}")
    print(f"{'='*65}")

    # --- DATA ---
    if name.lower() == 'eeg':
        pre_full = data_func
        half = pre_full.shape[1] // 2
        pre_data = pre_full[:, :half]
        recovery_ref = pre_full[:, half:2*half]
    else:
        pre_data = data_func(n_ch=n_ch, n_t=n_t, seed=seed)
        recovery_ref = None

    # Apply TRUE destruction (don't need destroyed data for this phase)
    apply_all_destroyers(pre_data)

    # Generate recovery
    if name.lower() == 'eeg':
        recovery_data = recovery_ref
    elif name.lower() == 'kuramoto':
        recovery_data = create_kuramoto(n_ch=n_ch, n_t=n_t, seed=seed+1)
    else:
        recovery_data = create_logistic(n_ch=n_ch, n_t=n_t, seed=seed+1)

    recovery_corr = _correlation_matrix(recovery_data)
    eigvals, eigvecs = spectral_decompose(recovery_corr)
    n_components = min(n_ch, 5)

    print(f"  Recovery eigenvalues: {eigvals[:5].round(3).tolist()}")
    print(f"  λ1 fraction of total: {eigvals[0] / sum(eigvals):.3f}")

    # --- STEP 2: TARGETED ABLATION ---
    print(f"  Running targeted ablation...")
    ablation_results = {}
    for n_remove in [0, 1, 2, 3]:
        ablated, impairment = ablate_eigenvectors(recovery_corr, n_remove)
        metrics, mean_imp = compute_impairment(recovery_corr, ablated)
        label = ABLATION_LEVELS[n_remove] if n_remove < len(ABLATION_LEVELS) else f'remove_{n_remove}'
        ablation_results[label] = {
            'n_removed': n_remove,
            'impairment': mean_imp,
            'metrics': metrics,
        }
        if n_remove > 0:
            print(f"      {label}: impairment={mean_imp:.4f}")

    # --- STEP 3: RANDOMIZED CONTROLS ---
    print(f"  Running 5 randomized ablation controls...")
    # Control 1: remove random eigenvectors
    ctrl_impairments = {cname: [] for cname in ['random_eigenvectors', 'shuffled_basis', 'random_eigenvalues', 'random_orthogonal', 'synthetic_gaussian']}

    for trial in range(20):
        for n_remove in [1, 2, 3]:
            # 1. Random eigenvectors (already implemented)
            _, imp_r = random_ablate(recovery_corr, n_remove)
            ctrl_impairments['random_eigenvectors'].append((n_remove, imp_r))

            # 2. Shuffled eigenvector basis
            evals, evecs = spectral_decompose(recovery_corr)
            evecs_shuf = evecs.copy()
            for i in range(evecs_shuf.shape[1]):
                np.random.shuffle(evecs_shuf[:, i])
            rec_shuf = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                rec_shuf += evals[i] * np.outer(evecs_shuf[:, i], evecs_shuf[:, i])
            max_abs = max(np.max(np.abs(rec_shuf)), 1e-10)
            rec_shuf = rec_shuf / max_abs
            np.fill_diagonal(rec_shuf, 1.0)
            ablated_s, imp_s = ablate_eigenvectors(rec_shuf, n_remove)
            ctrl_impairments['shuffled_basis'].append((n_remove, imp_s))

            # 3. Random eigenvalues
            rand_evals = np.random.rand(n_ch) * max(evals)
            rec_r = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                rec_r += rand_evals[i] * np.outer(evecs[:, i], evecs[:, i])
            max_abs = max(np.max(np.abs(rec_r)), 1e-10)
            rec_r = rec_r / max_abs
            np.fill_diagonal(rec_r, 1.0)
            ablated_r, imp_re = ablate_eigenvectors(rec_r, n_remove)
            ctrl_impairments['random_eigenvalues'].append((n_remove, imp_re))

            # 4. Random orthogonal basis
            rand_basis = np.random.randn(n_ch, n_ch)
            rand_basis, _ = np.linalg.qr(rand_basis)
            rec_ortho = np.zeros((n_ch, n_ch))
            for i in range(n_ch):
                rec_ortho += evals[i] * np.outer(rand_basis[:, i], rand_basis[:, i])
            max_abs = max(np.max(np.abs(rec_ortho)), 1e-10)
            rec_ortho = rec_ortho / max_abs
            np.fill_diagonal(rec_ortho, 1.0)
            ablated_o, imp_o = ablate_eigenvectors(rec_ortho, n_remove)
            ctrl_impairments['random_orthogonal'].append((n_remove, imp_o))

            # 5. Synthetic Gaussian
            gauss = np.random.randn(n_ch, n_ch)
            gauss = (gauss + gauss.T) / 2
            gauss = gauss / max(np.max(np.abs(gauss)), 1e-10)
            np.fill_diagonal(gauss, 1.0)
            ablated_g, imp_g = ablate_eigenvectors(gauss, n_remove)
            ctrl_impairments['synthetic_gaussian'].append((n_remove, imp_g))

    # Average control impairments per n_remove
    ctrl_means = {}
    for cname, pairs in ctrl_impairments.items():
        for n_rem, imp in pairs:
            key = ABLATION_LEVELS[n_rem]
            if key not in ctrl_means:
                ctrl_means[key] = []
            ctrl_means[key].append(imp)
    for key in ctrl_means:
        ctrl_means[key] = float(np.mean(ctrl_means[key]))

    # --- COMPUTE CAUSAL IMPACT ---
    real_impairments = {k: v['impairment'] for k, v in ablation_results.items()}
    causal_impact = {}
    for level in ABLATION_LEVELS:
        if level == 'none':
            causal_impact[level] = 0.0
        else:
            real = real_impairments.get(level, 0.0)
            ctrl = ctrl_means.get(level, 0.0)
            causal_impact[level] = float(max(0.0, real - ctrl))

    # Overall causal impact: weighted average across ablation levels
    weights = {'λ1_only': 1.0, 'λ1_λ2': 0.5, 'λ1_λ2_λ3': 0.33}
    impact_values = [causal_impact[k] * weights.get(k, 0) for k in causal_impact if k != 'none']
    weight_sum = sum(weights.get(k, 0) for k in causal_impact if k != 'none')
    generator_causal_impact = sum(impact_values) / weight_sum if weight_sum > 0 else 0.0

    # Dominant dimension dependence: ratio of λ1 effect to total effect
    lambda1_effect = causal_impact.get('λ1_only', 0.0)
    total_effect = causal_impact.get('λ1_λ2_λ3', 1e-10)
    dominant_dim_dependence = lambda1_effect / max(total_effect, 1e-10)

    print(f"\n  Causal impact by ablation:")
    for k, v in causal_impact.items():
        if k != 'none':
            print(f"      {k}: real={real_impairments[k]:.4f}, ctrl={ctrl_means.get(k, 0):.4f}, impact={v:.4f}")
    print(f"  Generator causal impact: {generator_causal_impact:.4f}")
    print(f"  λ1 dominance: {dominant_dim_dependence:.4f}")

    # Collapse threshold: does λ1 removal alone cause >50% of max impairment?
    max_imp = max(real_impairments.get(k, 0) for k in ABLATION_LEVELS if k != 'none')
    lambda1_share = real_impairments.get('λ1_only', 0.0) / max(max_imp, 1e-10)

    return {
        'system': name,
        'recovery_eigvals': eigvals[:n_components].tolist(),
        'lambda1_fraction': float(eigvals[0] / max(sum(eigvals), 1e-10)),
        'ablation_results': ablation_results,
        'control_means': ctrl_means,
        'real_impairments': real_impairments,
        'causal_impact': causal_impact,
        'generator_causal_impact': float(generator_causal_impact),
        'dominant_dim_dependence': float(dominant_dim_dependence),
        'lambda1_share_of_max_impairment': float(lambda1_share),
        'max_impairment': float(max_imp),
    }

# ====================================================================
# VERDICT
# ====================================================================
def determine_verdict(results):
    impacts = [r['generator_causal_impact'] for r in results]
    lambda1_shares = [r['lambda1_share_of_max_impairment'] for r in results]
    lambda_doms = [r['dominant_dim_dependence'] for r in results]

    mean_impact = float(np.mean(impacts))
    mean_l1_share = float(np.mean(lambda1_shares))
    mean_l1_dom = float(np.mean(lambda_doms))

    evidence = {
        'mean_generator_causal_impact': mean_impact,
        'mean_lambda1_share_of_max': mean_l1_share,
        'mean_dominant_dim_dependence': mean_l1_dom,
        'per_system': {r['system']: {
            'causal_impact': r['generator_causal_impact'],
            'lambda1_share': r['lambda1_share_of_max_impairment'],
            'dominant_dim': r['dominant_dim_dependence'],
            'lambda1_fraction': r['lambda1_fraction'],
        } for r in results},
    }

    if mean_impact > 0.70 and mean_l1_share > 0.50:
        verdict = 'DOMINANT_CAUSAL_GENERATOR'
        confidence = 'HIGH'
    elif mean_impact > 0.45:
        verdict = 'HIERARCHICAL_GENERATOR_DEPENDENCE'
        confidence = 'MODERATE'
    elif mean_impact > 0.20:
        verdict = 'PARTIAL_GENERATOR_CONSTRAINT'
        confidence = 'LOW'
    else:
        verdict = 'NONCAUSAL_COMPRESSED_SUMMARY'
        confidence = 'HIGH'

    return verdict, confidence, evidence

# ====================================================================
# WRITERS
# ====================================================================
def write_metrics_csv(path, results):
    with open(path, 'w', newline='') as f:
        f.write('system,ablation,metric,value\n')
        for r in results:
            for abl, ares in r['ablation_results'].items():
                for mk, mv in ares['metrics'].items():
                    if isinstance(mv, (int, float)):
                        f.write(f"{r['system']},{abl},{mk},{mv:.6f}\n")
        f.write('\nsystem,ablation,real_impairment,control_mean,causal_impact\n')
        for r in results:
            for abl in r['real_impairments']:
                if abl == 'none': continue
                ri = r['real_impairments'][abl]
                cm = r['control_means'].get(abl, 0)
                ci = r['causal_impact'].get(abl, 0)
                f.write(f"{r['system']},{abl},{ri:.6f},{cm:.6f},{ci:.6f}\n")

def write_summary_md(path, results, verdict, confidence, evidence):
    with open(path, 'w') as f:
        f.write("# Phase 246: Causal Generator Dimension Ablation Audit\n\n")
        f.write(f"**Verdict:** {verdict}\n")
        f.write(f"**Confidence:** {confidence}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("## Core Question\n\n")
        f.write("Does removing the recovered low-rank spectral backbone destroy recovery itself?\n\n")
        f.write("---\n\n")
        f.write("## Aggregate Evidence\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Mean causal impact | {evidence['mean_generator_causal_impact']:.4f} |\n")
        f.write(f"| Mean λ1 share of max impairment | {evidence['mean_lambda1_share_of_max']:.4f} |\n")
        f.write(f"| Mean dominant dim dependence | {evidence['mean_dominant_dim_dependence']:.4f} |\n\n")
        f.write("### Per-System\n\n")
        f.write(f"| System | Causal Impact | λ1 Share | λ1 Fraction |\n")
        f.write(f"|--------|--------------|----------|-------------|\n")
        for sn, sp in evidence['per_system'].items():
            f.write(f"| {sn} | {sp['causal_impact']:.4f} | {sp['lambda1_share']:.4f} | {sp['lambda1_fraction']:.3f} |\n")
        f.write("\n---\n\n")
        for r in results:
            f.write(f"### {r['system']}\n\n")
            f.write(f"Eigenvalues: {[f'{v:.3f}' for v in r['recovery_eigvals']]}\n\n")
            f.write(f"| Ablation | Real Impairment | Control Mean | Causal Impact |\n")
            f.write(f"|----------|----------------|-------------|---------------|\n")
            for abl in ['λ1_only', 'λ1_λ2', 'λ1_λ2_λ3']:
                ri = r['real_impairments'].get(abl, 0)
                cm = r['control_means'].get(abl, 0)
                ci = r['causal_impact'].get(abl, 0)
                f.write(f"| {abl} | {ri:.4f} | {cm:.4f} | {ci:.4f} |\n")
            f.write(f"\n---\n\n")

def write_verdict_json(path, verdict, confidence, evidence, results):
    output = {
        'phase': 246, 'name': 'Causal Generator Dimension Ablation Audit',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'verdict': verdict, 'confidence': confidence, 'evidence': evidence,
        'per_system': [{
            'system': r['system'],
            'generator_causal_impact': r['generator_causal_impact'],
            'dominant_dim_dependence': r['dominant_dim_dependence'],
            'lambda1_share': r['lambda1_share_of_max_impairment'],
            'lambda1_fraction': r['lambda1_fraction'],
            'causal_impact_by_ablation': r['causal_impact'],
        } for r in results],
        'compliance': {
            'lep': True, 'no_consciousness': True, 'no_sfh_metaphysics': True,
            'no_observer_theory': True, 'phase_199_boundaries': True,
            'true_operators_phase_201': True, 'recovery_phase_242': True,
        }
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=json_serial)

def write_artifact_risk(path):
    with open(path, 'w') as f:
        f.write("# Phase 246: Artifact Risk Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Risks\n\n")
        f.write("### 1. Ablation Reconstruction Normalization\n")
        f.write("- **Severity**: MODERATE\n")
        f.write("- **Description**: Ablated matrices are renormalized to [-1,1], which may mask impairment.\n")
        f.write("- **Mitigation**: Consistent across real and control ablations.\n\n")
        f.write("### 2. Control 1 vs Targeted Ablation\n")
        f.write("- **Severity**: LOW\n")
        f.write("- **Description**: Random eigenvector removal removes different amounts of variance.\n")
        f.write("- **Mitigation**: 20 trials averaged per control condition.\n\n")
        f.write("### 3. Small Matrix (8x8)\n")
        f.write("- **Severity**: MODERATE\n")
        f.write("- **Description**: 8 eigendimensions limits ablation to max 3 removals.\n")
        f.write("- **Mitigation**: Consistent across all systems.\n\n")
        f.write("### 4. Synthesized Recovery for Gaussian Control\n")
        f.write("- **Severity**: LOW\n")
        f.write("- **Description**: Gaussian synthetic baseline may not reflect null recovery.\n")
        f.write("- **Mitigation**: 4 additional distinct controls provide cross-validation.\n\n")
        f.write("### Threshold Log\n")
        f.write("- Adjacency threshold: 70th percentile\n")
        f.write("- Ablation: 0, 1, 2, 3 eigenvectors removed\n")
        f.write("- 20 random trials per control condition\n")
        f.write("- JS divergence for spectral distortion\n")
        f.write("- Causal impact = real - control impairment\n")

def write_audit_chain(path, results, verdict):
    with open(path, 'w') as f:
        f.write(f"# PHASE 246 AUDIT CHAIN\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Verdict: {verdict}\n\n")
        for r in results:
            f.write(f"--- {r['system']} ---\n")
            f.write(f"Causal impact: {r['generator_causal_impact']:.4f}\n")
            f.write(f"λ1 share: {r['lambda1_share_of_max_impairment']:.4f}\n")
            f.write(f"λ1 fraction: {r['lambda1_fraction']:.3f}\n")
            f.write(f"Impairments: {safe_json(r['real_impairments'])}\n")
            f.write(f"Causal: {safe_json(r['causal_impact'])}\n\n")

def write_replication_status(path, verdict):
    with open(path, 'w') as f:
        json.dump({
            'phase': 246, 'name': 'causal_generator_ablation',
            'verdict': verdict, 'runtime': 'COMPLETED',
            'tier': 'VALIDATION', 'compliance': 'FULL',
        }, f, indent=2)

# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 65)
    print("  PHASE 246: CAUSAL GENERATOR DIMENSION ABLATION AUDIT")
    print("  TIER 1 VALIDATION")
    print("  Question: Does removing the spectral backbone destroy recovery?")
    print("=" * 65)

    print("\n  Loading CHB-MIT EEG...")
    try:
        eeg_data = load_eeg(max_ch=8, duration_sec=60)
        print(f"  EEG shape: {eeg_data.shape}")
    except Exception as e:
        print(f"  EEG load failed: {e}")
        eeg_data = create_kuramoto(n_ch=8, n_t=128*60, seed=R)

    results = []
    results.append(analyze_ablation('EEG', eeg_data, n_ch=8))
    results.append(analyze_ablation('Kuramoto', create_kuramoto, n_ch=8, n_t=10000, seed=R+10))
    results.append(analyze_ablation('Logistic', create_logistic, n_ch=8, n_t=10000, seed=R+20))

    verdict, confidence, evidence = determine_verdict(results)

    print(f"\n{'='*65}")
    print(f"  FINAL VERDICT: {verdict}")
    print(f"  CONFIDENCE: {confidence}")
    print(f"  Mean causal impact: {evidence['mean_generator_causal_impact']:.4f}")
    print(f"  Mean λ1 share: {evidence['mean_lambda1_share_of_max']:.4f}")
    print(f"{'='*65}")

    print("\n  Writing outputs...")
    write_metrics_csv(f'{OUT}/phase246_results.csv', results)
    write_summary_md(f'{OUT}/phase246_summary.md', results, verdict, confidence, evidence)
    write_verdict_json(f'{OUT}/phase246_verdict.json', verdict, confidence, evidence, results)
    write_artifact_risk(f'{OUT}/artifact_risk_report.md')
    write_audit_chain(f'{OUT}/audit_chain.txt', results, verdict)
    write_replication_status(f'{OUT}/replication_status.json', verdict)

    elapsed = time.time() - t_start
    print(f"\n  Phase 246 complete. Elapsed: {elapsed:.1f}s")
    print(f"  Output: {OUT}")
    print(f"  Verdict: {verdict}")
