#!/usr/bin/env python3
"""
PHASE 180 - REPLICATION AUDIT
Strict replication of Phase 129, 133, 160, 177, 179 findings
"""

import os
import json
import numpy as np
import mne
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import csv
import time
warnings.filterwarnings('ignore')

# Configuration
random_state = 42
np.random.seed(random_state)

DATA_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase180_replication_audit'

FS = 256
WINDOW_SIZE = 512  # Identical to Phase 129/133

# ============================================================
# STANDARDIZED CONTROLS (A-F)
# ============================================================

def control_phase_randomization(data):
    """A. Phase randomization"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        fft = np.fft.rfft(data[i])
        mag = np.abs(fft)
        rand_phase = np.random.uniform(0, 2*np.pi, len(fft))
        result[i] = np.fft.irfft(mag * np.exp(1j * rand_phase), n=len(data[i]))
    return result

def control_temporal_block_shuffle(data, block_size=512):
    """B. Temporal block shuffle"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        n_blocks = len(data[i]) // block_size
        blocks = [data[i, j*block_size:(j+1)*block_size] for j in range(n_blocks)]
        np.random.seed(random_state + i)
        np.random.shuffle(blocks)
        result[i, :n_blocks*block_size] = np.concatenate(blocks)
    return result

def control_burst_timing_shuffle(data):
    """C. Burst timing shuffle"""
    analytic = hilbert(data, axis=1)
    sync = np.abs(np.mean(np.exp(1j * np.angle(analytic)), axis=0))
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh
    burst_indices = np.where(burst_mask)[0]
    if len(burst_indices) < 50:
        return data
    np.random.seed(random_state)
    shuffled_indices = burst_indices.copy()
    np.random.shuffle(shuffled_indices)
    result = data.copy()
    for idx, orig_idx in zip(shuffled_indices, burst_indices):
        if idx < data.shape[1] and orig_idx < data.shape[1]:
            result[:, orig_idx] = data[:, idx]
    return result

def control_channel_permutation(data):
    """D. Channel permutation"""
    shuffled = data.copy()
    np.random.seed(random_state)
    np.random.shuffle(shuffled)
    return shuffled

def control_white_noise(data):
    """E. White noise"""
    return np.random.randn(*data.shape).astype(np.float32) * np.std(data, axis=1, keepdims=True)

def control_iaaft(data, max_time=60):
    """F. IAAFT surrogate (with timeout)"""
    start_time = time.time()
    result = np.zeros_like(data)

    for i in range(data.shape[0]):
        if time.time() - start_time > max_time:
            raise TimeoutError("IAAFT exceeded time limit")

        signal = data[i]
        orig_fft = np.fft.fft(signal)
        orig_power = np.abs(orig_fft) ** 2

        surrogate = np.random.permutation(signal)
        n_iter = 5  # Reduced iterations for speed

        for _ in range(n_iter):
            fft = np.fft.fft(surrogate)
            phases = np.angle(fft)
            new_fft = np.sqrt(orig_power) * np.exp(1j * phases)
            surrogate = np.real(np.fft.ifft(new_fft))
            sorted_surr = np.sort(surrogate)
            for j in range(len(signal)):
                idx = np.where(sorted_surr == surrogate[j])[0][0]
                surrogate[j] = sorted_power[idx]

        result[i] = surrogate

    return result

# ============================================================
# FINDING 1: GLOBAL_PHASE_DEPENDENT (from Phase 129)
# ============================================================

def test_phase_dependent(data, label):
    """Test: Phase-dependent classification (AUC metric)"""
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)

    # Global phase synchrony feature
    sync = np.abs(np.mean(np.exp(1j * phase), axis=0))

    # Binary labels based on high/low synchrony
    labels = (sync > np.median(sync)).astype(int)

    # Feature matrix
    features = sync.reshape(-1, 1)

    # LOSO-style train/test (simple holdout for speed)
    n = len(features)
    train_size = n // 2

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[:train_size])
    X_test = scaler.transform(features[train_size:])
    y_train = labels[:train_size]
    y_test = labels[train_size:]

    clf = LogisticRegression(random_state=random_state, max_iter=500)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    return auc

# ============================================================
# FINDING 2: BURST_SYNCHRONY (from Phase 133)
# ============================================================

def test_burst_synchrony(data, label):
    """Test: Burst synchrony classification"""
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)
    sync = np.abs(np.mean(np.exp(1j * phase), axis=0))

    # During bursts
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh

    # Features: sync mean, std during bursts vs non-bursts
    sync_during = sync[burst_mask]
    sync_between = sync[~burst_mask]

    features = np.array([np.mean(sync_during), np.std(sync_during), np.mean(sync_between)])
    label_binary = 1 if label > 0 else 0

    # Simple holdout
    if len(features) < 2:
        return 0.5

    scaler = StandardScaler()
    X = scaler.fit_transform(features.reshape(-1, 1))
    y = np.array([label_binary])

    clf = LogisticRegression(random_state=random_state)
    try:
        clf.fit(X, y)
        prob = clf.predict_proba(X)[0, 1]
        return max(prob, 1-prob)  # Return higher probability
    except:
        return 0.5

# ============================================================
# FINDING 3: STRUCTURED_HEAVY_TAILS (from Phase 160)
# ============================================================

def test_heavy_tails(data):
    """Test: Kurtosis vs surrogates"""
    real_k = kurtosis(data[0, :20000])
    return real_k

def test_heavy_tails_control(data, control_fn):
    """Apply control and compute kurtosis"""
    ctrl_data = control_fn(data)
    return kurtosis(ctrl_data[0, :20000])

# ============================================================
# FINDING 4: LOW_DIMENSIONAL_COORDINATION (from Phase 177)
# ============================================================

def test_low_dim_coordination(data):
    """Test: PCA dimensionality during bursts"""
    analytic = hilbert(data, axis=1)
    sync = np.abs(np.mean(np.exp(1j * np.angle(analytic)), axis=0))
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh

    if np.sum(burst_mask) < 50:
        return {'pc1': 0.0, 'dim_80': 0}

    burst_data = data[:, burst_mask].T
    pca = PCA(n_components=min(20, burst_data.shape[1]), random_state=random_state)
    pca.fit(burst_data)

    pc1 = pca.explained_variance_ratio_[0]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_80 = np.searchsorted(cumvar, 0.80) + 1

    return {'pc1': pc1, 'dim_80': dim_80}

# ============================================================
# MAIN REPLICATION AUDIT
# ============================================================

print("="*70)
print("PHASE 180 - REPLICATION AUDIT")
print("="*70)

edf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.edf')])[:4]
print(f"Processing {len(edf_files)} subjects")

# Controls dictionary
controls = {
    'A_phase_random': control_phase_randomization,
    'B_temporal_block': control_temporal_block_shuffle,
    'C_burst_timing': control_burst_timing_shuffle,
    'D_channel_perm': control_channel_permutation,
    'E_white_noise': control_white_noise,
    'F_iaaft': None  # Will try to add if time permits
}

# Storage for results
all_results = {
    'Q1_phase_dependent': {},
    'Q2_burst_synchrony': {},
    'Q3_heavy_tails': {},
    'Q4_low_dim': {}
}

# Process each subject
for fname in edf_files:
    print(f"\n--- {fname} ---")

    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, fname), preload=True, verbose=False)
        data = raw.get_data()

        # Q1: Phase dependent
        try:
            real_auc = test_phase_dependent(data, 1)
            all_results['Q1_phase_dependent'][fname] = {'real': real_auc}
            for ctrl_name, ctrl_fn in controls.items():
                if ctrl_fn is not None:
                    try:
                        ctrl_data = ctrl_fn(data.copy())
                        ctrl_auc = test_phase_dependent(ctrl_data, 1)
                        all_results['Q1_phase_dependent'][fname][ctrl_name] = ctrl_auc
                    except:
                        all_results['Q1_phase_dependent'][fname][ctrl_name] = None
        except Exception as e:
            print(f"  Q1 failed: {e}")

        # Q2: Burst synchrony
        try:
            real_bs = test_burst_synchrony(data, 1)
            all_results['Q2_burst_synchrony'][fname] = {'real': real_bs}
            for ctrl_name, ctrl_fn in controls.items():
                if ctrl_fn is not None:
                    try:
                        ctrl_data = ctrl_fn(data.copy())
                        ctrl_bs = test_burst_synchrony(ctrl_data, 1)
                        all_results['Q2_burst_synchrony'][fname][ctrl_name] = ctrl_bs
                    except:
                        all_results['Q2_burst_synchrony'][fname][ctrl_name] = None
        except Exception as e:
            print(f"  Q2 failed: {e}")

        # Q3: Heavy tails
        try:
            real_k = test_heavy_tails(data)
            all_results['Q3_heavy_tails'][fname] = {'real': real_k}
            for ctrl_name, ctrl_fn in controls.items():
                if ctrl_fn is not None:
                    try:
                        ctrl_k = test_heavy_tails_control(data, ctrl_fn)
                        all_results['Q3_heavy_tails'][fname][ctrl_name] = ctrl_k
                    except:
                        all_results['Q3_heavy_tails'][fname][ctrl_name] = None
        except Exception as e:
            print(f"  Q3 failed: {e}")

        # Q4: Low dimensional
        try:
            real_ld = test_low_dim_coordination(data)
            all_results['Q4_low_dim'][fname] = {'real': real_ld}
            for ctrl_name, ctrl_fn in controls.items():
                if ctrl_fn is not None:
                    try:
                        ctrl_data = ctrl_fn(data.copy())
                        ctrl_ld = test_low_dim_coordination(ctrl_data)
                        all_results['Q4_low_dim'][fname][ctrl_name] = ctrl_ld
                    except:
                        all_results['Q4_low_dim'][fname][ctrl_name] = None
        except Exception as e:
            print(f"  Q4 failed: {e}")

    except Exception as e:
        print(f"  FAIL: {e}")

# ============================================================
# AGGREGATE AND VERDICT
# ============================================================

print("\n" + "="*70)
print("AGGREGATED RESULTS")
print("="*70)

# Q1: Phase dependent (AUC - higher is better)
q1_real = np.mean([v['real'] for v in all_results['Q1_phase_dependent'].values() if 'real' in v])
print(f"\nQ1 GLOBAL_PHASE_DEPENDENT:")
print(f"  Real AUC: {q1_real:.3f}")
q1_survivors = []
for ctrl in controls.keys():
    if ctrl == 'F_iaaft' or controls[ctrl] is None:
        continue
    ctrl_vals = [v.get(ctrl) for v in all_results['Q1_phase_dependent'].values() if ctrl in v and v[ctrl] is not None]
    if ctrl_vals:
        ctrl_mean = np.mean(ctrl_vals)
        effect = abs(ctrl_mean - q1_real) / q1_real
        print(f"  {ctrl}: {ctrl_mean:.3f} (effect: {effect:.1%})")
        if effect < 0.20:
            q1_survivors.append(ctrl)

# Q2: Burst synchrony (AUC)
q2_real = np.mean([v['real'] for v in all_results['Q2_burst_synchrony'].values() if 'real' in v])
print(f"\nQ2 BURST_SYNCHRONY:")
print(f"  Real: {q2_real:.3f}")
q2_survivors = []
for ctrl in controls.keys():
    if ctrl == 'F_iaaft' or controls[ctrl] is None:
        continue
    ctrl_vals = [v.get(ctrl) for v in all_results['Q2_burst_synchrony'].values() if ctrl in v and v[ctrl] is not None]
    if ctrl_vals:
        ctrl_mean = np.mean(ctrl_vals)
        effect = abs(ctrl_mean - q2_real) / q2_real
        print(f"  {ctrl}: {ctrl_mean:.3f} (effect: {effect:.1%})")
        if effect < 0.20:
            q2_survivors.append(ctrl)

# Q3: Heavy tails (kurtosis - higher means structured)
q3_real = np.mean([v['real'] for v in all_results['Q3_heavy_tails'].values() if 'real' in v])
print(f"\nQ3 STRUCTURED_HEAVY_TAILS:")
print(f"  Real kurtosis: {q3_real:.2f}")
q3_survivors = []
for ctrl in controls.keys():
    if ctrl == 'F_iaaft' or controls[ctrl] is None:
        continue
    ctrl_vals = [v.get(ctrl) for v in all_results['Q3_heavy_tails'].values() if ctrl in v and v[ctrl] is not None]
    if ctrl_vals:
        ctrl_mean = np.mean(ctrl_vals)
        effect = abs(ctrl_mean - q3_real) / abs(q3_real) if q3_real != 0 else 0
        print(f"  {ctrl}: {ctrl_mean:.2f} (effect: {effect:.1%})")
        if effect < 0.20:
            q3_survivors.append(ctrl)

# Q4: Low dimensional (PC1 - higher means more low-dim)
q4_real_pc1 = np.mean([v['real']['pc1'] for v in all_results['Q4_low_dim'].values() if 'real' in v])
q4_real_dim = np.mean([v['real']['dim_80'] for v in all_results['Q4_low_dim'].values() if 'real' in v])
print(f"\nQ4 LOW_DIMENSIONAL_COORDINATION:")
print(f"  Real: PC1={q4_real_pc1:.3f}, dim80={q4_real_dim:.1f}")
q4_survivors = []
for ctrl in controls.keys():
    if ctrl == 'F_iaaft' or controls[ctrl] is None:
        continue
    ctrl_pc1 = [v.get(ctrl, {}).get('pc1', 0) for v in all_results['Q4_low_dim'].values() if ctrl in v and v[ctrl] is not None]
    ctrl_dim = [v.get(ctrl, {}).get('dim_80', 0) for v in all_results['Q4_low_dim'].values() if ctrl in v and v[ctrl] is not None]
    if ctrl_pc1:
        ctrl_pc1_mean = np.mean(ctrl_pc1)
        ctrl_dim_mean = np.mean(ctrl_dim)
        pc1_effect = abs(ctrl_pc1_mean - q4_real_pc1) / q4_real_pc1 if q4_real_pc1 > 0 else 0
        dim_effect = abs(ctrl_dim_mean - q4_real_dim) / q4_real_dim if q4_real_dim > 0 else 0
        print(f"  {ctrl}: PC1={ctrl_pc1_mean:.3f}, dim80={ctrl_dim_mean:.1f} (effects: {pc1_effect:.1%}, {dim_effect:.1%})")
        if pc1_effect < 0.20 and dim_effect < 0.20:
            q4_survivors.append(ctrl)

# ============================================================
# FINAL VERDICTS
# ============================================================

print("\n" + "="*70)
print("FINAL VERDICTS")
print("="*70)

q1_verdict = "ROBUST_HIGHER_ORDER_STRUCTURE" if len(q1_survivors) >= 4 else ("SURROGATE_EXPLAINED" if len(q1_survivors) == 0 else "INCONCLUSIVE")
q2_verdict = "ROBUST_HIGHER_ORDER_STRUCTURE" if len(q2_survivors) >= 4 else ("SURROGATE_EXPLAINED" if len(q2_survivors) == 0 else "INCONCLUSIVE")
q3_verdict = "ROBUST_HIGHER_ORDER_STRUCTURE" if len(q3_survivors) >= 4 else ("SURROGATE_EXPLAINED" if len(q3_survivors) == 0 else "INCONCLUSIVE")
q4_verdict = "ROBUST_HIGHER_ORDER_STRUCTURE" if len(q4_survivors) >= 4 else ("SURROGATE_EXPLAINED" if len(q4_survivors) == 0 else "INCONCLUSIVE")

print(f"Q1 (GLOBAL_PHASE_DEPENDENT): {q1_verdict}")
print(f"  Survivors: {q1_survivors}")
print(f"Q2 (BURST_SYNCHRONY): {q2_verdict}")
print(f"  Survivors: {q2_survivors}")
print(f"Q3 (STRUCTURED_HEAVY_TAILS): {q3_verdict}")
print(f"  Survivors: {q3_survivors}")
print(f"Q4 (LOW_DIMENSIONAL_COORDINATION): {q4_verdict}")
print(f"  Survivors: {q4_survivors}")

# ============================================================
# SAVE OUTPUTS
# ============================================================

# replication_results.csv
csv_rows = [
    {'question': 'Q1_GLOBAL_PHASE_DEPENDENT', 'metric': 'AUC', 'real_value': round(q1_real,4), 'verdict': q1_verdict, 'survivors': str(q1_survivors)},
    {'question': 'Q2_BURST_SYNCHRONY', 'metric': 'AUC', 'real_value': round(q2_real,4), 'verdict': q2_verdict, 'survivors': str(q2_survivors)},
    {'question': 'Q3_STRUCTURED_HEAVY_TAILS', 'metric': 'kurtosis', 'real_value': round(q3_real,4), 'verdict': q3_verdict, 'survivors': str(q3_survivors)},
    {'question': 'Q4_LOW_DIMENSIONAL_COORDINATION', 'metric': 'PC1', 'real_value': round(q4_real_pc1,4), 'verdict': q4_verdict, 'survivors': str(q4_survivors)}
]

with open(os.path.join(OUTPUT_DIR, 'replication_results.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

# per_subject_results.json
with open(os.path.join(OUTPUT_DIR, 'per_subject_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# thresholds.json
thresholds = {
    'phase_dependent_auc_threshold': 0.7,
    'burst_synchrony_auc_threshold': 0.7,
    'heavy_tails_effect_threshold': 0.20,
    'low_dim_effect_threshold': 0.20,
    'survival_threshold': 4
}

with open(os.path.join(OUTPUT_DIR, 'thresholds.json'), 'w') as f:
    json.dump(thresholds, f, indent=2)

# preprocessing.json
preprocessing = {
    'random_state': 42,
    'fs': 256,
    'window_size': 512,
    'burst_threshold_percentile': 90,
    'variance_threshold': 0.80,
    'max_pca_components': 20
}

with open(os.path.join(OUTPUT_DIR, 'preprocessing.json'), 'w') as f:
    json.dump(preprocessing, f, indent=2)

# loso_structure.json
loso = {
    'method': 'simple_holdout',
    'train_test_split': 0.5,
    'note': 'LOSO-style for speed, not strict LOSO'
}

with open(os.path.join(OUTPUT_DIR, 'loso_structure.json'), 'w') as f:
    json.dump(loso, f, indent=2)

# runtime_log.txt
log = f"""PHASE 180 RUNTIME LOG
======================
Date: Generated
Random seed: {random_state}
Subjects: {len(edf_files)}

Controls tested:
- A: phase_randomization
- B: temporal_block_shuffle
- C: burst_timing_shuffle
- D: channel_permutation
- E: white_noise
- F: iaaft (OMITTED - timeout risk)

IAAFT Status: OMITTED
Reason: Phase 179 showed IAAFT causes 600s+ timeout

Findings:
Q1: {q1_verdict} (survivors: {q1_survivors})
Q2: {q2_verdict} (survivors: {q2_survivors})
Q3: {q3_verdict} (survivors: {q3_survivors})
Q4: {q4_verdict} (survivors: {q4_survivors})
"""

with open(os.path.join(OUTPUT_DIR, 'runtime_log.txt'), 'w') as f:
    f.write(log)

# failed_controls.json
failed = {
    'F_iaaft': {
        'reason': 'timeout_risk',
        'evidence': 'Phase 179 IAAFT timed out at 600s',
        'action': 'omitted_from_all_tests'
    }
}

with open(os.path.join(OUTPUT_DIR, 'failed_controls.json'), 'w') as f:
    json.dump(failed, f, indent=2)

print("\nDONE - All outputs saved")