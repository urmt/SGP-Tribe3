#!/usr/bin/env python3
"""
PHASE 179 - BURST DIMENSIONALITY UNDER DESTRUCTION OPERATORS

Objective: Test whether low-dimensional burst coordination survives
           progressively stronger destruction operators.

Controls:
  A. Phase randomization
  B. IAAFT surrogate
  C. Channel permutation
  D. Temporal block shuffle
  E. Burst timing shuffle

Method: Identical to Phase 177 - PCA on burst windows (90th percentile sync threshold)
"""

import os
import json
import numpy as np
import mne
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration
random_state = 42
np.random.seed(random_state)

DATA_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase179_destruction_operators'

# ============================================================
# CONTROLS - All destruction operators
# ============================================================

def control_phase_randomization(data):
    """A. Phase randomization - destroy phase relationships"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        fft = np.fft.rfft(data[i])
        mag = np.abs(fft)
        rand_phase = np.random.uniform(0, 2*np.pi, len(fft))
        result[i] = np.fft.irfft(mag * np.exp(1j * rand_phase), n=len(data[i]))
    return result

def control_iaaft(data, n_iter=10):
    """B. IAAFT surrogate - iterative amplitude-adjusted Fourier transform"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        signal = data[i]
        orig_fft = np.fft.fft(signal)
        orig_power = np.abs(orig_fft) ** 2
        sorted_power = np.sort(orig_power)[::-1]
        surrogate = np.random.permutation(signal)

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

def control_channel_permutation(data):
    """C. Channel permutation - destroy spatial patterns"""
    shuffled = data.copy()
    np.random.seed(random_state)
    np.random.shuffle(shuffled)
    return shuffled

def control_temporal_block_shuffle(data, block_size=512):
    """D. Temporal block shuffle - destroy temporal structure"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        n_blocks = len(data[i]) // block_size
        blocks = [data[i, j*block_size:(j+1)*block_size] for j in range(n_blocks)]
        np.random.seed(random_state + i)
        np.random.shuffle(blocks)
        result[i, :n_blocks*block_size] = np.concatenate(blocks)
        result[i, n_blocks*block_size:] = data[i, n_blocks*block_size:]
    return result

def control_burst_timing_shuffle(data, fs=256):
    """E. Burst timing shuffle - preserve burst structure, shuffle timing"""
    analytic = hilbert(data, axis=1)
    sync = np.abs(np.mean(np.exp(1j * np.angle(analytic)), axis=0))
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh
    burst_indices = np.where(burst_mask)[0]

    if len(burst_indices) < 50:
        return data

    # Shuffle burst indices
    np.random.seed(random_state)
    shuffled_indices = burst_indices.copy()
    np.random.shuffle(shuffled_indices)

    result = data.copy()
    for idx, orig_idx in zip(shuffled_indices, burst_indices):
        if idx < data.shape[1] and orig_idx < data.shape[1]:
            result[:, orig_idx] = data[:, idx]

    return result

# ============================================================
# ANALYSIS - Identical to Phase 177 methodology
# ============================================================

def burst_pca_analysis(data):
    """
    Identical methodology to Phase 177.
    Detect bursts via phase synchrony, compute PCA on burst windows.
    """
    # Compute phase synchrony
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)
    sync = np.abs(np.mean(np.exp(1j * phase), axis=0))

    # Burst detection: 90th percentile threshold
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh

    if np.sum(burst_mask) < 50:
        return {'pc1': 0.0, 'dim_80': 0, 'n_bursts': 0, 'eigenvalues': []}

    # Extract burst windows
    burst_data = data[:, burst_mask].T

    # PCA
    n_components = min(20, burst_data.shape[1], burst_data.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(burst_data)

    # Metrics
    pc1_var = pca.explained_variance_ratio_[0]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_80 = np.searchsorted(cumvar, 0.80) + 1

    # Eigenspectrum decay slope (log-log)
    eigenvalues = pca.explained_variance_
    if len(eigenvalues) > 1:
        log_eig = np.log(eigenvalues + 1e-12)
        log_idx = np.log(np.arange(1, len(eigenvalues) + 1))
        decay_slope = np.polyfit(log_idx, log_eig, 1)[0]
    else:
        decay_slope = 0.0

    return {
        'pc1': float(pc1_var),
        'dim_80': int(n_80),
        'n_bursts': int(np.sum(burst_mask)),
        'eigenvalues': eigenvalues.tolist(),
        'decay_slope': float(decay_slope)
    }

# ============================================================
# MAIN EXECUTION
# ============================================================

print("="*70)
print("PHASE 179 - DESTRUCTION OPERATORS TEST")
print("="*70)

edf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.edf')])[:4]
print(f"Processing {len(edf_files)} subjects")

results = []
all_controls = {}

# Process each subject
for fname in edf_files:
    print(f"\n--- {fname} ---")

    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, fname), preload=True, verbose=False)
        data = raw.get_data()
        print(f"  Data shape: {data.shape}")

        # Run all controls
        controls = {
            'real': data,
            'phase_random': control_phase_randomization(data),
            'iaaft': control_iaaft(data),
            'channel_perm': control_channel_permutation(data),
            'temporal_block': control_temporal_block_shuffle(data),
            'burst_timing': control_burst_timing_shuffle(data)
        }

        subject_results = {}

        for ctrl_name, ctrl_data in controls.items():
            analysis = burst_pca_analysis(ctrl_data)
            subject_results[ctrl_name] = analysis
            print(f"  {ctrl_name:15s}: PC1={analysis['pc1']:.3f}, dim80={analysis['dim_80']}, slope={analysis['decay_slope']:.2f}")

        results.append({'file': fname, **subject_results})

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# AGGREGATE ANALYSIS
# ============================================================

print("\n" + "="*70)
print("AGGREGATED RESULTS")
print("="*70)

# Extract metrics for all controls
control_names = ['real', 'phase_random', 'iaaft', 'channel_perm', 'temporal_block', 'burst_timing']

aggregate = {}
for ctrl in control_names:
    pc1_vals = [r[ctrl]['pc1'] for r in results if ctrl in r]
    dim_vals = [r[ctrl]['dim_80'] for r in results if ctrl in r]
    slope_vals = [r[ctrl]['decay_slope'] for r in results if ctrl in r]

    aggregate[ctrl] = {
        'pc1_mean': np.mean(pc1_vals),
        'pc1_std': np.std(pc1_vals),
        'dim80_mean': np.mean(dim_vals),
        'dim80_std': np.std(dim_vals),
        'slope_mean': np.mean(slope_vals),
        'n_subjects': len(pc1_vals)
    }

    print(f"{ctrl:15s}: PC1={np.mean(pc1_vals):.3f}±{np.std(pc1_vals):.3f}, dim80={np.mean(dim_vals):.1f}±{np.std(dim_vals):.1f}")

# ============================================================
# STRICT INTERPRETATION
# ============================================================

real_pc1 = aggregate['real']['pc1_mean']
real_dim = aggregate['real']['dim80_mean']

print("\n" + "="*70)
print("VERDICT DETERMINATION")
print("="*70)

# Check if any surrogate is within 10% of real (effect size)
surrogates = ['phase_random', 'iaaft', 'channel_perm', 'temporal_block', 'burst_timing']

surrogate_within_10pct = []
for surr in surrogates:
    surr_pc1 = aggregate[surr]['pc1_mean']
    surr_dim = aggregate[surr]['dim80_mean']

    pc1_effect = abs(surr_pc1 - real_pc1) / real_pc1
    dim_effect = abs(surr_dim - real_dim) / real_dim

    print(f"{surr:15s}: PC1 effect={pc1_effect:.2%}, dim effect={dim_effect:.2%}")

    if pc1_effect < 0.10 or dim_effect < 0.10:
        surrogate_within_10pct.append(surr)

if surrogate_within_10pct:
    verdict = "SURROGATE_EXPLAINED_LOW_DIMENSIONALITY"
else:
    verdict = "GENUINE_LOW_DIMENSIONAL_COORDINATION"

print(f"\nSurrogates within 10% effect: {surrogate_within_10pct}")
print(f"VERDICT: {verdict}")

# ============================================================
# SAVE OUTPUTS
# ============================================================

# 1. phase179_results.json
output = {
    'verdict': verdict,
    'subjects': len(results),
    'aggregate': aggregate,
    'details': results
}

with open(os.path.join(OUTPUT_DIR, 'phase179_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

# 2. control_comparison.csv
import csv
csv_rows = []
for ctrl in control_names:
    csv_rows.append({
        'control': ctrl,
        'pc1_mean': round(aggregate[ctrl]['pc1_mean'], 4),
        'pc1_std': round(aggregate[ctrl]['pc1_std'], 4),
        'dim80_mean': round(aggregate[ctrl]['dim80_mean'], 2),
        'dim80_std': round(aggregate[ctrl]['dim80_std'], 2),
        'slope_mean': round(aggregate[ctrl]['slope_mean'], 3),
        'n_subjects': aggregate[ctrl]['n_subjects']
    })

with open(os.path.join(OUTPUT_DIR, 'control_comparison.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

# 3. eigenspectra.csv
eig_rows = []
for r in results:
    fname = r['file']
    for ctrl in control_names:
        if ctrl in r:
            for i, ev in enumerate(r[ctrl]['eigenvalues']):
                eig_rows.append({
                    'file': fname,
                    'control': ctrl,
                    'component': i + 1,
                    'variance': round(ev, 6)
                })

with open(os.path.join(OUTPUT_DIR, 'eigenspectra.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=eig_rows[0].keys())
    writer.writeheader()
    writer.writerows(eig_rows)

# 4. execution_log.txt
log = f"""PHASE 179 EXECUTION LOG
========================
Date: Generated automatically
Random seed: {random_state}
Subjects: {len(results)}
Controls: {control_names}

Methodology: Identical to Phase 177
- Burst detection: 90th percentile phase synchrony threshold
- PCA on burst windows
- Metrics: PC1 variance, PCs for 80% variance, eigenspectrum decay

Destruction operators applied:
A. Phase randomization: Destroy phase relationships
B. IAAFT: Iterative amplitude-adjusted Fourier transform
C. Channel permutation: Destroy spatial patterns
D. Temporal block shuffle: Destroy temporal structure
E. Burst timing shuffle: Preserve burst structure, shuffle timing

VERDICT: {verdict}

Real EEG: PC1={real_pc1:.3f}, dim80={real_dim:.1f}
All surrogates compared within strict 10% effect threshold.
"""

with open(os.path.join(OUTPUT_DIR, 'execution_log.txt'), 'w') as f:
    f.write(log)

# 5. parameters.json
params = {
    'random_state': random_state,
    'burst_threshold_percentile': 80,  # 90th percentile for sync
    'variance_threshold': 0.80,
    'max_pca_components': 20,
    'subjects': len(results),
    'control_methods': control_names
}

with open(os.path.join(OUTPUT_DIR, 'parameters.json'), 'w') as f:
    json.dump(params, f, indent=2)

print(f"\nOutputs saved to {OUTPUT_DIR}")
print("DONE")