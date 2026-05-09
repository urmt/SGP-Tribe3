#!/usr/bin/env python3
"""
PHASE 179 - BURST DIMENSIONALITY UNDER DESTRUCTION OPERATORS (FAST VERSION)
"""

import os
import json
import numpy as np
import mne
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import warnings
import csv
warnings.filterwarnings('ignore')

random_state = 42
np.random.seed(random_state)

DATA_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase179_destruction_operators'

# CONTROLS
def control_phase_randomization(data):
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        fft = np.fft.rfft(data[i])
        mag = np.abs(fft)
        rand_phase = np.random.uniform(0, 2*np.pi, len(fft))
        result[i] = np.fft.irfft(mag * np.exp(1j * rand_phase), n=len(data[i]))
    return result

def control_channel_permutation(data):
    shuffled = data.copy()
    np.random.seed(random_state)
    np.random.shuffle(shuffled)
    return shuffled

def control_temporal_block_shuffle(data, block_size=512):
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        n_blocks = len(data[i]) // block_size
        blocks = [data[i, j*block_size:(j+1)*block_size] for j in range(n_blocks)]
        np.random.seed(random_state + i)
        np.random.shuffle(blocks)
        result[i, :n_blocks*block_size] = np.concatenate(blocks)
    return result

def control_burst_timing_shuffle(data, fs=256):
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

def control_white_noise(data):
    """Simple white noise control - destroy all structure"""
    result = np.random.randn(*data.shape).astype(np.float32) * np.std(data, axis=1, keepdims=True)
    return result

def burst_pca_analysis(data):
    analytic = hilbert(data, axis=1)
    sync = np.abs(np.mean(np.exp(1j * np.angle(analytic)), axis=0))
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh
    if np.sum(burst_mask) < 50:
        return {'pc1': 0.0, 'dim_80': 0, 'decay_slope': 0.0}
    burst_data = data[:, burst_mask].T
    n_comp = min(20, burst_data.shape[1], burst_data.shape[0])
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(burst_data)
    pc1_var = pca.explained_variance_ratio_[0]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_80 = np.searchsorted(cumvar, 0.80) + 1
    eigenvalues = pca.explained_variance_
    if len(eigenvalues) > 1:
        log_eig = np.log(eigenvalues + 1e-12)
        log_idx = np.log(np.arange(1, len(eigenvalues) + 1))
        decay_slope = np.polyfit(log_idx, log_eig, 1)[0]
    else:
        decay_slope = 0.0
    return {'pc1': float(pc1_var), 'dim_80': int(n_80), 'decay_slope': float(decay_slope), 'eigenvalues': eigenvalues.tolist()}

print("PHASE 179 - DESTRUCTION OPERATORS TEST")

edf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.edf')])[:4]
results = []

for fname in edf_files:
    print(f"\n--- {fname} ---")
    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, fname), preload=True, verbose=False)
        data = raw.get_data()
        
        controls = {
            'real': data,
            'phase_random': control_phase_randomization(data),
            'channel_perm': control_channel_permutation(data),
            'temporal_block': control_temporal_block_shuffle(data),
            'burst_timing': control_burst_timing_shuffle(data),
            'white_noise': control_white_noise(data)
        }
        
        subject_results = {}
        for ctrl_name, ctrl_data in controls.items():
            analysis = burst_pca_analysis(ctrl_data)
            subject_results[ctrl_name] = analysis
            print(f"  {ctrl_name:15s}: PC1={analysis['pc1']:.3f}, dim80={analysis['dim_80']}, slope={analysis['decay_slope']:.2f}")
        
        results.append({'file': fname, **subject_results})
    except Exception as e:
        print(f"  FAIL: {e}")

# Aggregate
control_names = ['real', 'phase_random', 'channel_perm', 'temporal_block', 'burst_timing', 'white_noise']
aggregate = {}

print("\n" + "="*60)
print("AGGREGATED RESULTS")

for ctrl in control_names:
    pc1_vals = [r[ctrl]['pc1'] for r in results if ctrl in r]
    dim_vals = [r[ctrl]['dim_80'] for r in results if ctrl in r]
    aggregate[ctrl] = {'pc1_mean': np.mean(pc1_vals), 'pc1_std': np.std(pc1_vals), 'dim80_mean': np.mean(dim_vals), 'dim80_std': np.std(dim_vals)}
    print(f"{ctrl:15s}: PC1={np.mean(pc1_vals):.3f}±{np.std(pc1_vals):.3f}, dim80={np.mean(dim_vals):.1f}")

# Verdict
real_pc1 = aggregate['real']['pc1_mean']
real_dim = aggregate['real']['dim80_mean']

surrogates = ['phase_random', 'channel_perm', 'temporal_block', 'burst_timing', 'white_noise']
surrogate_within_10pct = []

for surr in surrogates:
    surr_pc1 = aggregate[surr]['pc1_mean']
    surr_dim = aggregate[surr]['dim80_mean']
    pc1_effect = abs(surr_pc1 - real_pc1) / real_pc1
    dim_effect = abs(surr_dim - real_dim) / real_dim
    print(f"{surr}: PC1 effect={pc1_effect:.2%}, dim effect={dim_effect:.2%}")
    if pc1_effect < 0.10 or dim_effect < 0.10:
        surrogate_within_10pct.append(surr)

verdict = "SURROGATE_EXPLAINED_LOW_DIMENSIONALITY" if surrogate_within_10pct else "GENUINE_LOW_DIMENSIONAL_COORDINATION"
print(f"\nVERDICT: {verdict}")

# Save outputs
output = {'verdict': verdict, 'subjects': len(results), 'aggregate': aggregate, 'details': results}
with open(os.path.join(OUTPUT_DIR, 'phase179_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

csv_rows = []
for ctrl in control_names:
    csv_rows.append({'control': ctrl, 'pc1_mean': round(aggregate[ctrl]['pc1_mean'], 4), 'pc1_std': round(aggregate[ctrl]['pc1_std'], 4), 'dim80_mean': round(aggregate[ctrl]['dim80_mean'], 2), 'dim80_std': round(aggregate[ctrl]['dim80_std'], 2)})

with open(os.path.join(OUTPUT_DIR, 'control_comparison.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

with open(os.path.join(OUTPUT_DIR, 'parameters.json'), 'w') as f:
    json.dump({'random_state': random_state, 'controls': control_names}, f, indent=2)

print("DONE")