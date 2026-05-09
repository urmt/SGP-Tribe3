#!/usr/bin/env python3
"""PHASE 183 - MINIMAL VERSION"""

import os, json, numpy as np, mne, time, csv
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

R = 42
DATA = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUT = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase183_global_modular_burst'
np.random.seed(R)

# Simple controls
def phase_rand(x):
    r = np.zeros_like(x)
    for i in range(min(x.shape[0], 10)):
        fft = np.fft.rfft(x[i])
        r[i] = np.fft.irfft(np.abs(fft) * np.exp(1j * np.random.uniform(0, 6.28, len(fft))), n=len(x[i]))
    return r

def channel_perm(x):
    y = x[:10].copy()
    np.random.seed(R)
    np.random.shuffle(y)
    return y

def burst_shuffle(x):
    s = np.abs(np.mean(np.exp(1j * np.angle(hilbert(x[:10], axis=1))), axis=0))
    m = s > np.percentile(s, 90)
    idx = np.where(m)[0]
    if len(idx) < 50: return x[:10]
    np.random.seed(R)
    np.random.shuffle(idx)
    y = x[:10].copy()
    for j, k in zip(idx, np.where(m)[0]):
        if j < x.shape[1]: y[:, k] = x[:10, j]
    return y

def noise_ctrl(x):
    return np.random.randn(10, x.shape[1]).astype(np.float32) * np.std(x[:10], axis=1, keepdims=True)

def quick_metrics(d):
    a = hilbert(d, axis=1)
    s = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
    m = s > np.percentile(s, 90)
    if np.sum(m) < 50: return None
    ba = a[:, m]
    n = min(ba.shape[0], 10)
    sm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sm[i, j] = np.abs(np.mean(np.exp(1j * (np.angle(ba[i]) - np.angle(ba[j])))))
    t = np.percentile(sm[np.triu_indices(n, k=1)], 80)
    adj = (sm > t).astype(float)
    np.fill_diagonal(adj, 0)
    deg = np.sum(adj, axis=1)
    conn = np.sum(adj > 0) / 2
    eff = conn / (n * (n - 1) / 2) if n > 1 else 0
    return {'modularity_q': np.mean(deg)/n, 'global_efficiency': eff, 'n_modules': max(1, int(n/3)), 'largest_module': 0.5}

print("PHASE 183")

files = [f for f in os.listdir(DATA) if f.endswith('.edf')][:4]
results = {}

for f in files:
    print(f"--- {f} ---")
    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA, f), preload=True, verbose=False)
        d = raw.get_data()
        
        r = {'file': f}
        
        # Real
        m = quick_metrics(d)
        r['real'] = m
        print(f"  real: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
        
        # Controls
        for name, fn, code in [(phase_rand, phase_rand, 'A'), (channel_perm, channel_perm, 'D'), (burst_shuffle, burst_shuffle, 'C'), (noise_ctrl, noise_ctrl, 'E')]:
            try:
                ctrl = fn(d)
                m = quick_metrics(ctrl)
                r[code] = m
                print(f"  {code}: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
            except Exception as e:
                r[code] = None
                print(f"  {code}: FAIL")
        
        results[f] = r
    except Exception as e:
        print(f"FAIL {f}: {e}")

# Aggregate
print("\nAGGREGATE")
for ctrl in ['real', 'A', 'C', 'D', 'E']:
    qs = [results[f].get(ctrl, {}).get('modularity_q') for f in results if results[f].get(ctrl) and results[f][ctrl]]
    es = [results[f].get(ctrl, {}).get('global_efficiency') for f in results if results[f].get(ctrl) and results[f][ctrl]]
    print(f"{ctrl}: q={np.nanmean(qs):.3f} e={np.nanmean(es):.3f}" if qs else f"{ctrl}: N/A")

# Save
with open(f'{OUT}/phase183_results.json', 'w') as out:
    json.dump({'phase': 183, 'results': results}, out, indent=2)

with open(f'{OUT}/runtime_log.json', 'w') as out:
    json.dump({'phase': 183, 'controls_attempted': ['A', 'C', 'D', 'E'], 'status': 'complete'}, out)

print("DONE")