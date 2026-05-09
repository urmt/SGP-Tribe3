#!/usr/bin/env python3
"""PHASE 183 - FIXED VERSION"""

import os, json, numpy as np, mne, time, csv
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

R = 42
DATA = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUT = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase183_global_modular_burst'
np.random.seed(R)

print("PHASE 183 - GLOBAL vs MODULAR")

files = [f for f in os.listdir(DATA) if f.endswith('.edf')][:4]
results = {}

for f in files:
    print(f"--- {f} ---")
    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA, f), preload=True, verbose=False)
        data = raw.get_data()
        n_ch = min(15, data.shape[0])  # Use first 15 channels
        
        # Limit data length for speed
        data = data[:n_ch, :100000]  # First 100k samples
        
        def get_metrics(d):
            try:
                a = hilbert(d, axis=1)
                s = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
                m = s > np.percentile(s, 90)
                if np.sum(m) < 50: return None
                ba = a[:, m]
                n = ba.shape[0]
                sm = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            pd = np.angle(ba[i]) - np.angle(ba[j])
                            sm[i, j] = np.abs(np.mean(np.exp(1j * pd)))
                th = np.percentile(sm[np.triu_indices(n, k=1)], 80)
                adj = (sm > th).astype(float)
                np.fill_diagonal(adj, 0)
                deg = np.sum(adj, axis=1)
                conn = np.sum(adj > 0) / 2
                eff = conn / (n * (n - 1) / 2) if n > 1 else 0
                mod = np.mean(deg) / n
                return {'modularity_q': float(mod), 'global_efficiency': float(eff)}
            except:
                return None
        
        # REAL
        r = {'file': f}
        m = get_metrics(data)
        r['real'] = m
        print(f"  real: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
        
        # A: Phase random
        pr = data.copy()
        for i in range(n_ch):
            fft = np.fft.rfft(pr[i])
            pr[i] = np.fft.irfft(np.abs(fft) * np.exp(1j * np.random.uniform(0, 6.28, len(fft))), n=len(pr[i]))
        m = get_metrics(pr)
        r['A'] = m
        print(f"  A: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
        
        # C: Burst timing
        a = hilbert(data, axis=1)
        s = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
        m = s > np.percentile(s, 90)
        idx = np.where(m)[0]
        if len(idx) >= 50:
            np.random.seed(R)
            np.random.shuffle(idx)
            bt = data.copy()
            for j, k in zip(idx, np.where(m)[0]):
                if j < data.shape[1]:
                    bt[:, k] = data[:, j]
            m = get_metrics(bt)
            r['C'] = m
            print(f"  C: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
        else:
            r['C'] = None
            print(f"  C: FAIL (no bursts)")
        
        # D: Channel perm
        cp = data.copy()
        np.random.seed(R)
        np.random.shuffle(cp)
        m = get_metrics(cp)
        r['D'] = m
        print(f"  D: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
        
        # E: White noise
        wn = np.random.randn(*data.shape) * np.std(data, axis=1, keepdims=True)
        m = get_metrics(wn)
        r['E'] = m
        print(f"  E: q={m['modularity_q']:.3f} e={m['global_efficiency']:.3f}")
        
        results[f] = r
    except Exception as e:
        print(f"FAIL {f}: {e}")

# Aggregate
print("\nAGGREGATE")
for ctrl in ['real', 'A', 'C', 'D', 'E']:
    qs = [results[f].get(ctrl, {}).get('modularity_q') for f in results if results[f].get(ctrl)]
    es = [results[f].get(ctrl, {}).get('global_efficiency') for f in results if results[f].get(ctrl)]
    print(f"{ctrl}: q={np.nanmean(qs):.3f} e={np.nanmean(es):.3f}" if qs else f"{ctrl}: N/A")

# Verdict
real_q = np.nanmean([results[f]['real']['modularity_q'] for f in results])
real_e = np.nanmean([results[f]['real']['global_efficiency'] for f in results])

surviving = []
for ctrl in ['A', 'C', 'D', 'E']:
    ctrl_q = np.nanmean([results[f].get(ctrl, {}).get('modularity_q', 0) for f in results])
    ctrl_e = np.nanmean([results[f].get(ctrl, {}).get('global_efficiency', 0) for f in results])
    if ctrl_q and ctrl_e:
        q_eff = abs(ctrl_q - real_q) / real_q if real_q > 0 else 0
        e_eff = abs(ctrl_e - real_e) / real_e if real_e > 0 else 0
        if q_eff > 0.15 or e_eff > 0.15:
            surviving.append(ctrl)

verdict = "GLOBAL_INTEGRATION_SURVIVES" if len(surviving) >= 3 else ("MIXED_OR_UNSTABLE" if surviving else "SURROGATE_EXPLAINED_NETWORK_STRUCTURE")
print(f"\nVERDICT: {verdict}")
print(f"Surviving: {surviving}")

# Save
with open(f'{OUT}/phase183_results.json', 'w') as out:
    json.dump({'phase': 183, 'verdict': verdict, 'results': results, 'surviving': surviving}, out, indent=2)

with open(f'{OUT}/runtime_log.json', 'w') as out:
    json.dump({'phase': 183, 'controls_attempted': ['A', 'C', 'D', 'E'], 'omitted': ['B', 'F']}, out)

print("DONE")