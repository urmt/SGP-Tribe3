#!/usr/bin/env python3
"""PHASE 180 - MINIMAL REPLICATION"""

import os, json, numpy as np, mne, csv
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

DATA = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUT = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase180_replication_audit'

np.random.seed(42)
files = [f for f in os.listdir(DATA) if f.endswith('.edf')][:4]

print("PHASE 180 - Testing 4 key findings with 4 controls")

results = {}

for f in files:
    print(f"Loading {f}...")
    raw = mne.io.read_raw_edf(os.path.join(DATA, f), preload=True, verbose=False)
    d = raw.get_data()
    
    # Controls
    controls = {}
    controls['real'] = d
    
    # A: phase random
    pr = np.zeros_like(d)
    for i in range(d.shape[0]):
        fft = np.fft.rfft(d[i])
        pr[i] = np.fft.irfft(np.abs(fft) * np.exp(1j * np.random.uniform(0, 6.28, len(fft))), n=len(d[i]))
    controls['A_pr'] = pr
    
    # D: channel perm
    cp = d.copy()
    np.random.seed(42)
    np.random.shuffle(cp)
    controls['D_cp'] = cp
    
    # C: burst timing
    sync = np.abs(np.mean(np.exp(1j * np.angle(hilbert(d, axis=1))), axis=0))
    mask = sync > np.percentile(sync, 90)
    idx = np.where(mask)[0]
    if len(idx) >= 50:
        np.random.seed(42)
        np.random.shuffle(idx)
        bt = d.copy()
        for j, k in zip(idx, np.where(mask)[0]):
            if j < d.shape[1]:
                bt[:, k] = d[:, j]
        controls['C_bt'] = bt
    else:
        controls['C_bt'] = d
    
    # E: white noise
    wn = np.random.randn(*d.shape) * np.std(d, axis=1, keepdims=True)
    controls['E_wn'] = wn
    
    # Tests
    r = {'file': f}
    for name, data in controls.items():
        try:
            # Kurtosis (Q3)
            k = kurtosis(data[0, :5000])
            
            # Low dim (Q4)
            s = np.abs(np.mean(np.exp(1j * np.angle(hilbert(data, axis=1))), axis=0))
            m = s > np.percentile(s, 90)
            if np.sum(m) >= 50:
                bd = data[:, m].T
                pca = PCA(n_components=min(20, bd.shape[1]), random_state=42)
                pca.fit(bd)
                pc1 = pca.explained_variance_ratio_[0]
                dim = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1
            else:
                pc1, dim = 0, 0
                
            r[name] = {'k': float(k), 'pc1': float(pc1), 'dim': int(dim)}
        except Exception as e:
            r[name] = None
            print(f"  {name}: FAIL {e}")
    
    results[f] = r

# Aggregate
print("\nAGGREGATE")
ag = {}
for metric in ['k', 'pc1', 'dim']:
    ag[metric] = {}
    for ctrl in ['real', 'A_pr', 'D_cp', 'C_bt', 'E_wn']:
        vals = [results[f].get(ctrl, {}).get(metric, np.nan) for f in results if results[f].get(ctrl)]
        ag[metric][ctrl] = np.nanmean(vals)
        print(f"{metric} {ctrl}: {np.nanmean(vals):.3f}")

# Verdicts
print("\nVERDICTS")

# Q3: Heavy tails
real_k = ag['k']['real']
s3 = []
for c in ['A_pr', 'D_cp', 'C_bt', 'E_wn']:
    e = abs(ag['k'][c] - real_k) / abs(real_k) if real_k != 0 else 0
    s3.append(c if e < 0.20 else None)
s3 = [x for x in s3 if x]
q3_v = "ROBUST" if len(s3) >= 3 else "SURROGATE_EXPLAINED"
print(f"Q3 (Heavy Tails): {q3_v} - survivors: {s3}")

# Q4: Low dim
real_pc1 = ag['pc1']['real']
real_dim = ag['dim']['real']
s4 = []
for c in ['A_pr', 'D_cp', 'C_bt', 'E_wn']:
    pc1_e = abs(ag['pc1'][c] - real_pc1) / real_pc1 if real_pc1 > 0 else 0
    dim_e = abs(ag['dim'][c] - real_dim) / real_dim if real_dim > 0 else 0
    s4.append(c if pc1_e < 0.20 and dim_e < 0.20 else None)
s4 = [x for x in s4 if x]
q4_v = "ROBUST" if len(s4) >= 3 else "SURROGATE_EXPLAINED"
print(f"Q4 (Low Dim): {q4_v} - survivors: {s4}")

# Save outputs
with open(f'{OUT}/replication_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['q', 'metric', 'real', 'verdict', 'survivors'])
    w.writeheader()
    w.writerow({'q': 'Q3', 'metric': 'kurtosis', 'real': f'{real_k:.1f}', 'verdict': q3_v, 'survivors': str(s3)})
    w.writerow({'q': 'Q4', 'metric': 'pc1_dim', 'real': f'{real_pc1:.3f}/{real_dim:.0f}', 'verdict': q4_v, 'survivors': str(s4)})

with open(f'{OUT}/per_subject_results.json', 'w') as f:
    json.dump(results, f)

print("SAVED")