#!/usr/bin/env python3
"""PHASE 183 - OPTIMIZED VERSION"""

import os, json, numpy as np, mne
from scipy.signal import hilbert
import time, warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
BURST_THRESHOLD = 90
EDGE_THRESHOLD = 80

DATA = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
OUT = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase183_global_modular_burst'

np.random.seed(RANDOM_STATE)

# Controls
def A(x):
    r = np.zeros_like(x)
    for i in range(x.shape[0]):
        fft = np.fft.rfft(x[i])
        r[i] = np.fft.irfft(np.abs(fft) * np.exp(1j * np.random.uniform(0, 6.28, len(fft))), n=len(x[i]))
    return r

def B(x):
    return np.roll(x, np.random.randint(0, x.shape[1]), axis=1)

def C(x):
    s = np.abs(np.mean(np.exp(1j * np.angle(hilbert(x, axis=1))), axis=0))
    m = s > np.percentile(s, BURST_THRESHOLD)
    idx = np.where(m)[0]
    if len(idx) < 50: return x
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(idx)
    y = x.copy()
    for j, k in zip(idx, np.where(m)[0]):
        if j < x.shape[1]:
            y[:, k] = x[:, j]
    return y

def D(x):
    y = x.copy()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(y)
    return y

def E(x):
    r = np.zeros_like(x)
    for i in range(x.shape[0]):
        fft = np.fft.rfft(x[i])
        np.random.seed(RANDOM_STATE + i)
        r[i] = np.fft.irfft(np.abs(fft) * np.exp(1j * np.random.uniform(-3.14, 3.14, len(fft))), n=len(x[i]))
    return r

def F(x, max_time=30):
    start = time.time()
    r = np.zeros_like(x)
    for i in range(x.shape[0]):
        if time.time() - start > max_time:
            raise TimeoutError("IAAFT timeout")
        sig = x[i]
        orig_p = np.abs(np.fft.fft(sig)) ** 2
        s = np.random.permutation(sig)
        for _ in range(2):
            f = np.fft.fft(s)
            s = np.real(np.fft.ifft(np.sqrt(orig_p) * np.exp(1j * np.angle(f))))
        r[i] = s
    return r

def network_metrics(data):
    a = hilbert(data, axis=1)
    sync = np.abs(np.mean(np.exp(1j * np.angle(a)), axis=0))
    thresh = np.percentile(sync, BURST_THRESHOLD)
    mask = sync > thresh
    
    if np.sum(mask) < 50:
        return None
    
    ba = a[:, mask]
    n = ba.shape[0]
    sm = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                pd = np.angle(ba[i]) - np.angle(ba[j])
                sm[i, j] = np.abs(np.mean(np.exp(1j * pd)))
    
    threshold = np.percentile(sm[np.triu_indices(n, k=1)], EDGE_THRESHOLD)
    adj = (sm > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    
    deg = np.sum(adj, axis=1)
    m = np.sum(deg) / 2
    
    # Modules
    vis = np.zeros(n, dtype=bool)
    mods = 0
    for i in range(n):
        if not vis[i] and deg[i] > 0:
            queue = [i]
            vis[i] = True
            while queue:
                node = queue.pop(0)
                neighbors = np.where(adj[node] > 0)[0]
                for nb in neighbors:
                    if not vis[nb]:
                        vis[nb] = True
                        queue.append(nb)
            mods += 1
    
    mod_sizes = []
    for m_id in range(mods):
        mod_sizes.append(np.sum(~vis))
    largest_mod = max(mod_sizes) / n if mod_sizes else 0
    
    # Efficiency
    conn_pairs = np.sum(adj > 0) / 2
    max_possible = n * (n - 1) / 2
    efficiency = conn_pairs / max_possible if max_possible > 0 else 0
    
    # Q
    Q = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                Aij = adj[i, j]
                Q += (Aij - (deg[i] * deg[j]) / (2 * m)) / (2 * m) if m > 0 else 0
    
    return {'modularity_q': float(Q), 'n_modules': int(mods), 'largest_module': float(largest_mod), 'global_efficiency': float(efficiency)}

controls = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F}
ctrl_names = {'A': 'phase_random', 'B': 'temporal_shift', 'C': 'burst_timing', 'D': 'channel_perm', 'E': 'spectrum_noise', 'F': 'iaaft'}

print("PHASE 183 - GLOBAL vs MODULAR")

files = [f for f in os.listdir(DATA) if f.endswith('.edf')][:4]
results = {}
runtime_log = {'phase': 183, 'controls': {}}

for f in files:
    print(f"--- {f} ---")
    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA, f), preload=True, verbose=False)
        d = raw.get_data()
        r = {'file': f}
        
        for code, fn in controls.items():
            start = time.time()
            runtime_log['controls'][code] = {'name': ctrl_names[code]}
            try:
                ctrl_data = fn(d)
                m = network_metrics(ctrl_data)
                r[code] = m if m else None
                runtime_log['controls'][code]['status'] = 'success'
                print(f"  {code}: {time.time()-start:.1f}s")
            except Exception as e:
                r[code] = None
                runtime_log['controls'][code]['status'] = 'failed'
                runtime_log['controls'][code]['error'] = str(e)
                print(f"  {code}: FAIL {e}")
        
        results[f] = r
    except Exception as e:
        print(f"FAIL {f}: {e}")

# Aggregate
print("\nAGGREGATE")
metrics = ['modularity_q', 'n_modules', 'largest_module', 'global_efficiency']
agg = {}
for ctrl in ['A', 'B', 'C', 'D', 'E', 'F']:
    agg[ctrl] = {}
    for m in metrics:
        vals = [results[f].get(ctrl, {}).get(m) for f in results if results[f].get(ctrl)]
        agg[ctrl][m] = np.nanmean([v for v in vals if v is not None]) if vals else None
        print(f"{ctrl} {m}: {agg[ctrl][m]:.3f}" if agg[ctrl][m] else f"{ctrl} {m}: N/A")

# Verdict
real_q = agg.get('A', {}).get('modularity_q')  # Use A as control baseline
surviving = []

print("\nVERDICT")

# Save outputs
with open(f'{OUT}/phase183_results.json', 'w') as out:
    json.dump({'phase': 183, 'results': results, 'aggregate': agg}, out, indent=2)

with open(f'{OUT}/runtime_log.json', 'w') as out:
    json.dump(runtime_log, out, indent=2)

print("DONE")