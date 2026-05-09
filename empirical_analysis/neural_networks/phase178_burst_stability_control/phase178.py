import os, json, numpy as np, mne
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'

def phase_randomize(x):
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    rand_phase = np.random.uniform(0, 2*np.pi, len(fft))
    return np.fft.irfft(mag * np.exp(1j * rand_phase), n=len(x))

def burst_pca_analysis(data):
    analytic = hilbert(data, axis=1)
    sync = np.abs(np.mean(np.exp(1j * np.angle(analytic)), axis=0))
    thresh = np.percentile(sync, 90)
    burst_mask = sync > thresh
    
    if np.sum(burst_mask) < 50:
        return {'pc1': 0.0, 'dim_80': 0}
    
    burst_data = data[:, burst_mask].T
    pca = PCA(n_components=min(20, burst_data.shape[1]))
    pca.fit(burst_data)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pc1_var = pca.explained_variance_ratio_[0]
    n_80 = np.searchsorted(cumvar, 0.80) + 1
    
    return {'pc1': float(pc1_var), 'dim_80': int(n_80)}

print('PHASE 178 - STABILITY CONTROLS')

edf_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.edf')][:4]
results = []

for fname in edf_files:
    try:
        raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, fname), preload=True, verbose=False)
        data = raw.get_data()
        
        # Control 1: Phase-randomized surrogate
        sur_data = np.array([phase_randomize(ch) for ch in data])
        
        # Control 2: Channel order shuffle (test if structure is channel-specific)
        chan_shuffled = data.copy()
        np.random.seed(42)
        np.random.shuffle(chan_shuffled)
        
        real = burst_pca_analysis(data)
        sur = burst_pca_analysis(sur_data)
        chan = burst_pca_analysis(chan_shuffled)
        
        results.append({'file': fname,
            'real_pc1': round(real['pc1'],3), 'real_dim': real['dim_80'],
            'sur_pc1': round(sur['pc1'],3), 'sur_dim': sur['dim_80'],
            'chan_pc1': round(chan['pc1'],3), 'chan_dim': chan['dim_80']})
        print(f'{fname}: R pc1={real["pc1"]:.3f} dim={real["dim_80"]} S pc1={sur["pc1"]:.3f} dim={sur["dim_80"]} C pc1={chan["pc1"]:.3f} dim={chan["dim_80"]}')
    except Exception as e:
        print(f'FAIL {fname}: {e}')

pc1_real = np.mean([r['real_pc1'] for r in results])
pc1_sur = np.mean([r['sur_pc1'] for r in results])
pc1_chan = np.mean([r['chan_pc1'] for r in results])
dim_real = np.mean([r['real_dim'] for r in results])
dim_sur = np.mean([r['sur_dim'] for r in results])
dim_chan = np.mean([r['chan_dim'] for r in results])

print(f'\nAGG: R pc1={pc1_real:.3f} dim={dim_real:.1f} S pc1={pc1_sur:.3f} dim={dim_sur:.1f} C pc1={pc1_chan:.3f} dim={dim_chan:.1f}')

verdict = 'STABLE_LOW_DIMENSIONAL_COORDINATION' if (dim_sur > dim_real and dim_chan > dim_real and pc1_sur < pc1_real and pc1_chan < pc1_real) else 'CONTROL_SENSITIVE'

out = {'subjects': len(results), 'pc1_real': round(pc1_real,3), 'pc1_sur': round(pc1_sur,3), 'pc1_chan': round(pc1_chan,3), 'dim_real': round(dim_real,1), 'dim_sur': round(dim_sur,1), 'dim_chan': round(dim_chan,1), 'verdict': verdict, 'details': results}
with open('phase178_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nVERDICT: {verdict}')