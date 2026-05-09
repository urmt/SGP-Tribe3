import os
import json
import numpy as np
import mne
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

DATA_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded'
WINDOW_SEC, STEP_SEC = 5, 2

def features(seg):
    phase = np.angle(hilbert(seg, axis=1))
    sync = np.abs(np.mean(np.exp(1j*phase), axis=0)).mean()
    power = np.mean(seg**2)
    var = np.var(seg)
    return [sync, power, var]

print('PHASE 128 - SPATIAL SCRAMBLE TEST')

edf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.edf')]

all_X, all_y, groups = [], [], []
sid = 0

for fname in edf_files:
    try:
        fpath = os.path.join(DATA_DIR, fname)
        raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        data = raw.get_data()
        fs = int(raw.info['sfreq'])
        
        w, step = WINDOW_SEC*fs, STEP_SEC*fs
        
        feats_r = []
        feats_s = []
        
        for start in range(0, data.shape[1]-w, step):
            seg = data[:, start:start+w]
            feats_r.append(features(seg))
            scram_data = seg.copy()
            np.random.shuffle(scram_data)
            feats_s.append(features(scram_data))
        
        min_n = min(len(feats_r), len(feats_s))
        feats_r = feats_r[:min_n]
        feats_s = feats_s[:min_n]
        
        X_subj = np.array(feats_r + feats_s)
        y_subj = np.array([1]*min_n + [0]*min_n)
        
        all_X.append(X_subj)
        all_y.append(y_subj)
        groups.extend([sid] * len(y_subj))
        sid += 1
        
        print(f'{fname}: {min_n} samples')
    except Exception as e:
        print(f'FAIL {fname}: {e}')

if all_X:
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(groups)
    
    logo = LeaveOneGroupOut()
    aucs = []
    
    for tr, te in logo.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:,1]))
    
    mean_auc = float(np.mean(aucs))
    effect = mean_auc - 0.5
    verdict = 'SPATIAL_STRUCTURE_PRESENT' if mean_auc > 0.75 and effect > 0.15 else 'SPATIAL_STRUCTURE_WEAK'
    
    results = {'subjects': sid, 'mean_auc': round(mean_auc, 3), 'effect': round(effect, 3), 'verdict': verdict}
else:
    results = {'verdict': 'NO_DATA'}

with open('phase128_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))