import os
import json
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
import mne
from tqdm import tqdm

BASE = "/home/student/sgp-tribe3"
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase109_real_multisubject")
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(42)

EDFFILES = [
    os.path.join(BASE, "empirical_analysis/neural_networks/phase105_real_eeg_download/raw/EEGMMIDB.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf")
]

MIN_WINDOWS = 20
WINDOW_SEC = 10
SFREQ_TARGET = 128

def hjorth_params(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    act = np.var(x)
    mob = np.sqrt(np.var(dx) / (act + 1e-12))
    comp = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (mob + 1e-12)
    return act, mob, comp

def extract_features(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(x) < 1e-10:
        return None

    f, p = signal.welch(x, fs=SFREQ_TARGET, nperseg=min(256, len(x)))
    p = np.abs(p) + 1e-12

    bands = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,12), 'beta': (12,30)}
    band_powers = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        band_powers[name] = np.mean(p[mask]) if mask.any() else 0

    p_norm = p / (np.sum(p) + 1e-12)
    spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12))

    zcr = np.sum(np.diff(np.sign(x)) != 0) / len(x)

    try:
        lag1 = np.corrcoef(x[:-1], x[1:])[0,1]
    except:
        lag1 = 0

    act, mob, comp = hjorth_params(x)

    feats = {
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'skew': float(stats.skew(x)),
        'kurtosis': float(stats.kurtosis(x)),
        'hjorth_activity': float(act),
        'hjorth_mobility': float(mob),
        'hjorth_complexity': float(comp),
        'spectral_entropy': float(spec_ent),
        'bandpower_delta': float(band_powers['delta']),
        'bandpower_theta': float(band_powers['theta']),
        'bandpower_alpha': float(band_powers['alpha']),
        'bandpower_beta': float(band_powers['beta']),
        'lag1_autocorr': float(lag1),
        'zero_crossing_rate': float(zcr)
    }
    return feats

validated_files = []
excluded_files = []
subject_windows = {}
subject_id = 0

for fpath in EDFFILES:
    fname = os.path.basename(fpath)
    try:
        raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)

        nchan = raw.info['nchan']
        duration = raw.n_times / raw.info['sfreq']
        sfreq = raw.info['sfreq']

        if nchan < 1:
            excluded_files.append({'file': fname, 'reason': 'no_eeg_channels'})
            continue
        if duration < 60:
            excluded_files.append({'file': fname, 'reason': f'duration_{duration:.1f}s_lt_60s'})
            continue
        if sfreq < 64:
            excluded_files.append({'file': fname, 'reason': f'sfreq_{sfreq}_lt_64hz'})
            continue

        raw.resample(SFREQ_TARGET)
        data = raw.get_data()
        sig = data[0]

        if not np.isfinite(sig).all():
            excluded_files.append({'file': fname, 'reason': 'non_finite_values'})
            continue
        if np.var(sig) == 0:
            excluded_files.append({'file': fname, 'reason': 'zero_variance'})
            continue

        validated_files.append({'file': fname, 'nchan': nchan, 'duration': duration, 'sfreq': sfreq})

        win_samples = WINDOW_SEC * SFREQ_TARGET
        windows = []
        for start in range(0, len(sig) - win_samples, win_samples):
            seg = sig[start:start+win_samples]
            feat = extract_features(seg)
            if feat is not None:
                windows.append(feat)

        if len(windows) >= MIN_WINDOWS:
            subject_windows[subject_id] = windows
            print(f"Subject {subject_id} ({fname}): {len(windows)} windows")
            subject_id += 1
        else:
            excluded_files.append({'file': fname, 'reason': f'only_{len(windows)}_windows_lt_{MIN_WINDOWS}'})

    except Exception as e:
        excluded_files.append({'file': fname, 'reason': str(e)})

with open(os.path.join(OUTDIR, 'validated_files.json'), 'w') as f:
    json.dump(validated_files, f, indent=2)

with open(os.path.join(OUTDIR, 'excluded_files.json'), 'w') as f:
    json.dump(excluded_files, f, indent=2)

print(f"\nValidated subjects: {len(subject_windows)}")

if len(subject_windows) < 5:
    result = {
        'subjects_used': len(subject_windows),
        'windows_total': sum(len(v) for v in subject_windows.values()),
        'mean_auc': None,
        'perm_auc': None,
        'shuffle_auc': None,
        'phase_auc': None,
        'cov_auc': None,
        'effect_size': None,
        'verdict': 'INSUFFICIENT_REAL_SUBJECTS'
    }
    with open(os.path.join(OUTDIR, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit()

X, y, groups = [], [], []

for sid, wins in subject_windows.items():
    for w in wins:
        X.append([w[k] for k in sorted(w.keys())])
        y.append(sid)
        groups.append(sid)

X = np.array(X)
y = np.array(y)
g = np.array(groups)

print(f"Total: {len(X)} windows, {len(np.unique(g))} subjects")

feature_names = sorted(subject_windows[0][0].keys())
pd.DataFrame(X, columns=feature_names).to_csv(os.path.join(OUTDIR, 'window_features.csv'), index=False)

logo = LeaveOneGroupOut()

real_aucs = []
perm_aucs = []
shuffle_aucs = []
phase_aucs = []
cov_aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, g)):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    if len(np.unique(ytr)) < 2:
        continue

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]
    real_aucs.append(roc_auc_score(yte, prob))

    rng = np.random.default_rng(42 + fold_idx)

    yp = rng.permutation(ytr)
    clf_perm = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf_perm.fit(Xtr, yp)
    prob_perm = clf_perm.predict_proba(Xte)[:,1]
    perm_aucs.append(roc_auc_score(yte, prob_perm))

    Xshuf = Xtr.copy()
    for i in range(Xshuf.shape[1]):
        Xshuf[:, i] = rng.permutation(Xshuf[:, i])
    clf_shuf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf_shuf.fit(Xshuf, ytr)
    prob_shuf = clf_shuf.predict_proba(Xte)[:,1]
    shuffle_aucs.append(roc_auc_score(yte, prob_shuf))

    Xphase = Xtr.copy()
    for i in range(Xphase.shape[1]):
        x = Xphase[:, i]
        fft = np.fft.fft(x)
        phase = rng.uniform(-np.pi, np.pi, len(x))
        Xphase[:, i] = np.real(np.fft.ifft(fft * np.exp(1j * phase)))
    clf_phase = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf_phase.fit(Xphase, ytr)
    prob_phase = clf_phase.predict_proba(Xte)[:,1]
    phase_aucs.append(roc_auc_score(yte, prob_phase))

    Xcov = Xtr.copy()
    cov = np.cov(Xtr.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.abs(eigvals)
    Xcov = Xtr @ eigvecs @ np.diag(np.sqrt(eigvals + 1e-12)) @ eigvecs.T
    clf_cov = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf_cov.fit(Xcov, ytr)
    prob_cov = clf_cov.predict_proba(Xte)[:,1]
    cov_aucs.append(roc_auc_score(yte, prob_cov))

real_auc = float(np.mean(real_aucs))
perm_auc = float(np.mean(perm_aucs))
shuffle_auc = float(np.mean(shuffle_aucs))
phase_auc = float(np.mean(phase_aucs))
cov_auc = float(np.mean(cov_aucs))
effect = real_auc - perm_auc

verdict = "REAL_CROSS_SUBJECT_SIGNAL" if (real_auc > 0.75 and perm_auc < 0.60 and effect > 0.15) else "NO_ROBUST_SIGNAL"

result = {
    'subjects_used': len(subject_windows),
    'windows_total': len(X),
    'mean_auc': real_auc,
    'perm_auc': perm_auc,
    'shuffle_auc': shuffle_auc,
    'phase_auc': phase_auc,
    'cov_auc': cov_auc,
    'effect_size': effect,
    'verdict': verdict
}

with open(os.path.join(OUTDIR, 'results.json'), 'w') as f:
    json.dump(result, f, indent=2)

pd.DataFrame([result]).to_csv(os.path.join(OUTDIR, 'summary.csv'), index=False)

qc_summary = {
    'validated_files': len(validated_files),
    'excluded_files': len(excluded_files),
    'subjects_after_qc': len(subject_windows)
}
with open(os.path.join(OUTDIR, 'qc_summary.json'), 'w') as f:
    json.dump(qc_summary, f, indent=2)

print("\nPHASE 109 RESULTS")
print(json.dumps(result, indent=2))